from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from itertools import chain
from PIL import Image
import asyncio
import aiofiles
from io import BytesIO

import pydicom
import cv2

try:
    from embed_explore import align_images_given_img
except Exception:
    def align_images_given_img(a, b):
        return a, b

exam_record = namedtuple('exam_foo',
                         ['eid', 'label',
                          "logit_yr1", "logit_yr2", "logit_yr3", "logit_yr4", "logit_yr5",
                          "year1_risk", "year2_risk", "year3_risk", "year4_risk", "year5_risk",
                          'l_cc_img', 'l_cc_path', 'r_cc_img', 'r_cc_path',
                          'l_mlo_img', 'l_mlo_path', 'r_mlo_img', 'r_mlo_path'])

class Mammo_CLIPMetadataset(Dataset):
    """
    Flexible dataset that consumes your dataframe directly.
    You can tell it which columns correspond to file path, exam id, laterality, view, and label.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 path_col='file',             # your image path column
                 exam_col='AccessionNumber',  # your exam id column
                 lat_col='laterality',        # your laterality column
                 view_col='view',             # your view (CC/MLO) column
                 label_col='cancer1yr_updated',
                 dataset="mayo",
                 allow_incomplete=False,
                 verbose=False,
                 transforms=None,
                 mode="training",
                 bu_path=None,
                 load_images_async=True,
                 align_images=True,
                 oversample_cancer_rate=None,
                 multiple_pairs_per_exam=False,
                 mean=0.3089279, std=0.25053555408335154):
        super().__init__()
        assert not (align_images and multiple_pairs_per_exam)

        # Map your columns to the expected names internally
        df_ = df.copy()
        df_['file_path'] = df_[path_col]
        df_['exam_id'] = df_[exam_col].astype(str)
        df_['laterality'] = df_[lat_col].astype(str).str.upper().str[0]
        df_['projection'] = df_[view_col].astype(str).str.upper()
        df_['view'] = df_[view_col].astype(str).str.upper()
        df_['cancer1yr_updated'] = df_[label_col].fillna(0).astype(int)

        self.df = df_
        self.dataset = dataset
        self.bu_path = bu_path
        self.transforms = transforms
        self.load_images_async = load_images_async
        self.align_images = align_images
        self.mode = mode
        self.mean = mean
        self.std = std
        self.multiple_pairs_per_exam = multiple_pairs_per_exam
        self.max_per_view = 4

        self.exams = []
        for eid, cur_exam in self.df.groupby('exam_id'):
            cur_exam = cur_exam.reset_index(drop=True)

            def rows(side, view):
                return cur_exam[(cur_exam['laterality'].eq(side)) & (cur_exam['projection'].eq(view))]

            if not self.multiple_pairs_per_exam:
                def pick(side, view):
                    r = rows(side, view)
                    return None if r.empty else r['file_path'].values[-1]

                l_cc = pick('L','CC'); r_cc = pick('R','CC')
                l_mlo = pick('L','MLO'); r_mlo = pick('R','MLO')

                complete = all([l_cc, r_cc, l_mlo, r_mlo])
                if not (allow_incomplete or complete):
                    if verbose: print(f"Skip incomplete exam {eid}")
                    continue

                label = 1 if (cur_exam['cancer1yr_updated'].values == 1).any() else 0
                label = torch.tensor(label, dtype=torch.long)

                # --- Risks only. Logits are not available -> hard-zero them to keep API stable ---
                def _get_first(cols, default=0.0):
                    # cols: list of possible column name variants
                    for col in cols:
                        if col in cur_exam.columns:
                            val = cur_exam[col].values[0]
                            if pd.notna(val):
                                return torch.tensor(val)
                    return torch.tensor(default)

                # Keep logits slots but set to 0.0 (we don't have logits)
                logit_yr1 = torch.tensor(0.0)
                logit_yr2 = torch.tensor(0.0)
                logit_yr3 = torch.tensor(0.0)
                logit_yr4 = torch.tensor(0.0)
                logit_yr5 = torch.tensor(0.0)

                # Risks: try both "1_year_risk" style and "year1_risk" style, fallback -> 0.0
                # year1_risk = _get_first(["1_year_risk", "year1_risk"], 0.0)
                # year2_risk = _get_first(["2_year_risk", "year2_risk"], 0.0)
                # year3_risk = _get_first(["3_year_risk", "year3_risk"], 0.0)
                # year4_risk = _get_first(["4_year_risk", "year4_risk"], 0.0)
                # year5_risk = _get_first(["5_year_risk", "year5_risk"], 0.0)

                year1_risk = cur_exam["1_year_risk"].values[0]
                year2_risk = cur_exam["2_year_risk"].values[0]
                year3_risk = cur_exam["3_year_risk"].values[0]
                year4_risk = cur_exam["4_year_risk"].values[0]
                year5_risk = cur_exam["5_year_risk"].values[0]

                self.exams.append(
                    exam_record(eid, label,
                                logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                                year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                                l_cc, None, r_cc, None, l_mlo, None, r_mlo, None)
                )
            else:
                # multiple pairs per exam: similar logic, omitted here for brevity
                pass

    # --- same load/process logic as before ---
    def load_img_path(self, path):
        return path if self.dataset.lower() == "mayo" else path  # extend BU logic if needed

    def process_img(self, pil_img):
        arr = np.array(pil_img)
        arr = arr.astype('float32')
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn)/(mx - mn)
        img = torch.tensor((arr - self.mean)/self.std, dtype=torch.float32)
        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3,1,1)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2,0,1)
        return img

    def load_img(self, path):
        img_path = self.load_img_path(path)
        img = Image.open(img_path).convert('RGB')
        return self.process_img(img)

    def __len__(self):
        return len(self.exams)

    def __getitem__(self, idx):
        exam = self.exams[idx]
        paths = exam[12::2]  # [l_cc,r_cc,l_mlo,r_mlo]
        imgs = [self.read_array(p) for p in paths]
        # imgs = [self.load_img(p) for p in paths]

        # Return order unchanged: eid, label, logits(zeros), risks, then (img,path)*4
        return ([
            exam[0],  # eid
            exam[1],  # label
            exam[2], exam[3], exam[4], exam[5], exam[6],          # logit_yr1..yr5 (zeros)
            exam[7], exam[8], exam[9], exam[10], exam[11]         # year1_risk..year5_risk
        ] + list(chain(*[(im, p) for im, p in zip(imgs, paths)])))

    def np_CountUpContinuingOnes(self, b_arr):
        # indice continuing zeros from left side.
        # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
        left = np.arange(len(b_arr))
        left[b_arr > 0] = 0
        left = np.maximum.accumulate(left)

        # from right side.
        # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
        rev_arr = b_arr[::-1]
        right = np.arange(len(rev_arr))
        right[rev_arr > 0] = 0
        right = np.maximum.accumulate(right)
        right = len(rev_arr) - 1 - right[::-1]

        return right - left - 1

    def np_ExtractBreast(self, img):
        img_copy = img.copy()
        img = np.where(img <= 40, 0, img)  # To detect backgrounds easily

        height, _ = img.shape

        # whether each col is non-constant or not
        y_a = height // 2 + int(height * 0.4)
        y_b = height // 2 - int(height * 0.4)
        b_arr = img[y_b:y_a].std(axis=0) != 0
        continuing_ones = self.np_CountUpContinuingOnes(b_arr)
        # longest should be the breast
        col_ind = np.where(continuing_ones == continuing_ones.max())[0]
        img = img[:, col_ind]

        # whether each row is non-constant or not
        _, width = img.shape
        x_a = width // 2 + int(width * 0.4)
        x_b = width // 2 - int(width * 0.4)
        b_arr = img[:, x_b:x_a].std(axis=1) != 0
        continuing_ones = self.np_CountUpContinuingOnes(b_arr)
        # longest should be the breast
        row_ind = np.where(continuing_ones == continuing_ones.max())[0]

        return img_copy[row_ind][:, col_ind]

    def read_array(self, dicom_path, voi_lut=False, fix_monochrome=True):
        dicom = pydicom.read_file(dicom_path)

        try:
            data = dicom.pixel_array
        except:
            print(f'error loading file - {dicom_path}')
        
        data = data[5:-5, 5:-5]

        data = pydicom.pixel_data_handlers.util.apply_voi_lut(data, dicom)

        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        data = (data - self.mean)/self.std

        if len(data.shape) == 3:
            img_id = data.shape[0]//2
            data = data[img_id, :, :]

        data = self.np_ExtractBreast(data)
        data = cv2.resize(data, (912, 1520), interpolation=cv2.INTER_AREA)

        data = np.stack((data,) * 3)
        data = np.transpose(data, (1, 2, 0))
        return data
