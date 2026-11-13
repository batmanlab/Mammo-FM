import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from mammo_sae.__main__ import main
from mammo_sae.core import SpatialSAE
from mammo_sae.impl.sae import MatryoshkaAutoencoder
from mammo_sae.localization import activation_box, average_precision


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(n_inputs=3,n_latents=3,activation='TopKReLU',tied=False,normalize=False,
                           latent_soft_cap=0.0,nesting_list=[1,3],relative_importance=[1.,1.])
        model = MatryoshkaAutoencoder(**self.config)
        with torch.no_grad():
            model.encoder.copy_(torch.eye(3)); model.decoder.copy_(torch.eye(3))
            model.pre_bias.zero_(); model.latent_bias.zero_()
        torch.save(dict(model_config=self.config,model=model.state_dict(),mean_center=torch.zeros(3),
                        scaling_factor=1.),self.root/'sae.pt')

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_select_import(self):
        r = self.root
        rows = []
        for i in range(8):
            for side in ['L','R']:
                rows.append(dict(patient_id='p%d'%i,exam_id='e%d'%i,id='%d%s'%(i,side),
                                 image_laterality=side,image_view_type='CC',image_path='/unused.png',
                                 mass=int(i<4 and side=='L')))
        pd.DataFrame(rows).to_csv(r/'source.csv',index=False)
        mapping = dict(patient_id='patient_id',exam_id='exam_id',image_id='id',
                       view=['image_laterality','image_view_type'],image_path='image_path',mass='mass')
        (r/'columns.json').write_text(json.dumps(mapping))
        main(['prepare-manifest','--input',str(r/'source.csv'),'--columns',str(r/'columns.json'),
              '--output',str(r/'images.csv')])
        main(['select','--manifest',str(r/'images.csv'),'--findings','mass','--per-finding','4',
              '--output',str(r/'selected')])
        selected=pd.read_csv(r/'selected/images.csv')
        self.assertEqual(len(selected),8)  # four affected-side images + two bilateral negatives
        old=pd.DataFrame(dict(exam_id=['e0']*4,view=['L_CC']*4,patch_h=[0,0,1,1],patch_w=[0,1,0,1],
                              spatial_dims=['(2, 2)']*4))
        old.to_csv(r/'old.csv',index=False)
        main(['import-metadata','--input',str(r/'old.csv'),'--manifest',str(r/'images.csv'),
              '--output',str(r/'metadata.csv')])
        converted=pd.read_csv(r/'metadata.csv')
        self.assertEqual(converted.patient_id.tolist(),['p0']*4)
        self.assertEqual(converted.patch_w.tolist(),[0,1,0,1])

    def test_import_metadata_rejects_empty_or_incomplete_inputs(self):
        from mammo_sae.data import import_metadata
        r = self.root
        pd.DataFrame(columns=['exam_id', 'view', 'patch_h', 'patch_w']).to_csv(r/'old.csv', index=False)
        pd.DataFrame([dict(patient_id='p', exam_id='e', image_id='i', view='L_CC',
                           image_path='/path/to/image.png')]).to_csv(r/'images.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            import_metadata(r/'old.csv', r/'images.csv', r/'output.csv')
        pd.DataFrame([dict(exam_id='e',view='L_CC',patch_h=0,patch_w=0,grid_h=1,grid_w=1)]).to_csv(r/'old.csv',index=False)
        pd.DataFrame([dict(exam_id='e',view='L_CC')]).to_csv(r/'images.csv',index=False)
        with self.assertRaisesRegex(ValueError, 'complete'):
            import_metadata(r/'old.csv', r/'images.csv', r/'output.csv')
        self.assertFalse((r/'output.csv').exists())

    def test_bu_mapping_disambiguates_exam_level_id(self):
        r=self.root
        pd.DataFrame([dict(patient_id='p',exam_id='e',id='exam',side=side,view='CC',
                           path='/old/controls/e/'+side+'CC.png',dataset='BU') for side in ['L','R']]).to_csv(r/'old.csv',index=False)
        mapping=dict(patient_id='patient_id',exam_id='exam_id',image_id=['id','side','view'],
                     view=['side','view'],image_path='path')
        (r/'map.json').write_text(json.dumps(mapping))
        main(['prepare-manifest','--input',str(r/'old.csv'),'--columns',str(r/'map.json'),'--dataset','bu',
              '--bu-image-root','/shared','--output',str(r/'new.csv')])
        f=pd.read_csv(r/'new.csv')
        self.assertEqual(f.image_id.nunique(),2)
        self.assertEqual(f.iloc[0].image_path,'/shared/controls/test_images_png/e/LCC.png')

    def test_associations_counts_patients_and_corrects_tests(self):
        r=self.root
        labels=[]; summary=[]
        for i in range(6):
            for exam in range(2):
                key=dict(patient_id='p%d'%i,exam_id='%d_%d'%(i,exam))
                labels.append(dict(key,mass=int(i<3)))
                summary.append(dict(key,neuron_id=0,max_activation=float(i<3)))
                summary.append(dict(key,neuron_id=1,max_activation=1.))
        pd.DataFrame(labels).to_csv(r/'labels.csv',index=False)
        pd.DataFrame(summary).to_csv(r/'summary.csv',index=False)
        main(['associations','--summary',str(r/'summary.csv'),'--labels',str(r/'labels.csv'),
              '--findings','mass','--output',str(r/'assoc')])
        result=pd.read_csv(r/'assoc/associations.csv')
        self.assertEqual(result.iloc[0].positive_patients,3)
        self.assertEqual(result.iloc[0].rate_difference,1)
        self.assertTrue((result.q_value >= result.p_value).all())

    def test_boxes_use_full_pixel_extent_and_duplicate_detections(self):
        result=activation_box(np.ones((2,3)),100,70,0.5,percentile=0)
        self.assertEqual([result[k] for k in ['x1','y1','x2','y2']],[0.,0.,100.,70.])
        key=dict(patient_id='p',exam_id='e',image_id='i')
        truth=pd.DataFrame([dict(key,x1=0,y1=0,x2=10,y2=10),dict(key,x1=20,y1=20,x2=30,y2=30)])
        pred=pd.DataFrame([dict(key,x1=0,y1=0,x2=10,y2=10,score=3),
                           dict(key,x1=0,y1=0,x2=10,y2=10,score=2),
                           dict(key,x1=20,y1=20,x2=30,y2=30,score=1)])
        score=average_precision(pred,truth,.5)
        self.assertEqual(score['false_positives'],1)
        self.assertAlmostEqual(score['ap'],5/6)

    def test_localization_and_evaluation_cli(self):
        r=self.root
        Image.new('RGB',(100,80),color='gray').save(r/'image.png')
        np.save(r/'features.npy',np.array([[0,1,0],[0,1,0],[0,1,0],[2,1,0]],dtype=np.float32))
        meta=pd.DataFrame(dict(patient_id=['p']*4,exam_id=['e']*4,image_id=['i']*4,
                               image_path=[str(r/'image.png')]*4,view=['L_CC']*4,
                               patch_h=[0,0,1,1],patch_w=[0,1,0,1],grid_h=[2]*4,grid_w=[2]*4))
        meta.to_csv(r/'metadata.csv',index=False)
        (r/'findings.json').write_text(json.dumps(dict(mass=dict(neurons=[0],threshold=1.0))))
        main(['localize','--features',str(r/'features.npy'),'--metadata',str(r/'metadata.csv'),
              '--checkpoint',str(r/'sae.pt'),'--finding-config',str(r/'findings.json'),'--output',str(r/'local')])
        truth=pd.read_csv(r/'local/detections.csv').drop(columns='score')
        truth.to_csv(r/'truth.csv',index=False)
        main(['evaluate-localization','--detections',str(r/'local/detections.csv'),'--annotations',str(r/'truth.csv'),
              '--images',str(r/'local/images.csv'),'--output',str(r/'metrics.json')])
        self.assertEqual(json.loads((r/'metrics.json').read_text())['mean_ap'],1)
        self.assertTrue((r/'local/image_000000_finding_000.png').exists())

    def test_generalizability_and_patient_leakage_rejection(self):
        r=self.root; rows=[]; labels=[]; features=[]
        for i in range(12):
            key=dict(patient_id='p%d'%i,exam_id='e%d'%i,image_id='i%d'%i)
            target=i%2
            labels.append(dict(key,target=target,split=['train','val','test'][i//4]))
            for _ in range(2):
                rows.append(key);features.append([1.+target*5,1.,1.])
        np.save(r/'features.npy',np.array(features,dtype=np.float32))
        pd.DataFrame(rows).to_csv(r/'metadata.csv',index=False)
        pd.DataFrame(labels).to_csv(r/'labels.csv',index=False)
        args=['generalizability','--features',str(r/'features.npy'),'--metadata',str(r/'metadata.csv'),
              '--checkpoint',str(r/'sae.pt'),'--labels',str(r/'labels.csv'),'--label','target']
        main(args+['--output',str(r/'probe')])
        metrics=json.loads((r/'probe/metrics.json').read_text())
        self.assertEqual(metrics['raw']['test_auroc'],1.)
        self.assertEqual(metrics['raw'],metrics['reconstructed'])
        labels[-1]['patient_id']='p0'
        pd.DataFrame(labels).to_csv(r/'labels.csv',index=False)
        with self.assertRaisesRegex(ValueError,'patient-disjoint'):
            main(args+['--output',str(r/'bad')])


if __name__=='__main__':
    unittest.main()
