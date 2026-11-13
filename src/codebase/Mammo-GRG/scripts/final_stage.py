#!/usr/bin/env python3
import argparse
import ast
import json
import os
from typing import List, Dict, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm


# ---------------------- Model utils ----------------------

def load_model_and_tokenizer(model_id: str, quant4bit: bool = True, dtype: str = "bfloat16"):
    torch_dtype = getattr(torch, dtype)
    bnb_config = None
    if quant4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=bnb_config,  # remove if running full precision
    )
    return model, tokenizer


def chat(model, tokenizer, messages: List[Dict], max_new_tokens=256, temperature=0.7, top_p=0.9) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------- ZS parsing & prompts ----------------------

FOUR_VIEWS = {("R", "CC"): "RCC", ("R", "MLO"): "RMLO", ("L", "CC"): "LCC", ("L", "MLO"): "LMLO"}

# Only *_pred columns are used
ZS_KEYS = {
    "mass": "mass_pred",
    "suspicious_calcification": "suspicious calcification_pred",
    "asymmetry": "asymmetry_pred",
}

SYSTEM_PROMPT = (
    "You are a radiology report assistant for screening mammography.\n"
    "Produce a FINAL REPORT with exactly two sections:\n"
    "1) Findings\n"
    "2) Impression\n\n"
    "FOUNDATION:\n"
    "- Treat the PRELIMINARY REPORT as the base text. Preserve its clinically relevant content, density description, laterality, locations (e.g., upper outer quadrant), and recommendations.\n"
    "- You MUST include (carry forward) all non-contradicted statements from the preliminary report.\n"
    "- Reconcile contradictions using the rules below.\n\n"
    "RECONCILIATION RULES (VERY IMPORTANT):\n"
    "1) If structured findings are POSITIVE for a category (mass, suspicious calcification, asymmetry), you MUST reflect that positivity in the final report:\n"
    "   - Remove or revise any preliminary 'no X' statements for that category.\n"
    "   - State laterality accurately: Left, Right, or 'bilaterally' ONLY if both sides are positive.\n"
    "2) If structured findings are NEGATIVE and the preliminary report already says 'no X', keep that negative statement.\n"
    "3) Do NOT add 'bilaterally' unless BOTH sides are positive.\n"
    "4) BI-RADS POLICY (SCREENING): You must output BI-RADS in {0, 1, 2} ONLY—never 3–6 for screening exams.\n"
    "   - If any structured finding is positive/indeterminate (i.e., additional diagnostic work-up is needed), assign BI-RADS 0.\n"
    "   - If there are no findings, assign BI-RADS 1.\n"
    "   - If only benign/non-suspicious findings are present, assign BI-RADS 2.\n"
    "   - If the preliminary BI-RADS is provided, keep it if consistent with these rules; otherwise correct it.\n"
    "   - IMPORTANT: If  asymmetry is POSITIVE in the structured findings, override any preliminary BI-RADS and assign BI-RADS 0.\n"
    "5) Do NOT mention external models or the word 'classifier'.\n"
    "6) Use standard terminology (CC, MLO; laterality) and be concise, clinically appropriate.\n"
    "7) Preserve the overall template and sentence style of the preliminary report while outputting ONLY the two sections."
)


def _safe_parse_zero_shot(zs):
    if isinstance(zs, list):
        return zs
    if pd.isna(zs):
        return []
    if isinstance(zs, str):
        try:
            return ast.literal_eval(zs)
        except Exception:
            try:
                return json.loads(zs)
            except Exception:
                return []
    return []


def build_view_and_side_summaries(zs_list) -> Tuple[Dict, Dict, Dict]:
    """
    Returns:
      view_dict: { "RCC": {"mass": 0/1, ...}, "RMLO": {...}, "LCC": {...}, "LMLO": {...} }
      max_summary: { "mass": 0/1, "suspicious_calcification": 0/1, "asymmetry": 0/1 }
      side_presence (ZS): { finding: {"L": 0/1, "R": 0/1} }
    """
    view_dict: Dict[str, Dict[str, int]] = {}
    side_presence: Dict[str, Dict[str, int]] = {k: {"L": 0, "R": 0} for k in ZS_KEYS.keys()}

    for d in zs_list:
        lat = d.get("image_laterality")
        view = d.get("image_view_type")
        key = FOUR_VIEWS.get((lat, view))
        if not key:
            continue
        view_dict.setdefault(key, {})
        for finding, pred_key in ZS_KEYS.items():
            pred_val = int(d.get(pred_key, 0))
            view_dict[key][finding] = pred_val
            if lat in ("L", "R") and pred_val == 1:
                side_presence[finding][lat] = 1

    max_summary: Dict[str, int] = {}
    for finding in ZS_KEYS.keys():
        preds = [view_dict[v].get(finding, 0) for v in view_dict]
        max_summary[finding] = 1 if any(preds) else 0

    return view_dict, max_summary, side_presence


# ---------------------- DF-based presence + reasoning prints ----------------------

def _norm_lat(val):
    if not isinstance(val, str):
        return None
    v = val.strip().upper()
    if v in ("L", "LEFT", "LT"):
        return "L"
    if v in ("R", "RIGHT", "RT"):
        return "R"
    return None


def df_side_presence_from_row(row) -> Dict[str, Dict[str, int]]:
    """
    Build left/right presence from *df* columns (not zero-shot):
      Uses: mass, mass_laterality, calc, calc_laterality,
            asymmetry, asymmetry_laterality
    Returns: {finding: {"L":0/1, "R":0/1}}
    """
    side_presence = {k: {"L": 0, "R": 0} for k in ["mass", "suspicious_calcification", "asymmetry"]}
    mapping = {
        "mass": ("mass", "mass_laterality"),
        "suspicious_calcification": ("calc", "calc_laterality"),
        "asymmetry": ("asymmetry", "asymmetry_laterality"),
    }

    for finding, (flag_key, lat_key) in mapping.items():
        flag = row.get(flag_key, 0)
        try:
            flag = int(flag)
        except Exception:
            flag = 0
        lat = _norm_lat(row.get(lat_key)) if lat_key in row else None
        if flag == 1:
            if lat == "L":
                side_presence[finding]["L"] = 1
            elif lat == "R":
                side_presence[finding]["R"] = 1
            else:
                # laterality unspecified in df; leave both 0
                pass
    return side_presence


def print_reasoning_block(prelim_report: str,
                          df_side_presence: Dict[str, Dict[str, int]],
                          zs_side_presence: Dict[str, Dict[str, int]]):
    """
    Prints numbered, natural-language reasoning lines.
    Uses ZERO-SHOT (ZS) presence for the Yes/No values,
    so the reasoning matches what the model will reconcile against.
    """
    label_map = {
        "suspicious_calcification": "calcification",
        "mass": "mass",
        "asymmetry": "asymmetry",
    }

    def yn(v: int) -> str:
        return "Yes" if v else "No"

    print("=== Reasoning ===")
    print(f"1) Preliminary report:\n{prelim_report.strip()}\n")

    n = 2
    for key in [
        "suspicious_calcification",
        "mass",
        "asymmetry"
    ]:
        # ZS signals drive the Yes/No here
        left = yn(zs_side_presence[key]["L"])
        right = yn(zs_side_presence[key]["R"])
        lbl = label_map[key]
        print(f"{n}) Checking for {lbl} on the left breast: {left}");
        n += 1
        print(f"{n}) Checking for {lbl} on the right breast: {right}");
        n += 1


# ---------------------- Prompt builders ----------------------

def build_classifier_note(max_summary: Dict[str, int], side_presence: Dict[str, Dict[str, int]]) -> str:
    """
    Human-readable bullets:
      - If both sides positive -> "...: bilaterally."
      - If only left -> "...: Left breast."
      - If only right -> "...: Right breast."
      - If none positive -> 'No positive ML findings across views.'
    """
    label_map = {
        "mass": "Suspicious Mass",
        "suspicious_calcification": "Suspicious Calcification",
        "asymmetry": "Asymmetry",
    }

    lines: List[str] = []
    for finding, label in label_map.items():
        if max_summary.get(finding, 0) != 1:
            continue
        left_pos = side_presence[finding]["L"] == 1
        right_pos = side_presence[finding]["R"] == 1
        if left_pos and right_pos:
            where = "bilaterally."
        elif left_pos:
            where = "Left breast."
        elif right_pos:
            where = "Right breast."
        else:
            where = "side unspecified."
        lines.append(f"- {label}: {where}")

    return "\n".join(lines) if lines else "- No positive ML findings across views."


def build_user_prompt(prelim_report: str, view_dict: Dict, max_summary: Dict, side_presence: Dict,
                      extra_context: str = None) -> str:
    classifier_bullets = build_classifier_note(max_summary, side_presence)
    compact_json = json.dumps(max_summary, ensure_ascii=False)
    return (
            "Preliminary report (from report generator):\n"
            f"{prelim_report.strip()}\n\n"
            "Structured ML findings (max across RCC/RMLO/LCC/LMLO):\n"
            f"{classifier_bullets}\n\n"
            "Max-summary JSON (do not quote or mention 'classifier' in the final text):\n"
            f"{compact_json}\n\n"
            "Rewrite the report into the required two sections ONLY (Findings, Impression). "
            "Follow the template and structure of the preliminary report."
            + (f"\n\nAdditional context: {extra_context}" if extra_context else "")
    )


def row_to_messages(row: pd.Series) -> List[Dict]:
    # Zero-shot presence
    zs = _safe_parse_zero_shot(row.get("zero_shot_per_image"))
    _view_dict, max_summary, side_presence_zs = build_view_and_side_summaries(zs)

    # DF-based presence
    side_presence_df = df_side_presence_from_row(row)

    prelim = (row.get("generated_report") or "").strip()

    # Print the improved reasoning BEFORE generation
    print_reasoning_block(prelim, side_presence_df, side_presence_zs)

    # Build LLaMA prompts (use ZS signals to inform integration)
    user_prompt = build_user_prompt(prelim, _view_dict, max_summary, side_presence_zs)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------- Generation loop ----------------------

def generate_final_reports(
        df: pd.DataFrame,
        model,
        tokenizer,
        max_new_tokens=320,
        temperature=0.3,
        top_p=0.9
) -> List[str]:
    outputs = []
    for idx in tqdm(range(len(df)), desc="Generating reports", unit="row"):
        row = df.iloc[idx]
        messages = row_to_messages(row)
        print(f"patient_id: {row['patient_id']}, exam_id: {row['exam_id']}")
        try:
            out = chat(model, tokenizer, messages,
                       max_new_tokens=max_new_tokens,
                       temperature=temperature,
                       top_p=top_p)
            print("=== Generated Final Report ===")
            print(out, "\n")
        except Exception as e:
            out = f"[GENERATION_ERROR] {e}"
        outputs.append(out)
    return outputs


# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate final mammography reports with LLaMA 3.1 using zero-shot per-image findings and printed reasoning.")
    p.add_argument("--input-csv", required=True,
                   help="Path to input CSV (must include columns: generated_report, zero_shot_per_image).")
    p.add_argument("--output-csv", required=True, help="Path to write augmented CSV with final_generated_report_zs.")
    p.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model id.")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (requires more VRAM).")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="Computation dtype.")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    # Optional convenience filters:
    p.add_argument("--patient-id", default=None, help="If set, process only this patient_id.")
    p.add_argument("--exam-id", default=None, help="If set, process only this exam_id.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    print(f"[+] Loading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(df.columns)
    # Optional filters
    if args.patient_id is not None:
        df = df[df["patient_id"] == args.patient_id]
    if args.exam_id is not None:
        df = df[df["exam_id"] == args.exam_id]

    # Basic sanity checks
    needed = {"generated_report", "zero_shot_per_image"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    print(f"[+] Loading model: {args.model_id} (4-bit={'OFF' if args.no_4bit else 'ON'})")
    model, tokenizer = load_model_and_tokenizer(args.model_id, quant4bit=not args.no_4bit, dtype=args.dtype)

    print("[+] Generating final reports...")
    df["final_generated_report_zs"] = generate_final_reports(
        df,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"[+] Saving to: {args.output_csv}")
    df.to_csv(args.output_csv, index=False)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
