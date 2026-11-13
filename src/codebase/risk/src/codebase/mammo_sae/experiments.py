"""Portable finding analysis and frozen-feature generalizability experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

KEYS = ["patient_id", "exam_id"]


def table(path):
    return pd.read_csv(path, dtype={"patient_id": str, "exam_id": str, "image_id": str})


def findings(frame, columns):
    missing = set(KEYS + columns) - set(frame.columns)
    if missing or frame[KEYS + columns].isna().any().any():
        raise ValueError("Finding table requires complete keys and binary label columns: " + str(missing))
    if not frame[columns].isin([0, 1]).all().all():
        raise ValueError("Finding labels must be 0 or 1; missing labels cannot be treated as negatives")
    return frame.groupby(KEYS, as_index=False)[columns].max()


def select_images(manifest, columns, per_finding, output, seed=42):
    """Balance pure finding examinations, keeping only affected-side images.

    Mixed findings are retained in a cohort summary but excluded from selection.
    Negative examinations retain all views and use half the per-finding count,
    matching the original sampler's bilateral-negative policy.
    """
    frame = table(manifest)
    if per_finding < 2 or not {"image_id", "view", "image_path"}.issubset(frame):
        raise ValueError("Provide per_finding >= 2 and an image manifest")
    exams = findings(frame, columns)
    exams["category"] = exams[columns].apply(
        lambda row: "no_findings" if row.sum() == 0 else (row.idxmax() if row.sum() == 1 else "mixed"), axis=1)
    rng = np.random.default_rng(seed)
    selected = []
    for finding in columns + ["no_findings"]:
        available = exams[exams.category == finding]
        count = min(len(available), per_finding if finding != "no_findings" else per_finding//2)
        if count:
            chosen = available.iloc[rng.choice(len(available), size=count, replace=False)]
            images = frame.merge(chosen[KEYS], on=KEYS, validate="many_to_one")
            if finding != "no_findings":
                sides = images.loc[images[finding] == 1, KEYS + ["view"]].copy()
                sides["side"] = sides.view.str.split("_").str[0]
                images["side"] = images.view.str.split("_").str[0]
                images = images.merge(sides[KEYS + ["side"]].drop_duplicates(), on=KEYS+["side"], validate="many_to_one").drop(columns="side")
            images["selection_finding"] = finding
            selected.append(images)
    if not selected:
        raise ValueError("No pure finding or negative examinations were available")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(selected, ignore_index=True).to_csv(output / "images.csv", index=False)
    exams.groupby("category").size().rename("examinations").to_csv(output / "cohort_counts.csv")


def associations(summary_path, labels_path, columns, output, pure=False):
    """Patient-level presence in the exported top-K summary, with Welch tests.

    Repeated examinations are aggregated per patient. Unlisted neurons are
    absent from top-K, not necessarily zero in the full activation vector.
    """
    summary = table(summary_path)
    if not set(KEYS+["neuron_id", "max_activation"]).issubset(summary):
        raise ValueError("Expected a portable neuron summary")
    if summary.duplicated(KEYS+["neuron_id"]).any():
        raise ValueError("Duplicate examination/neuron rows")
    labels = findings(table(labels_path), columns)
    cohort = summary[KEYS].drop_duplicates().merge(labels, on=KEYS, how="left", validate="one_to_one")
    if cohort[columns].isna().any().any():
        raise ValueError("Some summarized examinations lack finding labels")
    if pure:
        cohort = cohort.loc[cohort[columns].sum(axis=1) <= 1]
    joined = summary.merge(cohort, on=KEYS, validate="many_to_one")
    patient_labels = cohort.groupby("patient_id")[columns].max()
    presence = joined.assign(present=joined.max_activation > 0).pivot_table(
        index="patient_id", columns="neuron_id", values="present", aggfunc="max", fill_value=False)
    presence = presence.reindex(patient_labels.index, fill_value=False)
    rows = []
    for finding in columns:
        positive = patient_labels[finding].to_numpy() == 1
        if positive.sum() < 2 or (~positive).sum() < 2:
            raise ValueError("Each finding requires at least two positive and two negative patients")
        for neuron in presence.columns:
            values = presence[neuron].to_numpy(dtype=float)
            a, b = values[positive], values[~positive]
            # Constant groups have an exact, explicit handling rather than NaNs.
            if a.var() == 0 and b.var() == 0:
                pvalue = 1.0 if a[0] == b[0] else 0.0
            else:
                pvalue = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            rows.append(dict(finding=finding, neuron_id=int(neuron), positive_patients=len(a),
                             negative_patients=len(b), positive_rate=a.mean(), negative_rate=b.mean(),
                             rate_difference=a.mean()-b.mean(), p_value=pvalue))
    if not rows:
        raise ValueError("No neuron associations could be computed")
    result = pd.DataFrame(rows)
    # Benjamini-Hochberg across all reported finding/neuron comparisons.
    order = np.argsort(result.p_value.to_numpy())
    adjusted = result.p_value.to_numpy()[order] * len(result) / np.arange(1, len(result)+1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result["q_value"] = 1.0
    result.loc[order, "q_value"] = adjusted
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "associations.csv", index=False)
    import matplotlib
    matplotlib.use('Agg')
    plotted = result.sort_values("rate_difference", ascending=False).head(30).copy()
    plotted['finding_neuron'] = plotted.finding + ':' + plotted.neuron_id.astype(str)
    plotted.plot.bar(x="finding_neuron", y="rate_difference", figsize=(12, 4), legend=False)
    from matplotlib import pyplot as plt
    plt.ylabel("Positive minus negative patient presence rate")
    plt.tight_layout()
    plt.savefig(output / "associations.png")
    plt.close()


@torch.no_grad()
def generalizability(sae, features, metadata, labels_path, label, output,
                     level=-1, batch_size=256, regularization=(0.01, 0.1, 1.0), seed=42):
    """Compare raw vs reconstructed spatial features using separate linear probes.

    The caller supplies image labels and explicit patient-disjoint train/val/test
    roles. C is selected on validation AUROC independently for each feature type.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keys = KEYS + ["image_id"]
    labels = table(labels_path)
    if not set(keys+[label,"split"]).issubset(labels) or labels[keys+[label,"split"]].isna().any().any():
        raise ValueError("Image labels require patient/exam/image keys, binary label and split")
    if labels.duplicated(keys).any() or not labels[label].isin([0,1]).all():
        raise ValueError("Image labels must be unique and binary")
    if set(labels.split) != {"train", "val", "test"} or (labels.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Use patient-disjoint train, val and test splits")
    observed = metadata[keys].drop_duplicates()
    if len(observed.merge(labels[keys], on=keys)) != len(observed) or len(observed) != len(labels):
        raise ValueError("Features and labels must cover exactly the same image cohort")
    raw, reconstructed, image_rows = [], [], []
    for identity, frame in metadata.groupby(keys, sort=False):
        indices = frame.index.to_numpy()
        raw_sum = np.zeros(features.shape[1], dtype=np.float64)
        recon_sum = np.zeros_like(raw_sum)
        for start in range(0, len(indices), batch_size):
            x = np.asarray(features[indices[start:start+batch_size]])
            raw_sum += x.sum(0, dtype=np.float64)
            recon_sum += sae(x, level).cpu().numpy().sum(0, dtype=np.float64)
        raw.append(raw_sum/len(indices))
        reconstructed.append(recon_sum/len(indices))
        image_rows.append(dict(zip(keys, identity)))
    aligned = pd.DataFrame(image_rows).merge(labels, on=keys, validate="one_to_one")
    masks = {role: aligned.split.to_numpy() == role for role in ["train", "val", "test"]}
    y = aligned[label].to_numpy()
    if any(len(np.unique(y[mask])) != 2 for mask in masks.values()):
        raise ValueError("Both classes must occur in each split for AUROC selection/evaluation")
    if not regularization or any(c <= 0 for c in regularization):
        raise ValueError("Regularization values must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = aligned.loc[masks['test'], keys+[label]].copy()
    results = {}
    for name, vectors in [('raw', raw), ('reconstructed', reconstructed)]:
        x = np.asarray(vectors)
        best, best_score, chosen = None, -float('inf'), None
        for c in regularization:
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
            probe.fit(x[masks['train']], y[masks['train']])
            score = roc_auc_score(y[masks['val']], probe.predict_proba(x[masks['val']])[:,1])
            if score > best_score:
                best, best_score, chosen = probe, score, c
        probability = best.predict_proba(x[masks['test']])[:,1]
        predictions[name+'_probability'] = probability
        results[name] = dict(C=chosen, validation_auroc=best_score,
                             test_auroc=roc_auc_score(y[masks['test']], probability),
                             test_auprc=average_precision_score(y[masks['test']], probability))
        # Portable learned coefficients, without executable pickle serialization.
        scaler, classifier = best.steps[0][1], best.steps[1][1]
        np.savez(output/(name+'_probe.npz'), mean=scaler.mean_, scale=scaler.scale_,
                 coefficient=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
    predictions.to_csv(output/'test_predictions.csv', index=False)
    (output/'metrics.json').write_text(json.dumps(results, indent=2))
