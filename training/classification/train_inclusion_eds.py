from pathlib import Path
import json
import tarfile
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Inclusion detection and classification"
)
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "training_results" / "classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = [
    "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "S",
    "K", "Ca", "Ti", "Cr", "Mn", "Fe", "La", "Ce"
]
ARCHIVES = {
    "train": "features_train_public.tar",
    "validation": "features_val_public.tar",
    "test": "features_test_public.tar"
}
EXPECTED_COUNTS = {
    "train": 2668,
    "validation": 563,
    "test": 618
}
def load_split(split_name):
    archive_path = DATA_DIR / ARCHIVES[split_name]
    samples = {}
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            filename = Path(member.name).name
            if filename.endswith(".eds.json"):
                sample_id = filename[:-len(".eds.json")]
                file_type = "eds"
            elif filename.endswith(".label.cls"):
                sample_id = filename[:-len(".label.cls")]
                file_type = "label"
            elif filename.endswith(".image.png"):
                sample_id = filename[:-len(".image.png")]
                file_type = "image"
            else:
                continue
            samples.setdefault(sample_id, {})
            if file_type in samples[sample_id]:
                raise ValueError(
                    f"Duplicate {file_type} for sample {sample_id}"
                )
            samples[sample_id][file_type] = member
        if len(samples) != EXPECTED_COUNTS[split_name]:
            raise ValueError(
                f"{split_name}: expected {EXPECTED_COUNTS[split_name]} "
                f"samples, found {len(samples)}"
            )
        X = []
        y = []
        manifest = []
        for sample_id in sorted(samples):
            sample = samples[sample_id]
            if set(sample) != {"eds", "label", "image"}:
                raise ValueError(
                    f"Incomplete sample triplet: {sample_id}; "
                    f"found {list(sample.keys())}"
                )
            eds_stream = archive.extractfile(sample["eds"])
            label_stream = archive.extractfile(sample["label"])
            if eds_stream is None or label_stream is None:
                raise ValueError(
                    f"Unable to read sample: {sample_id}"
                )
            eds = json.load(eds_stream)
            if set(eds.keys()) != set(FEATURES):
                raise ValueError(
                    f"Unexpected EDS feature keys in {sample_id}"
                )
            values = []
            for feature in FEATURES:
                value = float(eds[feature])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite {feature} value in {sample_id}"
                    )
                values.append(value)
            label_text = label_stream.read().decode("utf-8").strip()
            if label_text not in {"0", "1"}:
                raise ValueError(
                    f"Invalid label {label_text!r} in {sample_id}"
                )
            X.append(values)
            y.append(int(label_text))
            manifest.append({
                "split": split_name,
                "sample_id": sample_id,
                "eds_file": sample["eds"].name,
                "image_file": sample["image"].name,
                "label_file": sample["label"].name,
                "label": int(label_text)
            })
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.shape != (EXPECTED_COUNTS[split_name], len(FEATURES)):
        raise ValueError(
            f"Unexpected feature matrix shape: {X.shape}"
        )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"Both classes must be present in {split_name}"
        )
    return X, y, manifest
def evaluate(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "records": int(len(y)),
        "class_distribution": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "ROC_AUC": float(
            roc_auc_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )
    }
def main():
    print("Loading official dataset splits...")
    X_train, y_train, train_manifest = load_split("train")
    X_val, y_val, val_manifest = load_split("validation")
    X_test, y_test, test_manifest = load_split("test")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)
    print("\nTraining ExtraTrees classifier...")
    model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Evaluating validation split...")
    validation_results = evaluate(model, X_val, y_val)
    print("Evaluating official test split...")
    test_results = evaluate(model, X_test, y_test)
    artifact = {
        "model": model,
        "features": FEATURES,
        "classes": {
            0: "False inclusion / artifact",
            1: "True non-metallic inclusion"
        }
    }
    candidate_path = (
        MODEL_DIR / "matx_inclusion_eds_extratrees_candidate.joblib"
    )
    joblib.dump(artifact, candidate_path)
    all_manifest = (
        train_manifest + val_manifest + test_manifest
    )
    manifest_path = (
        RESULT_DIR / "inclusion_sample_pairing_manifest.csv"
    )
    pd.DataFrame(all_manifest).to_csv(
        manifest_path,
        index=False
    )
    report = {
        "model": "ExtraTreesClassifier",
        "feature_modality": "EDS chemistry",
        "features": FEATURES,
        "parameters": {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        },
        "split_counts": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test))
        },
        "training": {
            "class_distribution": {
                str(int(label)): int(count)
                for label, count in zip(
                    *np.unique(y_train, return_counts=True)
                )
            }
        },
        "validation": validation_results,
        "test": test_results,
        "candidate_model": str(candidate_path),
        "pairing_manifest": str(manifest_path),
        "limitations": [
            "Specific to the supplied Ti-ULC dataset.",
            "EDS chemistry classifier, not an SEM image detector.",
            "Independent-source validation is still required."
        ]
    }
    report_path = (
        RESULT_DIR / "matx_inclusion_training_candidate_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    print("\nINCLUSION CANDIDATE TRAINING COMPLETED")
    print(json.dumps(report, indent=4))
if __name__ == "__main__":
    main()
