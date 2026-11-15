from pathlib import Path
import pandas as pd
import numpy as np
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\regression")
df = pd.read_csv(DATA)
df = df[df["source"] == "Kaggle_Stainless_Steel"].copy()
for col in ["C", "Mn", "Cr", "Mo", "Ni", "Si",
            "tensile_strength", "yield_strength", "elongation"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Reference conditions are comparison data, not specimen provenance.
references = [
    {"grade":"201","condition":"Annealed reference",
     "UTS":515,"YS":275,"elongation":40,
     "source":"Published AISI 201 reference"},
    {"grade":"201","condition":"1/2 hard reference",
     "UTS":1030,"YS":760,"elongation":10,
     "source":"Published AISI 201 reference"},
    {"grade":"201","condition":"3/4 hard reference",
     "UTS":1210,"YS":930,"elongation":7,
     "source":"Published AISI 201 reference"},
    {"grade":"201","condition":"Full hard reference",
     "UTS":1280,"YS":965,"elongation":5,
     "source":"Published AISI 201 reference"},
    {"grade":"440C","condition":"Annealed",
     "UTS":758,"YS":448,"elongation":14,
     "source":"MDPI Metals 2025 / eFunda"},
    {"grade":"440C","condition":"Annealed and cold drawn",
     "UTS":862,"YS":690,"elongation":7,
     "source":"MDPI Metals 2025"},
    {"grade":"440C","condition":"Hardened and tempered at 315 C",
     "UTS":1965,"YS":1896,"elongation":2,
     "source":"MDPI Metals 2025 / Carpenter Technology"},
]
rows = []
for _, r in df.iterrows():
    grade = str(r["grade_id"])
    applicable = [x for x in references if x["grade"] == grade]
    result = {
        "heat_id": r["heat_id"],
        "grade_id": grade,
        "C": r["C"], "Mn": r["Mn"], "Cr": r["Cr"],
        "Ni": r["Ni"], "Si": r["Si"],
        "observed_UTS": r["tensile_strength"],
        "observed_YS": r["yield_strength"],
        "observed_elongation": r["elongation"],
    }
    # Provisional chemistry screening assumes composition in wt%.
    if grade in ["201", "202"]:
        result["Si_screening_limit_wt_percent"] = 1.0
        result["Si_screening"] = (
            "REVIEW_ABOVE_1_WT_PERCENT"
            if pd.notna(r["Si"]) and r["Si"] > 1.0
            else "NOT_ABOVE_SCREENING_LIMIT"
        )
    else:
        result["Si_screening_limit_wt_percent"] = np.nan
        result["Si_screening"] = "NOT_SCREENED"
    candidates = []
    for ref in applicable:
        observed = [
            r["tensile_strength"],
            r["yield_strength"],
            r["elongation"]
        ]
        expected = [ref["UTS"], ref["YS"], ref["elongation"]]
        if any(pd.isna(v) for v in observed):
            continue
        pct = [
            abs(o - e) / abs(e) * 100
            for o, e in zip(observed, expected)
        ]
        # All three properties must be reasonably close.
        # These are screening tolerances, not standard acceptance limits.
        if grade == "440C":
            limits = [5, 5, 20]
        else:
            limits = [10, 15, 25]
        within = all(d <= lim for d, lim in zip(pct, limits))
        candidates.append({
            "condition": ref["condition"],
            "reference_source": ref["source"],
            "UTS_difference_percent": pct[0],
            "YS_difference_percent": pct[1],
            "elongation_difference_percent": pct[2],
            "screening_match": within,
            "score": sum(pct)
        })
    if candidates:
        candidates.sort(key=lambda x: x["score"])
        best = candidates[0]
        result.update({
            "closest_reference_condition": best["condition"],
            "reference_source": best["reference_source"],
            "UTS_difference_percent": best["UTS_difference_percent"],
            "YS_difference_percent": best["YS_difference_percent"],
            "elongation_difference_percent": best["elongation_difference_percent"],
            "all_three_within_screening_tolerance": best["screening_match"],
            "condition_interpretation": (
                "REFERENCE_CONDITION_PATTERN_MATCH_NOT_PROVEN"
                if best["screening_match"]
                else "NO_CLOSE_REFERENCE_CONDITION_MATCH"
            )
        })
    else:
        result.update({
            "closest_reference_condition": "NO_REFERENCE_AVAILABLE",
            "reference_source": None,
            "UTS_difference_percent": np.nan,
            "YS_difference_percent": np.nan,
            "elongation_difference_percent": np.nan,
            "all_three_within_screening_tolerance": False,
            "condition_interpretation": "NO_REFERENCE_AVAILABLE"
        })
    rows.append(result)
report = pd.DataFrame(rows)
path = OUT / "matx_kaggle_stainless_corrected_condition_matching.csv"
report.to_csv(path, index=False)
print("=" * 100)
print("MAT-X CORRECTED CONDITION REFERENCE MATCHING")
print("=" * 100)
for grade in ["201", "202", "440C"]:
    part = report[report["grade_id"] == grade]
    print(f"\nGRADE {grade}")
    cols = [
        "heat_id", "Si", "observed_UTS", "observed_YS",
        "observed_elongation", "closest_reference_condition",
        "UTS_difference_percent", "YS_difference_percent",
        "elongation_difference_percent",
        "condition_interpretation", "Si_screening"
    ]
    print(part[cols].to_string(index=False))
print("\nMATCH COUNTS")
print(report.groupby(
    ["grade_id", "condition_interpretation"]
).size().to_string())
print("\nSAVED:", path)
print("NO ORIGINAL DATA OR MODEL MODIFIED.")
