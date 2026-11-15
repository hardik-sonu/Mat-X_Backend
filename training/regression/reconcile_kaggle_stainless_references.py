from pathlib import Path
import pandas as pd
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\regression")
df = pd.read_csv(DATA)
df = df[df["source"] == "Kaggle_Stainless_Steel"].copy()
for c in ["C", "Mn", "Cr", "Mo", "Ni", "Si",
          "tensile_strength", "yield_strength", "elongation"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
# Published reference values. These are comparison points,
# not assumed values for every product form or processing condition.
references = [
    {
        "grade_id": "201",
        "reference": "eFunda AISI 201",
        "condition": "Annealed bar",
        "UTS": 515, "YS": 275, "elongation": 40,
        "Si_max": 1.0,
        "reference_url": "https://www.efunda.com/materials/alloys/stainless_steels/show_stainless.cfm?ID=AISI_Type_201&Page_Title=Stainless+Steel+AISI+Type+201&show_prop=uts"
    },
    {
        "grade_id": "201",
        "reference": "eFunda AISI 201",
        "condition": "1/2 hard sheet/strip",
        "UTS": 1030, "YS": 760, "elongation": 10,
        "Si_max": 1.0,
        "reference_url": "https://www.efunda.com/materials/alloys/stainless_steels/show_stainless.cfm?ID=AISI_Type_201&Page_Title=Stainless+Steel+AISI+Type+201&show_prop=uts"
    },
    {
        "grade_id": "201",
        "reference": "eFunda AISI 201",
        "condition": "3/4 hard sheet/strip",
        "UTS": 1210, "YS": 930, "elongation": 7,
        "Si_max": 1.0,
        "reference_url": "https://www.efunda.com/materials/alloys/stainless_steels/show_stainless.cfm?ID=AISI_Type_201&Page_Title=Stainless+Steel+AISI+Type+201&show_prop=uts"
    },
    {
        "grade_id": "201",
        "reference": "eFunda AISI 201",
        "condition": "Full hard sheet/strip",
        "UTS": 1280, "YS": 965, "elongation": 5,
        "Si_max": 1.0,
        "reference_url": "https://www.efunda.com/materials/alloys/stainless_steels/show_stainless.cfm?ID=AISI_Type_201&Page_Title=Stainless+Steel+AISI+Type+201&show_prop=uts"
    },
    {
        "grade_id": "440C",
        "reference": "eFunda AISI 440C",
        "condition": "Tempered at 315 C",
        "UTS": 1970, "YS": 1900, "elongation": 2,
        "Si_max": None,
        "reference_url": "https://www.efunda.com/materials/alloys/stainless_steels/show_stainless.cfm?ID=AISI_Type_440C&Page_Title=Stainless+Steel+AISI+Type+440C&show_prop=uts"
    }
]
rows = []
for _, r in df[df["grade_id"].astype(str).isin(["201", "202", "440C"])].iterrows():
    grade = str(r["grade_id"])
    row = {
        "heat_id": r["heat_id"],
        "grade_id": grade,
        "source": r["source"],
        "C": r["C"], "Mn": r["Mn"], "Cr": r["Cr"],
        "Mo": r["Mo"], "Ni": r["Ni"], "Si": r["Si"],
        "UTS": r["tensile_strength"],
        "YS": r["yield_strength"],
        "elongation": r["elongation"],
    }
    if grade == "201":
        row["standard_Si_max"] = 1.0
        row["chemistry_review"] = (
            "SI_ABOVE_STANDARD_AISI_201_MAX"
            if pd.notna(r["Si"]) and r["Si"] > 1.0
            else "NO_SI_MAXIMUM_EXCEEDANCE"
        )
    elif grade == "202":
        row["standard_Si_max"] = 1.0
        row["chemistry_review"] = (
            "SI_ABOVE_STANDARD_AISI_202_MAX"
            if pd.notna(r["Si"]) and r["Si"] > 1.0
            else "NO_SI_MAXIMUM_EXCEEDANCE"
        )
    else:
        row["standard_Si_max"] = None
        row["chemistry_review"] = "NO_SCREENING_LIMIT_APPLIED"
    matches = []
    for ref in references:
        if ref["grade_id"] != grade:
            continue
        # Compare all three mechanical properties.
        # This is a descriptive nearest-reference calculation only.
        differences = [
            abs(r["tensile_strength"] - ref["UTS"]),
            abs(r["yield_strength"] - ref["YS"]),
            abs(r["elongation"] - ref["elongation"]) * 10
        ]
        score = sum(differences)
        matches.append((score, ref))
    if matches:
        score, best = min(matches, key=lambda x: x[0])
        row["nearest_reference_condition"] = best["condition"]
        row["reference_UTS"] = best["UTS"]
        row["reference_YS"] = best["YS"]
        row["reference_elongation"] = best["elongation"]
        row["reference_comparison_score"] = score
        row["reference_url"] = best["reference_url"]
    else:
        row["nearest_reference_condition"] = "NO_MATCHING_REFERENCE"
        row["reference_UTS"] = None
        row["reference_YS"] = None
        row["reference_elongation"] = None
        row["reference_comparison_score"] = None
        row["reference_url"] = None
    rows.append(row)
result = pd.DataFrame(rows)
path = OUT / "matx_kaggle_stainless_reference_reconciliation.csv"
result.to_csv(path, index=False)
print("=" * 90)
print("REFERENCE RECONCILIATION: GRADES 201, 202, 440C")
print("=" * 90)
for grade in ["201", "202", "440C"]:
    part = result[result["grade_id"] == grade]
    print(f"\nGRADE {grade}: {len(part)} RECORDS")
    print(part[[
        "heat_id", "Si", "UTS", "YS", "elongation",
        "chemistry_review", "nearest_reference_condition",
        "reference_UTS", "reference_YS", "reference_elongation"
    ]].to_string(index=False))
print("\nSAVED:", path)
print("\nNO ORIGINAL DATA OR MODELS MODIFIED.")
