from pathlib import Path
import re
import json
import hashlib
from collections import Counter
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\microstructure\abaqus_deep_audit")
OUT.mkdir(parents=True, exist_ok=True)
path = ROOT / "Grain orientation and cyclic behavior" / "Dataset" / "Cyclic_data.dat"
print("\n" + "=" * 85)
print("CYCLIC DATA FILE INSPECTION")
print("=" * 85)
if not path.exists():
    raise FileNotFoundError(path)
raw = path.read_bytes()
print("File:", path)
print("Size:", len(raw), "bytes")
print("SHA256:", hashlib.sha256(raw).hexdigest())
print("First 64 bytes:", raw[:64].hex(" "))
decoded = None
encoding_used = None
for enc in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        decoded = raw.decode(enc)
        encoding_used = enc
        break
    except UnicodeDecodeError:
        continue
report = {
    "file": str(path.relative_to(ROOT)),
    "size_bytes": len(raw),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "encoding": encoding_used,
    "text_decoded": decoded is not None
}
if decoded is None:
    print("\nCLASSIFICATION: Binary or undecodable file")
    report["classification"] = "BINARY_OR_UNDECODABLE"
else:
    lines = decoded.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    print("Encoding:", encoding_used)
    print("Total lines:", len(lines))
    print("Non-empty lines:", len(nonempty))
    print("\nFIRST 100 NON-EMPTY LINES:")
    for i, line in enumerate(nonempty[:100], 1):
        print(f"{i:04}: {line[:300]}")
    numeric_rows = 0
    token_counts = Counter()
    delimiters = Counter()
    numeric_values = []
    for line in nonempty:
        tokens = re.split(r"[,;\t ]+", line.strip())
        tokens = [t for t in tokens if t]
        token_counts[len(tokens)] += 1
        for delimiter in (",", ";", "\t"):
            if delimiter in line:
                delimiters[delimiter] += 1
        values = []
        for token in tokens:
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                pass
        if len(values) >= 2:
            numeric_rows += 1
            numeric_values.extend(values)
    print("\nSTRUCTURE:")
    print("Numeric-like rows:", numeric_rows)
    print("Column/token-count distribution:", dict(token_counts.most_common(15)))
    print("Delimiters:", dict(delimiters))
    print("Parsed numeric values:", len(numeric_values))
    if numeric_values:
        print("Numeric range:", min(numeric_values), "to", max(numeric_values))
    report.update({
        "classification": "TEXT",
        "line_count": len(lines),
        "nonempty_lines": len(nonempty),
        "numeric_like_rows": numeric_rows,
        "token_count_distribution": dict(token_counts),
        "delimiter_counts": dict(delimiters),
        "parsed_numeric_values": len(numeric_values),
        "numeric_min": min(numeric_values) if numeric_values else None,
        "numeric_max": max(numeric_values) if numeric_values else None,
        "preview": nonempty[:100]
    })
(OUT / "cyclic_data_inspection.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("\nREPORT:")
print(OUT / "cyclic_data_inspection.json")
print("\nINSPECTION COMPLETE")
