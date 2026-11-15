# Data Sources, Adapters, and Provenance
## 1. Data-source strategy
MAT-X Version 1.0 identifies public materials data as its primary data strategy. The
backend includes adapter endpoints for the following source categories:
| Source | Intended role | Data interpretation |
|---|---|---|
| SteelBench | Experimental steel-property and heat-level data | Experimental records, subject to source metadata |
| NIST JARVIS | Computational materials knowledge | Computational values must remain distinguishable |
| AFLOW | Computational materials information | Computational values must remain distinguishable |
| Zenodo | Public research datasets | Dataset-specific metadata and licensing apply |
| Local datasets | User-provided CSV/XLSX data | Requires validation and source attribution |
The existence of an adapter does not mean every record from the source has been
downloaded, ingested, validated, or persisted locally.
## 2. Provenance fields
The Version 1.0 specification calls for preservation of:
- Original source.
- Dataset name.
- DOI or source URL.
- Original record identifier.
- Material identifier.
- Measurement method.
- Publication or source reference.
- Units.
- Transformation history.
- Experimental or computational classification.
- Original or imputed classification.
Where a field is not available from a source, it should be reported as unavailable
rather than fabricated.
## 3. Scientific value categories
MAT-X must keep these categories distinct:
| Category | Meaning |
|---|---|
| Experimental | Measured experimentally |
| Computational | Obtained through DFT, simulation, or another computational method |
| Derived | Calculated from other values |
| Imputed | Estimated to replace a missing value |
| Predicted | Produced by a MAT-X model |
A prediction must never be presented as an experimental measurement.
## 4. Local data lake
The standardized project data layout includes:
```text
data/
├── raw/
│   ├── steelbench/
│   ├── microsteel/
│   ├── 316l/
│   └── inclusions/
├── processed/
│   ├── normalized/
│   ├── deduplicated/
│   └── validated/
├── features/
│   ├── tabular/
│   └── image/
├── models/
└── metadata/
```
The raw layer is intended to preserve original downloaded files. Processed and feature
layers should retain traceability to their source records and transformation steps.
The presence of a directory does not imply that the corresponding dataset has been
fully downloaded or processed.
## 5. External source reliability
External API responses may be unavailable, incomplete, rate-limited, or changed by
their providers. Clients should handle source errors and incomplete results explicitly.
Do not interpret an empty response as proof that a material or property does not exist.
## 6. Dataset licensing
Before redistributing a dataset or its derived files, verify its license and citation
requirements at the original provider. This documentation does not grant redistribution
rights.
