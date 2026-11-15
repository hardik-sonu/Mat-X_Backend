# MAT-X Data Lake
This directory defines the standardized MAT-X data organization.
## Structure
- raw/: Original source datasets, retained without modification.
- raw/steelbench/: SteelBench source records.
- raw/microsteel/: Reserved for original MicroSteel datasets.
- raw/316l/: Reserved for original 316L datasets.
- raw/inclusions/: Reserved for original inclusion datasets.
- processed/normalized/: Normalized records.
- processed/deduplicated/: Deduplicated records.
- processed/validated/: Records passing documented validation.
- features/tabular/: Derived tabular features.
- features/image/: Derived image and microstructure features.
- models/: Model artifacts and associated metadata.
- metadata/: Dataset manifests, provenance, and processing records.
## Data integrity and provenance
Raw source files must not be silently overwritten by processing.
Record source identity, source location, retrieval or acquisition date,
SHA-256, transformations, and validation status where available.
Derived and predicted values must remain distinguishable from original
source measurements. Missing datasets are not represented as available
data merely because their directories exist.
Dataset-specific validation and scientific applicability remain the
responsibility of their respective ingestion pipelines.
