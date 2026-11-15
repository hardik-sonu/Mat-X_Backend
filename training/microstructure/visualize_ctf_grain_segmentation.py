from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
out = Path("training_results/microstructure/abaqus_deep_audit")
thresholds = [2, 5, 10, 15]
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.ravel()
for ax, threshold in zip(axes, thresholds):
    path = out / f"ctf_grain_pixels_{threshold}deg.csv"
    df = pd.read_csv(path)
    xs = np.sort(df["X"].unique())
    ys = np.sort(df["Y"].unique())
    nx, ny = len(xs), len(ys)
    df = df.sort_values(["Y", "X"]).reset_index(drop=True)
    if len(df) != nx * ny:
        raise RuntimeError(f"Incomplete grid at {threshold} degrees.")
    labels = df["grain_id"].to_numpy().reshape(ny, nx)
    if int((labels > 0).sum()) != 73163:
        raise RuntimeError(f"Unexpected valid-pixel count at {threshold} degrees.")
    image = ax.imshow(
        labels,
        origin="lower",
        interpolation="nearest",
        cmap="nipy_spectral",
        aspect="equal"
    )
    count = int(df.loc[df["grain_id"] > 0, "grain_id"].nunique())
    ax.set_title(f"{threshold}° threshold — {count} regions")
    ax.set_xlabel("X grid position")
    ax.set_ylabel("Y grid position")
fig.suptitle(
    "Provisional EBSD Grain Segmentation — EBSD.ctf",
    fontsize=15
)
fig.tight_layout()
image_path = out / "ctf_provisional_grain_segmentation_comparison.png"
fig.savefig(image_path, dpi=220, bbox_inches="tight")
plt.close(fig)
print("SEGMENTATION VISUALIZATION COMPLETE")
print("Saved:", image_path)
print("Four thresholds:", thresholds)
print("Source CTF unchanged.")
