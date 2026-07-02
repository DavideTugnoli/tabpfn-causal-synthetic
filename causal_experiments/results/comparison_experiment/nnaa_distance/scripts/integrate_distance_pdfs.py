#!/usr/bin/env python3
"""Copy the |NNAA - 0.5| forest PDFs next to the raw-metric plots.

Distance plots live alongside the other metrics in the official forest
trees, distinguished by the ``_distance0p5`` filename suffix:

- ``main_cleaned``    -> ``forest_plots/paper/comparison_experiment``
- ``noise1e-2_paper`` -> ``forest_plots/paper_noise1e-2/comparison_experiment``
  (single noise dataset, matching the published noise-figure design; the
  multi-dataset ``noise1e-2_cleaned`` bundle exists only for the
  raw-vs-distance audit and must NOT be used as a paper-figure source).

Noise copies also get the ``_noise1e-2`` suffix used by that tree.
Only files whose stem contains ``nnaa`` are copied; raw plots are never
overwritten (the suffix keeps the names disjoint). The ``_distance0p5`` marker
is added only when missing, so source PDFs that already carry it (the current
build output) are copied without doubling the suffix.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
COMPARISON_ROOT = BUNDLE.parent

TARGETS = (
    (
        BUNDLE / "main_cleaned/forest_plots/paper/comparison_experiment",
        COMPARISON_ROOT / "forest_plots/paper/comparison_experiment",
        "",
    ),
    (
        BUNDLE / "noise1e-2_paper/forest_plots/paper/comparison_experiment",
        COMPARISON_ROOT / "forest_plots/paper_noise1e-2/comparison_experiment",
        "_noise1e-2",
    ),
)


def main() -> None:
    copied, skipped = 0, []
    for src_root, dst_root, extra_suffix in TARGETS:
        for pdf in sorted(src_root.glob("*/pdf/*nnaa*.pdf")):
            comparison_dir = pdf.parents[1].name
            dst_dir = dst_root / comparison_dir / "pdf"
            if not dst_dir.exists():
                skipped.append(f"{dst_root.name}/{comparison_dir}")
                continue
            stem = pdf.stem if "_distance0p5" in pdf.stem else f"{pdf.stem}_distance0p5"
            dst = dst_dir / f"{stem}{extra_suffix}.pdf"
            shutil.copy2(pdf, dst)
            copied += 1
    print(f"[OK] {copied} distance PDFs integrated into the official trees")
    if skipped:
        print("[INFO] comparisons without an official folder (skipped):", sorted(set(skipped)))


if __name__ == "__main__":
    main()
