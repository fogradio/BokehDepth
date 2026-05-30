"""CSV -> training JSONL helper for Stage-1 (bokeh-generation) training.

Goal
----
``train_flux_I2I.py`` reads its data from one or more JSONL "manifests".
This script turns a simple per-sample CSV table into such a manifest, so
you can build a training set out of any combination of in-the-wild,
synthetic or paired captures without writing dataset-specific code.

The bundled :func:`build_record` is intentionally dataset-agnostic: it does
not assume a particular folder layout, does not reach out for EXIF on
disk, and only fills the fields documented in ``dataset/dataset.py``.

CSV schema
----------
All columns are optional except ``input_image_path``. Empty cells are
treated as missing.

Required
~~~~~~~~
- ``input_image_path``        Absolute path to the (typically all-in-focus) source image.

I2I (image-to-image) — provide if you have pre-rendered bokeh pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``target_image_path``       Absolute path to the corresponding bokeh image.
- ``foreground_clear``        ``true`` / ``false``; rows marked false are dropped at
                              I2I collate time (use it to filter samples whose subject
                              is not in focus).

Conditioning
~~~~~~~~~~~~
Provide either ``dof_cond`` directly (preferred when you already have a
calibrated K value), OR enough EXIF-style fields for ``calc_dof_cond`` to
compute it (``N``, ``fmm``, ``f35mm``, ``s1``, ``image_width``,
``image_height``). When both are absent the row is skipped with a warning.

- ``dof_cond``                Calibrated K value used as the bokeh conditioning scalar.
- ``N``                       F-number (aperture).
- ``fmm``                     Focal length in mm.
- ``f35mm``                   35mm-equivalent focal length in mm.
- ``s1``                      Focus distance in metres.
- ``image_width``             Source image width in pixels (auto-read from disk if blank).
- ``image_height``            Source image height in pixels (auto-read from disk if blank).

Auxiliary inputs
~~~~~~~~~~~~~~~~
- ``depth_map_path``          Absolute path to ``.npz`` (key ``"depth"``) or ``.npy``.
- ``fg_mask_path``            Absolute path to a grayscale PNG mask of the subject.
- ``captions``                One or more captions separated by ``|``.
- ``task_type``               Free-form string, e.g. ``add_bokeh`` / ``adjust_bokeh``.
- ``disp_focus``              Focal-plane disparity in [0, 1]; leave blank if unknown.
- ``suitable_for_synthetic``  ``true`` / ``false``. Drives the T2I real-vs-synth split.
                              Defaults to ``true`` when ``target_image_path`` is empty,
                              ``false`` otherwise (matches the convention in the
                              original training pipeline).
- ``photo_id``                Optional unique identifier. If absent we hash the input
                              path. The trainer's ``filter_recency`` flag keeps only
                              IDs greater than ``20000000000``; assign IDs in that
                              range if you want a sample to survive the filter.

Usage
-----
::

    python build_manifest.py samples.csv --output dataset.jsonl

Multiple CSV files can be concatenated into one manifest by passing them
in order. Use ``--strict`` to abort on the first invalid row (defaults to
skipping the row and printing a warning).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the script's parent directory to sys.path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.utils import calc_dof_cond  # noqa: E402


def _is_true(value: Optional[str]) -> bool:
    """Parse a CSV cell as a boolean. Empty / unknown strings count as False."""
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    f = _parse_optional_float(value)
    return None if f is None else int(f)


def _split_captions(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [c.strip() for c in str(raw).split("|") if c.strip()]


def _read_image_size(image_path: str) -> Optional[tuple]:
    """Return (width, height) using PIL; None on failure."""
    try:
        from PIL import Image  # local import: avoid hard dependency for callers
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def _hash_photo_id(path: str) -> str:
    """Deterministic fallback ID derived from the absolute input path."""
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return f"sha1-{digest[:16]}"


def build_record(row: Dict[str, str], strict: bool = False) -> Optional[Dict[str, Any]]:
    """Convert one CSV row into a training-ready JSONL record.

    Returns ``None`` when the row cannot be turned into a valid record and
    ``strict`` is False. Raises ``ValueError`` otherwise.
    """
    input_path = (row.get("input_image_path") or "").strip()
    if not input_path:
        msg = "missing required field: input_image_path"
        if strict:
            raise ValueError(msg)
        print(f"[skip] {msg} in row {row}")
        return None
    if not os.path.isabs(input_path):
        input_path = str(Path(input_path).resolve())

    target_path_raw = (row.get("target_image_path") or "").strip()
    target_path = str(Path(target_path_raw).resolve()) if target_path_raw else None

    depth_path_raw = (row.get("depth_map_path") or "").strip()
    depth_path = str(Path(depth_path_raw).resolve()) if depth_path_raw else ""

    fg_mask_path_raw = (row.get("fg_mask_path") or "").strip()
    fg_mask_path = str(Path(fg_mask_path_raw).resolve()) if fg_mask_path_raw else ""

    # ---- Conditioning: dof_cond direct, or computed from EXIF-style fields ----
    dof_cond: Optional[float] = _parse_optional_float(row.get("dof_cond"))
    N = _parse_optional_float(row.get("N"))
    fmm = _parse_optional_float(row.get("fmm"))
    f35mm = _parse_optional_float(row.get("f35mm")) or fmm
    s1 = _parse_optional_float(row.get("s1"))
    img_w = _parse_optional_int(row.get("image_width"))
    img_h = _parse_optional_int(row.get("image_height"))

    if dof_cond is None:
        if (img_w is None or img_h is None):
            size = _read_image_size(input_path)
            if size is not None:
                img_w = img_w or size[0]
                img_h = img_h or size[1]
        if all(v is not None for v in (N, fmm, f35mm, s1, img_w, img_h)):
            try:
                dof_cond = float(calc_dof_cond(N, fmm, f35mm, s1, img_w, img_h))
            except Exception as exc:  # noqa: BLE001
                msg = f"calc_dof_cond failed for {input_path}: {exc}"
                if strict:
                    raise ValueError(msg) from exc
                print(f"[skip] {msg}")
                return None

    if dof_cond is None:
        msg = (
            f"missing dof_cond for {input_path}: provide either a `dof_cond` column "
            "or all of N, fmm, f35mm, s1 (image_width/_height auto-read from disk)."
        )
        if strict:
            raise ValueError(msg)
        print(f"[skip] {msg}")
        return None

    # ---- Other fields ----
    captions = _split_captions(row.get("captions"))
    task_type = (row.get("task_type") or "").strip() or (
        "add_bokeh" if target_path else "adjust_bokeh"
    )
    disp_focus = _parse_optional_float(row.get("disp_focus"))

    if "suitable_for_synthetic" in row and (row.get("suitable_for_synthetic") or "").strip():
        suitable = _is_true(row.get("suitable_for_synthetic"))
    else:
        # Conservative default: I2I samples are paired (False); the rest are eligible
        # for the on-the-fly BokehMe synthesis path used by the T2I synth dataloader.
        suitable = target_path is None

    foreground_clear = True
    if "foreground_clear" in row and (row.get("foreground_clear") or "").strip():
        foreground_clear = _is_true(row.get("foreground_clear"))

    photo_id = (row.get("photo_id") or "").strip() or _hash_photo_id(input_path)

    record: Dict[str, Any] = {
        "photo_id": photo_id,
        "input_image_path": input_path,
        "fg_mask_path": fg_mask_path,
        "depth_map_path": depth_path,
        "captions": captions,
        "camera_anns": {
            "dof-cond": float(dof_cond),
        },
        "suitable_for_synthetic": bool(suitable),
        "task_type": task_type,
        "disp_focus": disp_focus,
        "foreground_clear": bool(foreground_clear),
    }

    # Echo EXIF / camera fields when present so downstream code can introspect them
    for camera_key, value in (("N", N), ("fmm", fmm), ("f35mm", f35mm), ("s1", s1)):
        if value is not None:
            record["camera_anns"][camera_key] = float(value)

    if target_path is not None:
        record["target_image_path"] = target_path

    return record


def iter_csv_rows(csv_path: Path):
    """Yield rows from a CSV. Header normalisation strips whitespace."""
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return
        # Normalise headers: strip whitespace, lower-case keys mapped consistently
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            yield {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more input CSV files.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to the JSONL manifest to write.")
    parser.add_argument("--strict", action="store_true",
                        help="Abort on the first invalid row instead of skipping it.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out_fh:
        for csv_path in args.inputs:
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")
            print(f"[info] reading {csv_path}")
            for row in iter_csv_rows(csv_path):
                record = build_record(row, strict=args.strict)
                if record is None:
                    skipped += 1
                    continue
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"[done] wrote {written} records to {args.output} (skipped {skipped}).")


if __name__ == "__main__":
    main()
