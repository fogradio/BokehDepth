from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DAV2_ROOT = REPO_ROOT / "DepthAnythingV2"
TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "assets",
    "examples",
    "weights",
}
FORBIDDEN_DAV2_TEXT = (
    "AA-DSKFA",
    "AADSKFA",
    "KIFA",
    "ablation",
    "compress",
    "dpt_defocus_stack",
    "DepthAnythingV2DefocusStack",
    "latent",
    "perceiver",
    "prev_latent",
    "rev_latent",
)


def iter_release_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in TEXT_EXTENSIONS:
            yield path


def test_release_repository_text_is_english_only():
    offenders = []
    for path in iter_release_text_files(REPO_ROOT):
        text = path.read_text(encoding="utf-8")
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_depthanythingv2_release_tree_contains_only_dsfa_entrypoints():
    expected_files = {
        "README.md",
        "configs/dsfa_inference.json",
        "configs/dsfa_train.json",
        "datasets/__init__.py",
        "datasets/defocus_stack.py",
        "depth_anything_v2/__init__.py",
        "depth_anything_v2/dinov2.py",
        "depth_anything_v2/dpt.py",
        "depth_anything_v2/dpt_dsfa.py",
        "depth_anything_v2/dinov2_layers/__init__.py",
        "depth_anything_v2/util/transform.py",
        "scripts/infer_dsfa.py",
        "scripts/train_dsfa.py",
        "scripts/dist_train_dsfa.sh",
    }

    missing = sorted(
        rel_path for rel_path in expected_files if not (DAV2_ROOT / rel_path).is_file()
    )
    assert missing == []

    dav2_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in iter_release_text_files(DAV2_ROOT)
    ).lower()
    forbidden_hits = [token for token in FORBIDDEN_DAV2_TEXT if token.lower() in dav2_text]
    assert forbidden_hits == []
