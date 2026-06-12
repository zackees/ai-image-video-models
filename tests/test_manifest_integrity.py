"""Validate the manifest tree against the consumer wire contract.

Consumer contract (QualityScaler >= 2026.3.10, FluidFrames.RIFE main):

1. GET https://raw.githubusercontent.com/zackees/ai-image-video-models/main/
   assets/<category>/<kind>/<model_key>/manifest.json
   where model_key = filename.removesuffix("_fp16.onnx" | "_fp32.onnx").
2. entry = manifest[manifest["latest"]]
3. Download entry["href"] (media.githubusercontent.com LFS endpoint).
4. Stream-sha256 against entry["sha256"]; abort on mismatch.
5. Zstd-decompress to <model_path>.tmp, atomic os.replace.

Required leaf fields per version entry: href (https str), sha256 (64-hex),
size (int > 0), compression ("zstd" — the only value consumers handle).

Tests 16/17 work with either smudged LFS files or raw pointer text, so they
pass in CI checkouts with lfs: false as well as local clones.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"
MEDIA_BASE = "https://media.githubusercontent.com/media/zackees/ai-image-video-models/main"

QUALITYSCALER_MODELS = [
    "BSRGANx2",
    "BSRGANx4",
    "IRCNN_Lx1",
    "IRCNN_Mx1",
    "LVAx2",
    "MSharpx4",
    "RealESRGANx4",
    "RealESR_Ax4",
    "RealESR_Gx4",
]
FLUIDFRAMES_MODELS = ["RIFE", "RIFE_Lite", "RIFE_s"]

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_leaf_dirs():
    for model_dir in sorted(ASSETS.glob("*/*/*/")):
        if (model_dir / "manifest.json").exists() and list(model_dir.glob("*.zst")):
            yield model_dir


def _iter_versions(leaf_manifest: dict):
    for key, entry in leaf_manifest.items():
        if key != "latest":
            yield key, entry


def _parse_lfs_pointer_or_file(path: Path) -> dict:
    head = path.open("rb").read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        text = path.read_text(encoding="utf-8")
        oid_m = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
        size_m = re.search(r"^size (\d+)$", text, re.M)
        assert oid_m and size_m, f"malformed LFS pointer: {path}"
        return {"oid": oid_m.group(1), "size": int(size_m.group(1)), "is_pointer": True}
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"oid": h.hexdigest(), "size": path.stat().st_size, "is_pointer": False}


LEAF_DIRS = list(_iter_leaf_dirs())
LEAF_IDS = [d.relative_to(ASSETS).as_posix() for d in LEAF_DIRS]


def _leaf_entries():
    for leaf in LEAF_DIRS:
        manifest = _load_json(leaf / "manifest.json")
        for version, entry in _iter_versions(manifest):
            yield leaf, version, entry


def test_root_manifest_has_categories_array():
    root = _load_json(REPO_ROOT / "manifest.json")
    assert isinstance(root["categories"], list) and root["categories"]


def test_root_category_entries_have_name_and_manifest_path():
    root = _load_json(REPO_ROOT / "manifest.json")
    for cat in root["categories"]:
        assert isinstance(cat["name"], str) and isinstance(cat["manifest_path"], str)


def test_root_category_manifest_paths_exist():
    root = _load_json(REPO_ROOT / "manifest.json")
    for cat in root["categories"]:
        assert (REPO_ROOT / cat["manifest_path"]).is_file(), cat["manifest_path"]


def _iter_category_manifests():
    root = _load_json(REPO_ROOT / "manifest.json")
    for cat in root["categories"]:
        yield REPO_ROOT / cat["manifest_path"], _load_json(REPO_ROOT / cat["manifest_path"])


def test_category_manifest_has_models_array():
    for path, manifest in _iter_category_manifests():
        assert isinstance(manifest["models"], list) and manifest["models"], path


def test_category_model_entries_have_name_and_manifest_path():
    for path, manifest in _iter_category_manifests():
        for model in manifest["models"]:
            assert isinstance(model["name"], str) and isinstance(model["manifest_path"], str)


def test_category_model_manifest_paths_exist():
    for path, manifest in _iter_category_manifests():
        for model in manifest["models"]:
            leaf = path.parent / model["manifest_path"]
            assert leaf.is_file(), leaf


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_manifest_has_latest_string(leaf):
    manifest = _load_json(leaf / "manifest.json")
    assert isinstance(manifest["latest"], str)


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_latest_references_existing_version_key(leaf):
    manifest = _load_json(leaf / "manifest.json")
    assert manifest["latest"] in manifest


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_version_entries_have_href(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert isinstance(entry["href"], str) and entry["href"].startswith("https://")


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_version_entries_have_sha256(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["sha256"]


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_version_entries_have_size_int(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert isinstance(entry["size"], int) and entry["size"] > 0


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_version_entries_have_compression_zstd(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert entry["compression"] == "zstd"


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_href_matches_media_githubusercontent_pattern(leaf):
    on_disk = {f"{MEDIA_BASE}/{p.relative_to(REPO_ROOT).as_posix()}" for p in leaf.glob("*.zst")}
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert entry["href"] in on_disk, entry["href"]


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_href_filename_encodes_version_key(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        filename = entry["href"].rsplit("/", 1)[-1]
        assert f"-v{version}" in filename, (version, filename)


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_consumer_model_key_matches_leaf_dir_name(leaf):
    suffix = "_fp32.onnx.zst" if leaf.relative_to(ASSETS).parts[0] == "fluidframes" else "_fp16.onnx.zst"
    for archive in leaf.glob("*.zst"):
        model_key = re.sub(r"-v[0-9]+(?:\.[0-9]+)*", "", archive.name).removesuffix(suffix)
        assert model_key == leaf.name, (archive.name, leaf.name)


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_sha256_matches_lfs_pointer_or_real_file(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        archive = leaf / entry["href"].rsplit("/", 1)[-1]
        assert _parse_lfs_pointer_or_file(archive)["oid"] == entry["sha256"], archive


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_size_matches_lfs_pointer_or_real_file(leaf):
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        archive = leaf / entry["href"].rsplit("/", 1)[-1]
        assert _parse_lfs_pointer_or_file(archive)["size"] == entry["size"], archive


def test_every_asset_zst_is_referenced_by_exactly_one_manifest_entry():
    on_disk = {p.relative_to(REPO_ROOT).as_posix() for p in ASSETS.rglob("*.zst")}
    referenced = [
        entry["href"].removeprefix(f"{MEDIA_BASE}/") for _, _, entry in _leaf_entries()
    ]
    assert sorted(referenced) == sorted(on_disk)


def test_consumer_category_paths_present():
    for name in QUALITYSCALER_MODELS:
        assert (ASSETS / "qualityscaler" / "onnx" / name / "manifest.json").is_file(), name
    for name in FLUIDFRAMES_MODELS:
        assert (ASSETS / "fluidframes" / "rife" / name / "manifest.json").is_file(), name


def test_required_qualityscaler_models_present():
    manifest = _load_json(ASSETS / "qualityscaler" / "manifest.json")
    names = {m["name"] for m in manifest["models"]}
    assert set(QUALITYSCALER_MODELS) <= names


def test_required_fluidframes_models_present():
    manifest = _load_json(ASSETS / "fluidframes" / "manifest.json")
    names = {m["name"] for m in manifest["models"]}
    assert set(FLUIDFRAMES_MODELS) <= names


@pytest.mark.parametrize("leaf", LEAF_DIRS, ids=LEAF_IDS)
def test_leaf_notes_field_when_notes_json_present(leaf):
    notes_path = leaf / "notes.json"
    if not notes_path.exists():
        pytest.skip("no notes.json for this model")
    for version, entry in _iter_versions(_load_json(leaf / "manifest.json")):
        assert isinstance(entry.get("notes"), str) and entry["notes"], (leaf.name, version)
