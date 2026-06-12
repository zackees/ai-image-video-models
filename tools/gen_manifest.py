"""Regenerate manifest.json files for the assets/ tree.

Scans assets/<category>/<kind>/<model>/<model>-v<version>.<ext>.zst files and
writes a leaf manifest.json per model, a manifest.json per category, and a
root manifest.json indexing all categories.

If a model dir contains notes.json ({"default": str, "<version>": str}), each
emitted version entry gets a "notes" field (version-specific text wins over
"default"). Consumers ignore the field; it is provenance for humans.

Usage: python tools/gen_manifest.py
"""

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"
MEDIA_BASE = "https://media.githubusercontent.com/media/zackees/ai-image-video-models/main"

VERSION_RE = re.compile(r"-v(?P<version>[0-9]+(?:\.[0-9]+)*)")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def natural_sort_key(version: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"[.]", version)]


def main() -> None:
    categories = {}
    # sort by posix string, not Path: Windows Path ordering is case-insensitive
    # and would emit a different model order than Linux CI
    for model_dir in sorted(ASSETS.glob("*/*/*/"), key=lambda p: p.as_posix()):
        archives = sorted(model_dir.glob("*.zst"), key=lambda p: p.name)
        if not archives:
            continue
        versions = {}
        for archive in archives:
            m = VERSION_RE.search(archive.name)
            if not m:
                raise ValueError(f"no -v<version> in {archive.name}")
            rel = archive.relative_to(REPO_ROOT).as_posix()
            versions[m.group("version")] = {
                "href": f"{MEDIA_BASE}/{rel}",
                "sha256": sha256_of(archive),
                "size": archive.stat().st_size,
                "compression": "zstd",
            }
        notes_path = model_dir / "notes.json"
        if notes_path.exists():
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
            for version, entry in versions.items():
                note = notes.get(version, notes.get("default"))
                if note is not None:
                    entry["notes"] = note

        latest = sorted(versions, key=natural_sort_key)[-1]
        write_json(model_dir / "manifest.json", {"latest": latest, **versions})

        category = model_dir.relative_to(ASSETS).parts[0]
        rel_manifest = model_dir.relative_to(ASSETS / category).as_posix() + "/manifest.json"
        categories.setdefault(category, []).append({"name": model_dir.name, "manifest_path": rel_manifest})

    for category, models in sorted(categories.items()):
        write_json(ASSETS / category / "manifest.json", {"models": models})

    write_json(
        REPO_ROOT / "manifest.json",
        {"categories": [{"name": c, "manifest_path": f"assets/{c}/manifest.json"} for c in sorted(categories)]},
    )


if __name__ == "__main__":
    main()
