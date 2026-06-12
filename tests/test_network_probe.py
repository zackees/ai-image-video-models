"""Range-GET the first 4 bytes of every manifest href and assert zstd magic.

Catches the failure mode where the media endpoint serves a Git LFS pointer
text file instead of the real archive bytes. Run with: pytest --run-network
"""

import json
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _all_hrefs():
    hrefs = []
    for manifest_path in sorted(ASSETS.glob("*/*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, entry in manifest.items():
            if key != "latest":
                hrefs.append(entry["href"])
    return hrefs


@pytest.mark.network
@pytest.mark.parametrize("href", _all_hrefs(), ids=lambda h: h.rsplit("/", 1)[-1])
def test_href_serves_zstd_bytes(href):
    req = urllib.request.Request(href, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status in (200, 206), (href, resp.status)
        head = resp.read(4)
    assert head == ZSTD_MAGIC, f"{href} served non-zstd bytes: {head!r} (LFS pointer leak?)"
