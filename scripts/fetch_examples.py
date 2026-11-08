"""Download example images from Wikimedia Commons with licence metadata.

Only JPEG photographs licensed CC0, CC BY or CC BY-SA (any version) are
kept (PNG results are mostly maps and diagrams); files with additional
restrictions (e.g. personality rights) are skipped. Images
are saved as thumbnails (default 1280 px wide) under data/examples/<category>/
(git-ignored), and each search response is cached in data/examples/.search/
so that a rerun after rate-limit failures resumes with the same files.
Title, author, licence and source URL are recorded in examples/ATTRIBUTION.csv,
which is committed.

Two modes:

  python scripts/fetch_examples.py                  # search Commons, write ATTRIBUTION.csv
  python scripts/fetch_examples.py --from-attribution examples/ATTRIBUTION.csv
                                                    # re-download exactly the recorded files

Search results on Commons change over time, so the attribution file (not the
search) is the reproducible record of which images were used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = ("nature-visual-features/0.1 (research demo; "
              "https://github.com/lizxu-byte/nature-visual-features)")
ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = {
    "forest": "forest trees",
    "park_garden": "park garden lawn",
    "water": "lake shore",
    "urban_street": "street scene shops pedestrians",
    "building_facade": "building facade",
    "window_view": "view through window",
}

FIELDS = ["category", "file", "title", "author", "licence", "licence_url", "source_url",
          "thumb_url", "width", "height"]


def is_allowed_licence(short_name: str) -> bool:
    """True for CC0, CC BY x.y and CC BY-SA x.y (optionally with a jurisdiction port).

    Accepts the spaced form shown on Commons ("CC BY-SA 4.0") and the dashed
    form ("cc-by-sa-4.0"); anything with NC or ND, and non-CC licences, fail.
    """
    tokens = [t for t in re.split(r"[ -]+", (short_name or "").strip().lower()) if t]
    if tokens[:1] == ["cc0"]:
        rest = tokens[1:]
    elif tokens[:2] == ["cc", "by"]:
        rest = tokens[2:]
        if rest[:1] == ["sa"]:
            rest = rest[1:]
    else:
        return False
    if rest and re.fullmatch(r"\d\.\d", rest[0]):
        rest = rest[1:]
    return len(rest) == 0 or (len(rest) == 1 and rest[0].isalpha() and rest[0] not in {"nc", "nd"})


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def parse_page(page: dict, category: str, min_width: int = 0) -> dict | None:
    """Turn one API page (with imageinfo + extmetadata) into an attribution row, or None.

    Files not wider than min_width are skipped: for them the API returns the
    original file rather than a standard-size thumbnail, which Wikimedia
    rate-limits more strictly, and they would be analysed at a lower
    resolution than the rest.
    """
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})

    def value(key):
        return meta.get(key, {}).get("value", "")

    licence = strip_html(value("LicenseShortName"))
    if not is_allowed_licence(licence):
        return None
    if strip_html(value("Restrictions")):
        return None
    if info.get("mime") != "image/jpeg":
        return None
    if min_width and (info.get("width") or 0) <= min_width:
        return None
    thumb = info.get("thumburl") or info.get("url")
    if not thumb:
        return None
    title = page["title"]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.removeprefix("File:"))[:80]
    return {
        "category": category,
        "file": f"{category}/{Path(stem).stem}.jpg",
        "title": title,
        "author": strip_html(value("Artist")) or "unknown",
        "licence": licence,
        "licence_url": strip_html(value("LicenseUrl")),
        "source_url": info.get("descriptionurl", ""),
        "thumb_url": thumb,
        "width": info.get("thumbwidth") or info.get("width"),
        "height": info.get("thumbheight") or info.get("height"),
    }


def retry_delay(err: urllib.error.HTTPError, attempt: int) -> float | None:
    """Seconds to wait before retrying a 429/503 response, or None to give up.

    Wikimedia rate-limits shared IPs and sends Retry-After; honour it (at
    least 5 s), otherwise back off exponentially.
    """
    if err.code not in {429, 503}:
        return None
    try:
        after = float(err.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        after = 0.0
    return max(after, 5.0 * 2 ** attempt)


def fetch_bytes(url: str, timeout: float, retries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            delay = retry_delay(err, attempt)
            if delay is None or attempt == retries:
                raise
            print(f"HTTP {err.code}; retrying in {delay:.0f} s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def api_get(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    return json.loads(fetch_bytes(f"{API}?{query}", timeout=60))


def search(query: str, width: int, limit: int = 50) -> list[dict]:
    data = api_get({
        "action": "query", "generator": "search", "gsrnamespace": 6, "gsrlimit": limit,
        "gsrsearch": f"{query} filetype:bitmap",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": width,
    })
    return data.get("query", {}).get("pages", [])


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(fetch_bytes(url, timeout=120))


def cached_search(query: str, width: int, cache: Path) -> list[dict]:
    """search(), with the response kept in cache so that reruns see the same results."""
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    pages = search(query, width)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages), "utf-8")
    return pages


def from_search(per_category: int, width: int, out_dir: Path) -> list[dict]:
    rows = []
    for category, query in CATEGORIES.items():
        kept = 0
        key = hashlib.sha1(f"{query}|{width}".encode()).hexdigest()[:8]
        cache = out_dir / ".search" / f"{category}_{key}.json"
        for page in cached_search(query, width, cache):
            row = parse_page(page, category, min_width=width)
            if row is None:
                continue
            if not (out_dir / row["file"]).exists():
                try:
                    download(row["thumb_url"], out_dir / row["file"])
                except urllib.error.HTTPError as err:
                    print(f"skipped {row['title']}: HTTP {err.code}", file=sys.stderr)
                    continue
            rows.append(row)
            kept += 1
            print(f"{category}: {row['title']} ({row['licence']})", file=sys.stderr)
            time.sleep(0.5)
            if kept >= per_category:
                break
        if kept < per_category:
            print(f"warning: only {kept} usable images for {category}", file=sys.stderr)
    return rows


def from_attribution(path: Path, out_dir: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        dest = out_dir / row["file"]
        if not dest.exists():
            download(row["thumb_url"], dest)
            time.sleep(0.5)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280, help="thumbnail width in pixels")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "examples")
    parser.add_argument("--attribution", type=Path, default=ROOT / "examples" / "ATTRIBUTION.csv")
    parser.add_argument("--from-attribution", type=Path,
                        help="re-download the files listed in this attribution CSV")
    args = parser.parse_args(argv)

    if args.from_attribution:
        rows = from_attribution(args.from_attribution, args.out_dir)
        print(f"{len(rows)} images in {args.out_dir}", file=sys.stderr)
        return 0
    rows = from_search(args.per_category, args.width, args.out_dir)
    args.attribution.parent.mkdir(parents=True, exist_ok=True)
    with open(args.attribution, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.attribution}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
