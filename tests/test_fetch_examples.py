"""Offline tests for the licence filter and metadata parsing of scripts/fetch_examples.py."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_examples", Path(__file__).resolve().parents[1] / "scripts" / "fetch_examples.py")
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


@pytest.mark.parametrize("name", ["CC0", "CC BY 4.0", "CC BY-SA 4.0", "CC BY-SA 3.0 de",
                                  "CC BY 2.0", "cc-by-sa-4.0"])
def test_allowed_licences(name):
    assert fetch.is_allowed_licence(name)


@pytest.mark.parametrize("name", ["Public domain", "GFDL", "CC BY-NC 2.0", "CC BY-NC-SA 4.0",
                                  "CC BY-ND 4.0", "", "Copyrighted free use", "FAL"])
def test_rejected_licences(name):
    assert not fetch.is_allowed_licence(name)


def page(licence="CC BY-SA 4.0", restrictions="", mime="image/jpeg"):
    return {
        "title": "File:Beech forest (Example) 2021.jpg",
        "imageinfo": [{
            "mime": mime,
            "url": "https://upload.wikimedia.org/full.jpg",
            "thumburl": "https://upload.wikimedia.org/thumb.jpg",
            "width": 4000, "height": 2666, "thumbwidth": 1280, "thumbheight": 853,
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Beech_forest.jpg",
            "extmetadata": {
                "LicenseShortName": {"value": licence},
                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                "Artist": {"value": '<a href="//commons.wikimedia.org/wiki/User:X">A. N. '
                                    "Example</a> &amp; co"},
                "Restrictions": {"value": restrictions},
            },
        }],
    }


def test_parse_page_builds_attribution_row():
    row = fetch.parse_page(page(), "forest")
    assert row["author"] == "A. N. Example & co"
    assert row["licence"] == "CC BY-SA 4.0"
    assert row["file"] == "forest/Beech_forest_Example_2021.jpg"
    assert row["thumb_url"].endswith("thumb.jpg")
    assert row["width"] == 1280
    assert set(row) == set(fetch.FIELDS)


def test_parse_page_skips_disallowed():
    assert fetch.parse_page(page(licence="CC BY-NC 4.0"), "forest") is None
    assert fetch.parse_page(page(restrictions="personality"), "forest") is None
    assert fetch.parse_page(page(mime="image/tiff"), "forest") is None
    assert fetch.parse_page(page(mime="image/png"), "forest") is None


def test_parse_page_min_width():
    assert fetch.parse_page(page(), "forest", min_width=1280) is not None
    narrow = page()
    narrow["imageinfo"][0]["width"] = 640
    assert fetch.parse_page(narrow, "forest", min_width=1280) is None
    assert fetch.parse_page(narrow, "forest") is not None


def _http_error(code, retry_after=None):
    import email.message
    import urllib.error

    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "msg", headers, None)


def test_retry_delay_honours_retry_after():
    assert fetch.retry_delay(_http_error(429, "13"), 0) == 13
    assert fetch.retry_delay(_http_error(429, "1"), 0) == 5  # floor
    assert fetch.retry_delay(_http_error(503), 2) == 20  # exponential without header
    assert fetch.retry_delay(_http_error(404), 0) is None
