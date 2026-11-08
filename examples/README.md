# Example photographs

`scripts/fetch_examples.py` searches Wikimedia Commons for photographs in six
categories (forest, park/garden, water, urban street, building facade, view
through a window), keeps only CC0 / CC BY / CC BY-SA JPEGs without extra
restrictions, saves 1280 px thumbnails to the git-ignored `data/examples/`,
and records title, author, licence and source URL in `ATTRIBUTION.csv` here.

Full-size images are never committed. To reproduce a run exactly, re-download
the files recorded in `ATTRIBUTION.csv`:

    python scripts/fetch_examples.py --from-attribution examples/ATTRIBUTION.csv

`ATTRIBUTION.csv` lists the 30 photographs (5 per category) used for the
results in the README section "Demo results": 1280 px JPEG thumbnails of
files wider than 1280 px. Narrower files are skipped because Commons would
serve the original file (more strictly rate-limited, and a lower working
resolution than the rest). The selection is the search ranking at fetch
time, not a curated or representative sample.
