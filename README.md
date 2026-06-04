# Nebraska Location-Based Property Tax Model

## Setup

```
pip install requests numpy pandas geopandas shapely plotly tqdm
```

You'll need parcel data. The combined GeoJSON is ~160MB, too big for GitHub, so either grab it from wherever you keep yours and drop it in the project root as `ne_parcels_combined.geojson`, or set `SKIP_DOWNLOAD_PARCELS = False` and let it pull each county from Nebraska's ArcGIS service (slow but it works).

## Run

```
python nebraska_property_taxes.py
```

Prints city tax rates to stdout and generates `ne_property_tax_heatmap.html`.

## Config

All at the top of `nebraska_property_taxes.py`:

| Variable | Default | What it does |
|---|---|---|
| `R_TARGET` | 5,308,000,000 | Total revenue to raise ($) |
| `LAMBDA` | 0.05 | Rate decay per mile from city center |
| `ALPHA` | 1.05 | How much faster big-city rates grow |
| `MAX_RATE_PCT` | 10 | Maximum tax rate (%) |
| `MIN_RATE_PCT` | 0.0 | Minimum tax rate (%) |

Tweak those, re-run. The script calibrates K automatically to hit the revenue target.
