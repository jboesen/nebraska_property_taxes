# Nebraska Location-Based Property Tax Model

A model for what Nebraska property taxes *should* look like — rates that scale with city size and decay with distance from the city center. Bigger cities pay more (that's where speculation hurts most), rural areas pay less (farmers aren't the problem), and the revenue stays the same.

## The Idea

The formula is simple:

```
tax_rate(lat, lon) = K * max_over_cities[ (P_c / P_ref)^α * exp(-λ * distance_c) ]
```

- **K** — calibrated automatically to hit your revenue target
- **α** — how much faster big-city rates grow (superlinear, ~1.05)
- **λ** — how fast rates drop as you leave town (per mile)
- **P_c** — each city's population; bigger city, bigger influence
- **Floors and caps** — because 42% tax rates don't pass the sniff test

The model finds a single K such that the value-weighted sum of all parcel tax rates equals your target revenue. Everything else — the floor, the cap, λ, α — you tweak.

## What You Get

- **City tax rates** — printed to stdout for every metro in the state
- **Statewide heatmap** — saved as `ne_property_tax_heatmap.html` (Plotly, interactive)
- **Revenue check** — confirms the total matches your target

## Setup

You need Python 3.11+ and the usual suspects:

```
pip install requests numpy pandas geopandas shapely plotly tqdm
```

The repo includes `ne_state.geojson` (Nebraska's outline, built from Census shapefiles). You'll also need parcel data — the combined file is ~160MB, which is too big for GitHub, so you'll need to grab it or generate it.

### Getting Parcel Data

**Option A: Pre-combined file** — If you have `ne_parcels_combined.geojson`, drop it in the project root. The script picks it up automatically.

**Option B: Download from ArcGIS** — Set `SKIP_DOWNLOAD_PARCELS = False` and the script will pull each county from Nebraska's statewide parcels service. This takes a while but works.

## Running It

```
python nebraska_property_taxes.py
```

That's it. The script loads parcels, calibrates K, prints the city rate table, and spits out a heatmap.

### Tweaking the Knobs

Everything's at the top of the file:

| Variable | Default | What it does |
|---|---|---|
| `R_TARGET` | $5.308B | Revenue to raise (total property tax haul) |
| `LAMBDA` | 0.05 | Exponential decay per mile (higher = faster drop-off) |
| `ALPHA` | 1.05 | City-size exponent (1.0 = linear, >1 = superlinear) |
| `MAX_RATE_PCT` | 10 | Hard cap on any parcel's rate |
| `MIN_RATE_PCT` | 0.0 | Hard floor (overridden by the model's natural floor if higher) |

Play around with λ and the caps. Low λ means Omaha's influence reaches further into the countryside. High caps mean cities carry more of the burden. The sweet spot is somewhere in the middle — you want to punish speculation, not ranchers.

## Results Worth Stealing

The default config (λ=0.05, 1–10% bounds) gives:

| City | Rate | Why |
|---|---|---|
| Omaha | 10.00% | Biggest city, biggest problem with speculation |
| Lincoln | 10.00% | Same logic, hits the cap |
| Grand Island | 7.87% | Medium city, medium rate |
| Kearney | 5.83% | |
| Fremont | 10.00% | Close to Omaha, so Omaha's influence pushes it up |
| Cherry County | ~1.41% | Floor rate — ranchers pay almost nothing |

The floor settles at about 1.41%. That's what the most remote parts of the state pay. Compare that to the current system, where farmers in those same areas are getting crushed.

## Why Not Consumption Taxes?

Because they're regressive, they discourage spending, and they hurt the same rural communities they're supposed to help. Property taxes, properly designed, don't have to be the problem. This is how you fix them.
