
import os
import json
import math
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import plotly.graph_objects as go

# nebraska_property_taxes.py
# Location-based property tax model for Nebraska
# =========================
# SETTINGS — tweak these before running
# =========================

# Revenue target (dollars)
R_TARGET = 5_308_000_000.0

# Parcels data sources
STATEWIDE_URL = ""  # URL or local path to statewide parcel data (leave empty to use combined file)
COUNTY_URLS = []    # list of county-level parcel URLs

# US state boundaries shapefile (for building NE outline)
STATES_SHP = "cb_2018_us_state_5m.shp"

NE_STATE_GEOJSON = "ne_state.geojson"
SKIP_DOWNLOAD_NE_STATE = True

SKIP_DOWNLOAD_PARCELS  = True
PARCELS_DIR            = "ne_parcels"
PARCELS_COMBINED       = "ne_parcels_combined.geojson"
PARCELS_PARQUET        = "ne_parcels_combined.parquet"

# Model parameters
LAMBDA = 0.05        # decay rate per mile
ALPHA  = 1.05        # city-size scaling exponent
XSTAR_MILES = 7.5    # clamp distance for floor
MAX_RATE_PCT = 10    # cap
MIN_RATE_PCT = 0.0   # floor minimum

GRID_STEP = 0.02     # heatmap resolution (~1-1.5 miles)

# Output locations (using the above variable definitions)

# --------------------------------------------------------------------
import os, io, zipfile, json, tempfile, glob, shutil
import requests, geopandas as gpd, pandas as pd, numpy as np
from shapely.ops import unary_union
from shapely.prepared import prep

os.makedirs(PARCELS_DIR, exist_ok=True)

# ---- helpers --------------------------------------------------------
def _download(url, out_path):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    return out_path

def _extract_if_zip(path):
    if path.lower().endswith(".zip"):
        zdir = path[:-4] + "_unzipped"
        if os.path.exists(zdir):
            shutil.rmtree(zdir)
        os.makedirs(zdir, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(zdir)
        return zdir
    return path

def _detect_vector_source(folder_or_file):
    p = folder_or_file
    if os.path.isdir(p):
        # prefer shapefile; else .gpkg; else first file readable by fiona
        shp = glob.glob(os.path.join(p, "**", "*.shp"), recursive=True)
        if shp: return shp[0]
        gpkg = glob.glob(os.path.join(p, "**", "*.gpkg"), recursive=True)
        if gpkg: return gpkg[0]
        gdb  = glob.glob(os.path.join(p, "**", "*.gdb"), recursive=True)
        if gdb: return gdb[0]  # geopandas can read FileGDB if drivers available
        geojson = glob.glob(os.path.join(p, "**", "*.geojson"), recursive=True)
        if geojson: return geojson[0]
    return p  # could be .geojson directly

VALUE_ALIASES = [
    "land_value", "Land_Value", "LAND_VALUE",
    "Total_Assessed_Value", "TOTAL_ASSESSED_VALUE", "TOTAL_AV",
    "AssessedVal", "ASSESSEDVAL", "Assessed_Value", "AssessedValue",
    "TotalValue", "TOTAL_VALUE", "Total_Parcel_Value"
]

def _normalize_value_column(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print(f"[DEBUG] _normalize_value_column: Input GeoDataFrame has {len(g)} rows and columns: {list(g.columns)}")
    # find first present alias
    found = None
    for c in VALUE_ALIASES:
        if c in g.columns:
            found = c; break
    if not found:
        # try case-insensitive match
        lc = {c.lower(): c for c in g.columns}
        for a in VALUE_ALIASES:
            if a.lower() in lc:
                found = lc[a.lower()]; break
    if not found:
        raise RuntimeError("Could not find an assessed/land value column in: " + ", ".join(g.columns.astype(str)[:25]))
    print(f"[DEBUG] _normalize_value_column: Found value column '{found}', renaming to 'land_value'")
    if found != "land_value":
        g = g.rename(columns={found: "land_value"})
    g["land_value"] = pd.to_numeric(g["land_value"], errors="coerce").fillna(0.0)
    original_count = len(g)
    g = g[g["land_value"] > 0.0].copy()
    print(f"[DEBUG] _normalize_value_column: Filtered from {original_count} to {len(g)} rows with positive land values")
    print(f"[DEBUG] _normalize_value_column: Land value stats - Min: ${g['land_value'].min():,.0f}, Max: ${g['land_value'].max():,.0f}, Mean: ${g['land_value'].mean():,.0f}")
    return g

def _to_wgs84(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        return g.to_crs(epsg=4326)
    except Exception:
        if g.crs is None:
            # last resort: assume data are already lon/lat
            g.set_crs(epsg=4326, inplace=True)
        return g.to_crs(epsg=4326)

# ---- load NE mask ---------------------------------------------------
ne = gpd.read_file(NE_STATE_GEOJSON).to_crs(epsg=4326)
NE_MASK = unary_union(ne.geometry)
NE_PREP = prep(NE_MASK)

def _clip_to_ne(g):
    original_count = len(g)
    g = g[_to_wgs84(g).geometry.within(NE_MASK)].copy()
    print(f"[DEBUG] _clip_to_ne: Filtered from {original_count} to {len(g)} parcels within Nebraska boundaries")
    return g

# ---- main steps -----------------------------------------------------
def read_any_vector(path_or_url) -> gpd.GeoDataFrame:
    print(f"[DEBUG] read_any_vector: Processing source: {path_or_url}")
    local = path_or_url
    if path_or_url.startswith("http"):
        base = os.path.join(PARCELS_DIR, os.path.basename(path_or_url.split("?")[0]) or "download")
        local = base if os.path.splitext(base)[1] else base + ".dat"
        _download(path_or_url, local)
    extracted = _extract_if_zip(local)
    src = _detect_vector_source(extracted)
    print(f"[DEBUG] read_any_vector: Detected source file: {src}")
    g = gpd.read_file(src)
    print(f"[DEBUG] read_any_vector: Loaded {len(g)} features from {src}")
    return g

def combine_and_standardize(urls):
    print(f"[DEBUG] combine_and_standardize: Processing {len(urls)} data sources: {[u.split('/')[-1] for u in urls]}")
    frames = []
    for u in urls:
        try:
            print(f"[DEBUG] combine_and_standardize: Loading data from {u}")
            g = read_any_vector(u)
            print(f"[DEBUG] combine_and_standardize: Loaded {len(g)} raw features with geometry type: {g.geometry.iloc[0].geom_type if len(g) > 0 else 'None'}")

            g = _to_wgs84(g)
            g = _normalize_value_column(g)

            # centroid if not points
            if len(g) and g.geometry.iloc[0].geom_type != "Point":
                print(f"[DEBUG] combine_and_standardize: Converting to centroids for distance calculations")
                g = g.assign(_centroid=g.geometry.centroid).set_geometry("_centroid")
                print(f"[DEBUG] combine_and_standardize: Centroid conversion complete, {len(g)} features")

            g = _clip_to_ne(g)
            if len(g):
                frames.append(g[["land_value", g.geometry.name]])
                print(f"[DEBUG] combine_and_standardize: Added {len(g)} valid parcels to frames")
        except Exception as e:
            print(f"[warn] combine_and_standardize: skipping {u}: {e}")
    if not frames:
        raise RuntimeError("No parcel features were loaded.")
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.set_crs(epsg=4326)  # ensure CRS
    print(f"[DEBUG] combine_and_standardize: Combined all frames into {len(allg)} total parcels")
    print(f"[DEBUG] combine_and_standardize: Total land value sum: ${allg['land_value'].sum():,.0f}")
    allg.to_file(PARCELS_COMBINED, driver="GeoJSON")
    try:
        allg.to_parquet(PARCELS_PARQUET)
    except Exception:
        pass
    print(f"Wrote {PARCELS_COMBINED} with {len(allg):,} points")

# ---- choose statewide or counties ----------------------------------
if not os.path.exists(PARCELS_COMBINED):
    if STATEWIDE_URL:
        combine_and_standardize([STATEWIDE_URL])
    else:
        combine_and_standardize(COUNTY_URLS)

# Model parameters (defined at top of file)

# Nebraska parcels service
PARCELS_SERVICE = "https://giscat.ne.gov/Enterprise/rest/services/StatewideParcelsExternal/MapServer/0/query"

# Metro seats (NE side only) and in-state populations
CBSA = {
    "Omaha":        {"lat": 41.257160, "lon": -95.995102, "pop": 861487},
    "Lincoln":      {"lat": 40.806862, "lon": -96.681679, "pop": 344387},
    "Grand Island": {"lat": 40.929077, "lon": -98.368149, "pop": 76479},
    "Kearney":      {"lat": 40.701000, "lon": -99.081000, "pop": 57467},
    "Norfolk":      {"lat": 42.034000, "lon": -97.426094, "pop": 48782},
    "Columbus":     {"lat": 41.429700, "lon": -97.368400, "pop": 45175},
    "Hastings":     {"lat": 40.586300, "lon": -98.389900, "pop": 40366},
    "Fremont":      {"lat": 41.433900, "lon": -96.498100, "pop": 37187},
    "Scottsbluff":  {"lat": 41.866600, "lon": -103.667200, "pop": 36373},
    "North Platte": {"lat": 41.138935, "lon": -100.765305, "pop": 34020},
    "Lexington":    {"lat": 40.780600, "lon": -99.741800, "pop": 25932},
    "Beatrice":     {"lat": 40.268600, "lon": -96.746700, "pop": 21634},
    "South Sioux City": {"lat": 42.473600, "lon": -96.413100, "pop": 21268},
}
POP_REF = CBSA["Omaha"]["pop"]

# =========================
# UTILITIES
# =========================
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# =========================
# 1) BUILD NEBRASKA POLYGON FROM LOCAL SHAPEFILE
# =========================
def build_ne_from_shp():
    print(f"[DEBUG] build_ne_from_shp: Building Nebraska state boundary from {STATES_SHP}")
    if not os.path.exists(STATES_SHP):
        raise FileNotFoundError("Missing cb_2018_us_state_5m.shp at path: {}".format(STATES_SHP))
    gdf = gpd.read_file(STATES_SHP)
    print(f"[DEBUG] build_ne_from_shp: Loaded {len(gdf)} state features, columns: {list(gdf.columns)}")
    # Robust filtering: prefer STATEFP, fallback to NAME
    if "STATEFP" in gdf.columns:
        ne = gdf[gdf["STATEFP"] == "31"]
        print("[DEBUG] build_ne_from_shp: Filtered by STATEFP='31'")
    else:
        ne = gdf[gdf["NAME"].astype(str).str.upper() == "NEBRASKA"]
        print("[DEBUG] build_ne_from_shp: Filtered by NAME='NEBRASKA'")
    if ne.empty:
        raise RuntimeError("Could not find Nebraska in the states layer.")
    print(f"[DEBUG] build_ne_from_shp: Found {len(ne)} Nebraska features")
    ne = ne.to_crs(epsg=4326)
    ne_union = unary_union(ne.geometry)
    ne_gdf = gpd.GeoDataFrame({"state": ["Nebraska"]}, geometry=[ne_union], crs="EPSG:4326")
    ne_gdf.to_file(NE_STATE_GEOJSON, driver="GeoJSON")
    print("Wrote ne_state.geojson")
    print(f"[DEBUG] build_ne_from_shp: Nebraska boundary area: {ne_union.area:.2f} square degrees")

if (not SKIP_DOWNLOAD_NE_STATE) or (not os.path.exists(NE_STATE_GEOJSON)):
    build_ne_from_shp()

# =========================
# 2) PARCELS: DOWNLOAD AND COMBINE
# =========================
NE_COUNTIES = [
    "Adams","Antelope","Arthur","Banner","Blaine","Boone","Box Butte","Boyd","Brown","Buffalo",
    "Burt","Butler","Cass","Cedar","Chase","Cherry","Cheyenne","Clay","Colfax","Cuming","Custer",
    "Dakota","Dawes","Dawson","Deuel","Dixon","Dodge","Douglas","Dundy","Fillmore","Franklin",
    "Frontier","Furnas","Gage","Garden","Garfield","Gosper","Grant","Greeley","Hall","Hamilton",
    "Harlan","Hayes","Hitchcock","Holt","Hooker","Howard","Jefferson","Johnson","Kearney","Keith",
    "Keya Paha","Kimball","Knox","Lancaster","Lincoln","Logan","Loup","Madison","McPherson","Merrick",
    "Morrill","Nance","Nemaha","Nuckolls","Otoe","Pawnee","Perkins","Phelps","Pierce","Platte",
    "Polk","Red Willow","Richardson","Rock","Saline","Sarpy","Saunders","Scotts Bluff","Seward",
    "Sheridan","Sherman","Sioux","Stanton","Thayer","Thomas","Thurston","Valley","Washington","Wayne",
    "Webster","Wheeler","York"
]

def distinct_counties():
    # Return as strings to match the existing query: County_ID='...'
    return NE_COUNTIES.copy()

def download_county(county_id, out_path):
    offset = 0
    batch = 2000
    collected = []
    while True:
        params = {
            "where": "County_ID='{}'".format(county_id),
            "outFields": "State_PID,County_ID,Land_Value,Total_Assessed_Value",
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": batch,
        }
        r = requests.get(PARCELS_SERVICE, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        if not feats:
            break
        collected.extend(feats)
        if len(feats) < batch:
            break
        offset += batch
    if not collected:
        with open(out_path, "w") as f:
            f.write('{"type":"FeatureCollection","features":[]}')
        return
    out = {"type": "FeatureCollection", "features": collected}
    with open(out_path, "w") as f:
        json.dump(out, f)

def build_parcels():
    ensure_dir(PARCELS_DIR)
    print("Fetching distinct County_ID values...")
    counties = distinct_counties()
    if not counties:
        raise RuntimeError("No County_ID values returned from the parcels service.")
    print("Found {} counties".format(len(counties)))
    for cid in tqdm(counties, desc="Downloading parcels by county"):
        out = os.path.join(PARCELS_DIR, "parcels_{}.geojson".format(cid))
        if not os.path.exists(out):
            download_county(cid, out)
    # Combine
    frames = []
    for fn in os.listdir(PARCELS_DIR):
        if fn.endswith(".geojson"):
            path = os.path.join(PARCELS_DIR, fn)
            try:
                g = gpd.read_file(path)
                if len(g) > 0:
                    frames.append(g)
            except Exception:
                pass
    if not frames:
        raise RuntimeError("No parcel features to combine.")
    allg = pd.concat(frames, ignore_index=True)
    # normalize land_value column
    if "land_value" in allg.columns:
        pass
    elif "Land_Value" in allg.columns:
        allg = allg.rename(columns={"Land_Value": "land_value"})
    elif "Total_Assessed_Value" in allg.columns:
        allg = allg.rename(columns={"Total_Assessed_Value": "land_value"})
        print("Using Total_Assessed_Value as proxy for land_value.")
    else:
        raise RuntimeError("Need Land_Value or Total_Assessed_Value in parcels.")
    allg.to_file(PARCELS_COMBINED, driver="GeoJSON")
    try:
        allg.to_parquet(PARCELS_PARQUET)
    except Exception:
        pass
    print("Wrote {}".format(PARCELS_COMBINED))

if (not SKIP_DOWNLOAD_PARCELS) and (not os.path.exists(PARCELS_COMBINED) and not os.path.exists(PARCELS_PARQUET)):
    build_parcels()

# =========================
# 3) LOAD + CLIP PARCELS TO NE; USE CENTROIDS FOR DISTANCE
# =========================
ne = gpd.read_file(NE_STATE_GEOJSON).to_crs(epsg=4326)
NE_MASK = unary_union(ne.geometry)
prep_mask = prep(NE_MASK)

def load_parcels():
    print("[DEBUG] load_parcels: Loading parcel data...")
    if os.path.exists(PARCELS_PARQUET):
        print(f"[DEBUG] load_parcels: Loading from Parquet file: {PARCELS_PARQUET}")
        g = gpd.read_parquet(PARCELS_PARQUET)
    else:
        print(f"[DEBUG] load_parcels: Loading from GeoJSON file: {PARCELS_COMBINED}")
        g = gpd.read_file(PARCELS_COMBINED)
    print(f"[DEBUG] load_parcels: Loaded {len(g)} raw parcels")

    g = g.to_crs(epsg=4326)
    print(f"[DEBUG] load_parcels: Converted to EPSG:4326, geometry type: {g.geometry.iloc[0].geom_type if len(g) > 0 else 'None'}")

    # centroid if not points
    if g.geometry.iloc[0].geom_type != "Point":
        print("[DEBUG] load_parcels: Converting polygons to centroids for distance calculations")
        g = g.assign(_centroid=g.geometry.centroid).set_geometry("_centroid")

    # clip to NE polygon
    original_count = len(g)
    g = g[g.geometry.within(NE_MASK)].copy()
    print(f"[DEBUG] load_parcels: Clipped to Nebraska: {original_count} -> {len(g)} parcels")

    # normalize value column again if needed
    if "land_value" not in g.columns:
        for c in ["Land_Value", "LAND_VALUE", "Total_Assessed_Value", "TOTAL_ASSESSED_VALUE"]:
            if c in g.columns:
                g = g.rename(columns={c: "land_value"})
                print(f"[DEBUG] load_parcels: Renamed column '{c}' to 'land_value'")
                break
    if "land_value" not in g.columns:
        raise RuntimeError("No 'land_value' column present.")
    g["land_value"] = pd.to_numeric(g["land_value"], errors="coerce").fillna(0.0)
    original_count = len(g)
    g = g[g["land_value"] > 0.0].copy()
    print(f"[DEBUG] load_parcels: Filtered positive values: {original_count} -> {len(g)} parcels")
    print(f"[DEBUG] load_parcels: Final parcel value stats - Total: ${g['land_value'].sum():,.0f}, Avg: ${g['land_value'].mean():,.0f}")
    return g

gparcels = load_parcels()

# =========================
# 4) INFLUENCE (VECTORIZED)
# =========================
cbsa_names = list(CBSA.keys())
cbsa_lat = np.array([CBSA[n]["lat"] for n in cbsa_names])
cbsa_lon = np.array([CBSA[n]["lon"] for n in cbsa_names])
cbsa_pop = np.array([CBSA[n]["pop"] for n in cbsa_names])
scale_vec = (cbsa_pop / float(POP_REF)) ** ALPHA
print(f"[DEBUG] Influence setup: {len(cbsa_names)} metro areas with populations ranging from {cbsa_pop.min():,} to {cbsa_pop.max():,}")
print(f"[DEBUG] Influence setup: Scale factors range from {scale_vec.min():.3f} to {scale_vec.max():.3f}")
print(f"[DEBUG] Influence setup: Using lambda={LAMBDA}/mi, alpha={ALPHA}, reference pop={POP_REF:,}")

def max_influence(lat_vec, lon_vec):
    # lat_vec, lon_vec: 1D arrays
    lat_m = np.repeat(lat_vec[:, None], len(cbsa_names), axis=1)
    lon_m = np.repeat(lon_vec[:, None], len(cbsa_names), axis=1)
    d = haversine_miles(lat_m, lon_m, cbsa_lat[None, :], cbsa_lon[None, :])
    infl = scale_vec[None, :] * np.exp(-LAMBDA * d)
    max_infl = infl.max(axis=1)
    print(f"[DEBUG] max_influence: Processing {len(lat_vec)} points, distances range: {d.min():.1f}-{d.max():.1f} miles")
    print(f"[DEBUG] max_influence: Influence values range: {max_infl.min():.6f}-{max_infl.max():.6f}")
    return max_infl

# clamp rule: floor = THETA * K
min_scale = float(scale_vec.min())
THETA = min_scale * math.exp(-LAMBDA * XSTAR_MILES)

# =========================
# 5) SOLVE FOR K (VALUE-WEIGHTED REVENUE NEUTRAL)
# =========================
p_lat = gparcels.geometry.y.values
p_lon = gparcels.geometry.x.values
p_val = gparcels["land_value"].values
print(f"[DEBUG] K calibration: Loaded {len(p_val)} parcels with total value: ${p_val.sum():,.0f}")
print(f"[DEBUG] K calibration: Parcel coordinates range: lat {p_lat.min():.4f}-{p_lat.max():.4f}, lon {p_lon.min():.4f}-{p_lon.max():.4f}")

p_infl = max_influence(p_lat, p_lon)
print(f"[DEBUG] K calibration: Target revenue: ${R_TARGET:,.0f}")
print(f"[DEBUG] K calibration: THETA (floor scaling): {THETA:.6f}")

def total_revenue_for_K(K):
    floor_rate = max(MIN_RATE_PCT / 100.0, THETA * K)  # Ensure minimum 1% rate
    raw = K * p_infl
    uncapped_rate = np.maximum(floor_rate, raw)    # fraction (0.05 => 5%)
    rate = np.minimum(uncapped_rate, MAX_RATE_PCT / 100.0)  # Cap at MAX_RATE_PCT%
    revenue = float(np.dot(rate, p_val))
    max_rate_achieved = rate.max() * 100.0
    print(f"[DEBUG] total_revenue_for_K: K={K:.6f}, max_rate={max_rate_achieved:.2f}%, floor_rate={floor_rate*100:.2f}%, revenue=${revenue:,.0f}")
    return revenue

print("[DEBUG] K calibration: Starting binary search for K...")
lo = 0.0
hi = 1.0  # 25% is an absurd upper bound, good for bracketing
for i in range(64):
    mid = 0.5 * (lo + hi)
    rev = total_revenue_for_K(mid)
    if rev < R_TARGET:
        lo = mid
    else:
        hi = mid
    if i % 10 == 0 or i == 63:  # Log every 10 iterations and final
        print(f"[DEBUG] K calibration: Iteration {i+1}, K range: [{lo:.6f}, {hi:.6f}], revenue: ${rev:,.0f}")

K_star = 0.5 * (lo + hi)
FLOOR_star = max(MIN_RATE_PCT / 100.0, THETA * K_star)  # Ensure minimum 1% rate
print(f"[DEBUG] K calibration: Final K={K_star:.6f}, floor_rate={FLOOR_star:.6f}")
print("Calibrated: lambda={:.3f}/mi, K={:.3f}%, floor={:.3f}% (min: {:.1f}%)".format(LAMBDA, K_star * 100.0, FLOOR_star * 100.0, MIN_RATE_PCT))
print("Revenue check: ${:,.0f}".format(total_revenue_for_K(K_star)))

# =========================
# 5.5) CITY TAX RATES
# =========================
print("\n" + "="*60)
print("CITY TAX RATES UNDER NEW LOCATION-BASED SYSTEM")
print("="*60)

# Calculate tax rates for each city
city_rates = {}
for city_name, city_info in CBSA.items():
    # Calculate influence for this city
    city_lat = city_info["lat"]
    city_lon = city_info["lon"]

    # Calculate distance to all cities including itself
    distances = np.array([haversine_miles(city_lat, city_lon, other_info["lat"], other_info["lon"])
                         for other_info in CBSA.values()])
    influences = scale_vec * np.exp(-LAMBDA * distances)
    max_infl = influences.max()

    # Calculate effective tax rate
    raw_rate = K_star * max_infl
    effective_rate = np.maximum(FLOOR_star, raw_rate)
    # Apply the maximum rate cap
    effective_rate = np.minimum(effective_rate, MAX_RATE_PCT / 100.0)
    tax_rate_pct = effective_rate * 100.0

    city_rates[city_name] = {
        'rate_pct': tax_rate_pct,
        'influence': max_infl,
        'population': city_info['pop']
    }

# Print results sorted by population (largest first)
print(f"{'City':<20} {'Population':<12} {'Tax Rate':<10} {'Influence'}")
print("-" * 60)

for city_name in sorted(city_rates.keys(), key=lambda x: city_rates[x]['population'], reverse=True):
    city_data = city_rates[city_name]
    print(f"{city_name:<20} {city_data['population']:<12,} {city_data['rate_pct']:<10.2f}% {city_data['influence']:.4f}")

print(f"\nFloor rate (minimum for remote areas): {FLOOR_star*100:.2f}%")

# =========================
# 6) HEATMAP GRID (CLIPPED TO NE)
# =========================
minx, miny, maxx, maxy = NE_MASK.bounds
print(f"[DEBUG] Heatmap grid: Nebraska bounds: ({minx:.4f}, {miny:.4f}) to ({maxx:.4f}, {maxy:.4f})")

xs = np.arange(minx, maxx + GRID_STEP, GRID_STEP)
ys = np.arange(miny, maxy + GRID_STEP, GRID_STEP)
print(f"[DEBUG] Heatmap grid: Grid dimensions: {len(xs)} x {len(ys)} = {len(xs) * len(ys)} total points")

pts_lon = []
pts_lat = []
total_points = 0
for lat in ys:
    strip = [Point(lon, lat) for lon in xs]
    inside = np.array([prep_mask.contains(pt) for pt in strip], dtype=bool)
    if inside.any():
        kept_lons = np.array(xs)[inside]
        kept_lats = np.full(kept_lons.shape, lat, dtype=float)
        pts_lon.extend(kept_lons.tolist())
        pts_lat.extend(kept_lats.tolist())
        total_points += len(kept_lons)

print(f"[DEBUG] Heatmap grid: {total_points} points inside Nebraska boundaries")

pts_lon = np.array(pts_lon)
pts_lat = np.array(pts_lat)
if pts_lon.size == 0:
    raise RuntimeError("No grid points inside Nebraska mask. Check ne_state.geojson.")

print("[DEBUG] Heatmap grid: Calculating influence for grid points...")
infl_grid = max_influence(pts_lat, pts_lon)
uncapped_rate_grid = np.maximum(FLOOR_star, K_star * infl_grid)
rate_grid = np.minimum(uncapped_rate_grid, MAX_RATE_PCT / 100.0) * 100.0  # percent, capped at MAX_RATE_PCT
print(f"[DEBUG] Heatmap grid: Rate grid stats - Min: {rate_grid.min():.2f}%, Max: {rate_grid.max():.2f}%, Mean: {rate_grid.mean():.2f}%")

df_grid = pd.DataFrame({"lon": pts_lon, "lat": pts_lat, "rate_pct": rate_grid})
pivot = df_grid.pivot_table(index="lat", columns="lon", values="rate_pct")
lat_vals = pivot.index.values
lon_vals = pivot.columns.values
z_vals = pivot.values

# =========================
# 7) PLOTLY HEATMAP
# =========================
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=z_vals, x=lon_vals, y=lat_vals,
    colorscale="Reds", zsmooth="best",
    colorbar=dict(title="Rate (%)")
))

def add_border(geom):
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(width=1, color="black"), name="Nebraska"))
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            x, y = part.exterior.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(width=1, color="black"), name="Nebraska"))

add_border(NE_MASK)

fig.add_trace(go.Scatter(
    x=[CBSA[n]["lon"] for n in CBSA],
    y=[CBSA[n]["lat"] for n in CBSA],
    mode="markers+text",
    text=list(CBSA.keys()),
    textposition="top center",
    marker=dict(size=6, color="black"),
    name="Metro seats"
))

fig.update_layout(
    title="Nebraska Location-Based Property Tax (value-weighted; lambda={:.2f}/mi, K={:.2f}%, floor={:.3f}%)".format(
        LAMBDA, K_star * 100.0, FLOOR_star * 100.0
    ),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    yaxis_scaleanchor="x",
    template="plotly_white",
    margin=dict(l=10, r=10, t=40, b=10),
)

print("[DEBUG] Plotting: Generating heatmap visualization...")
print(f"[DEBUG] Plotting: Grid dimensions for plotting: {len(lat_vals)} x {len(lon_vals)}")
print(f"[DEBUG] Plotting: Rate range for visualization: {z_vals.min():.2f}% to {z_vals.max():.2f}%")

fig.write_html("ne_property_tax_heatmap.html", include_plotlyjs="cdn")
print("Saved ne_property_tax_heatmap.html")
print("[DEBUG] Plotting: Heatmap generation complete!")

