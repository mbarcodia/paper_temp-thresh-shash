"""
Trend bias comparison using the MAPP-proj1 larger CMIP6 ensemble.

SSPs included : SSP1-2.6, SSP2-4.5, SSP3-7.0  (SSP1-1.9 excluded)
Variables     : global annual mean tas, CNA annual mean tas
Trend period  : 1970-2023
Baseline      : 1850-1899 (subtracted per grid cell, per member)
Obs           : _Land_and_Ocean_LatLong1_185001_202312_ann_mean_2pt5degree.nc

Deduplication rule
------------------
When a GCM+SSP combination has both a multi-member ncecat file and
single-member files, only the ncecat file is used (it already contains
those single members).  If multiple ncecat files exist for the same
GCM+SSP, the file with the most members (checked by opening briefly)
is retained.

Output
------
  trend_bias_mapp_ssp126_245_370_1970_2023.png
"""

import sys, os, re, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
import regionmask
from scipy import stats

# ============================================================================
# SETUP
# ============================================================================

# Point this at the directory containing the CMIP6 tas .nc files (ncecat
# multi-member and/or single-member files) and the Berkeley Earth
# observation file referenced above.
DATA_DIR  = "data/"
TREND_Y0, TREND_Y1 = 1970, 2023
BASE_Y0,  BASE_Y1  = 1850, 1899
trend_years = np.arange(TREND_Y0, TREND_Y1 + 1)

SSPS_KEEP = {"ssp126", "ssp245", "ssp370"}

_ar6   = regionmask.defined_regions.ar6.land
_i_cna = _ar6.abbrevs.index("CNA")

LABEL_SIZE   = 13
TICK_SIZE    = 12
TITLE_SIZE   = 12
LEGEND_SIZE  = 8    # smaller because many GCMs in legend
GCM_LEG_SIZE = 8


def _isel_years(da, y0, y1):
    yr  = da.time.dt.year.values
    idx = np.where((yr >= y0) & (yr <= y1))[0]
    return da.isel(time=idx)


def _cna_mean(da):
    """Area-weighted mean over the CNA AR6 land region (any regular grid)."""
    mask = _ar6.mask(da.lon.values, da.lat.values)
    da_m = da.where(mask == _i_cna)
    w = np.cos(np.deg2rad(da_m.lat))
    w.name = "weights"
    return da_m.weighted(w).mean(("lat", "lon"), skipna=True)


def _global_mean(da):
    w = np.cos(np.deg2rad(da.lat))
    w.name = "weights"
    return da.weighted(w).mean(("lat", "lon"), skipna=True)


def trend_per_decade(ts):
    s, *_ = stats.linregress(np.arange(len(ts)), ts)
    return s * 10.0


def percentile_of_value(dist, val):
    return float((dist < val).mean() * 100.0)


def pct_label(pct):
    n = math.ceil(pct)
    sfx = "th" if 11 <= n % 100 <= 13 else {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
    return f"{n}{sfx}"


# ============================================================================
# BUILD FILE CATALOG WITH DEDUPLICATION
# ============================================================================

def extract_ssp(fn):
    m = re.search(r'ssp(\d+)', os.path.basename(fn))
    return f"ssp{m.group(1)}" if m else None


def extract_gcm(fn):
    # Match: _ssp{NNN}_{GCM}_r
    m = re.search(r'_ssp\d+_([^_]+)_r', os.path.basename(fn))
    return m.group(1) if m else None


def n_members_in_file(fn):
    """Open file just enough to read the member dimension size."""
    try:
        da = xr.open_dataarray(fn, engine="netcdf4")
        n = da.sizes.get("member", 1)
        da.close()
        return n
    except Exception:
        try:
            ds = xr.open_dataset(fn, engine="netcdf4")
            n = ds.sizes.get("member", 1)
            ds.close()
            return n
        except Exception:
            return 0


print("Building file catalog ...")

all_files = glob.glob(DATA_DIR + "tas_Amon_hist*.nc")
# Keep only the 3 target SSPs, exclude ssp119
all_files = [f for f in all_files
             if extract_ssp(f) in SSPS_KEEP
             and "ssp119" not in f]

# Group by (gcm, ssp), keep the file with most members
catalog = {}   # (gcm, ssp) -> (n_members, filepath)

for fn in sorted(all_files):
    gcm = extract_gcm(fn)
    ssp = extract_ssp(fn)
    if gcm is None or ssp is None:
        continue
    key = (gcm, ssp)
    n = n_members_in_file(fn)
    if key not in catalog or n > catalog[key][0]:
        catalog[key] = (n, fn)

print(f"  Found {len(catalog)} (GCM, SSP) combinations after deduplication")
for (gcm, ssp), (n, fn) in sorted(catalog.items()):
    print(f"    {ssp}  {gcm:<25}: {n:>3} members — {os.path.basename(fn)}")

# Collect all unique GCM names for color mapping
all_gcms_sorted = sorted({gcm for (gcm, ssp) in catalog})
cmap = plt.cm.get_cmap("tab20", len(all_gcms_sorted))
GCM_COLORS = {gcm: cmap(i) for i, gcm in enumerate(all_gcms_sorted)}

# ============================================================================
# LOAD DATA AND COMPUTE TRENDS
# ============================================================================

global_trends = []   # list of floats
cna_trends    = []
gcm_labels    = []   # list of str (GCM name for each member)

print("\nProcessing CMIP6 files ...")
print("Order: local 1850-1899 baseline → subtract → CNA/global mean → trend\n")

for (gcm, ssp), (n_mem, fn) in sorted(catalog.items()):
    da = xr.open_dataarray(fn, engine="netcdf4")   # (member, time, lat, lon)

    # 1. Local 1850-1899 baseline per grid cell per member
    base  = _isel_years(da, BASE_Y0,  BASE_Y1).mean("time")   # (member, lat, lon)
    # 2. Subtract baseline
    anom  = da - base                                           # (member, time, lat, lon)
    # 3. Select 1970-2023
    anom_trend = _isel_years(anom, TREND_Y0, TREND_Y1)        # (member, 45-ish, lat, lon)

    n_actual = anom_trend.sizes["member"]

    # 4. Global and CNA spatial means → (member, n_years) as numpy
    global_ts = _global_mean(anom_trend).values   # (member, n_years)
    cna_ts    = _cna_mean(anom_trend).values

    # 5. Trend per member
    n_yrs = global_ts.shape[1]
    t     = np.arange(n_yrs)
    for m in range(n_actual):
        global_trends.append(trend_per_decade(global_ts[m]))
        cna_trends.append(trend_per_decade(cna_ts[m]))
        gcm_labels.append(gcm)

    da.close()
    print(f"  {ssp}  {gcm:<25}: {n_actual} members processed")

global_trends = np.array(global_trends)
cna_trends    = np.array(cna_trends)
gcm_labels    = np.array(gcm_labels)

print(f"\nTotal ensemble members: {len(global_trends)}")

# ============================================================================
# BERKELEY EARTH OBSERVATIONS
# ============================================================================

print("\nProcessing Berkeley Earth obs ...")

da_obs = xr.open_dataarray(
    DATA_DIR + "_Land_and_Ocean_LatLong1_185001_202312_ann_mean_2pt5degree.nc"
)

obs_base_global = _isel_years(da_obs, BASE_Y0, BASE_Y1).mean("time", skipna=True)
obs_base_cna    = obs_base_global   # same file, different spatial aggregation

obs_anom = da_obs - obs_base_global
obs_trend_slice = _isel_years(obs_anom, TREND_Y0, TREND_Y1)

obs_global_ts = _global_mean(obs_trend_slice).values
obs_cna_ts    = _cna_mean(obs_trend_slice).values

obs_global_trend = trend_per_decade(obs_global_ts)
obs_cna_trend    = trend_per_decade(obs_cna_ts)

print(f"  Obs global annual mean trend : {obs_global_trend:.4f} °C/decade")
print(f"  Obs CNA annual mean trend    : {obs_cna_trend:.4f} °C/decade")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_stats(label, obs_trend, model_trends):
    pct = percentile_of_value(model_trends, obs_trend)
    print(f"\n=== {label} ===")
    print(f"  Berkeley Earth trend   : {obs_trend:.4f} °C/decade")
    print(f"  Model mean             : {np.mean(model_trends):.4f} °C/decade")
    print(f"  Model median           : {np.median(model_trends):.4f} °C/decade")
    print(f"  Model  5th / 95th pct  : {np.percentile(model_trends,5):.4f} / {np.percentile(model_trends,95):.4f}")
    print(f"  Model 25th / 75th pct  : {np.percentile(model_trends,25):.4f} / {np.percentile(model_trends,75):.4f}")
    print(f"  Obs at the {pct_label(pct)} percentile of model distribution (raw: {pct:.1f}th)")
    print(f"  Total members          : {len(model_trends)}")

print_stats(f"Global annual mean tas, {TREND_Y0}-{TREND_Y1}", obs_global_trend, global_trends)
print_stats(f"CNA annual mean tas,    {TREND_Y0}-{TREND_Y1}", obs_cna_trend,    cna_trends)

# ============================================================================
# PLOT
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


def plot_panel(ax, model_trends, gcm_labels_arr, obs_trend,
               median_color, xlabel, panel_title):
    bins = np.histogram_bin_edges(model_trends, bins=35)
    trends_by_gcm = [model_trends[gcm_labels_arr == g] for g in all_gcms_sorted]
    colors_list   = [GCM_COLORS[g] for g in all_gcms_sorted]

    ax.hist(
        trends_by_gcm, bins=bins, stacked=True,
        color=colors_list, edgecolor="white", linewidth=0.2, alpha=0.9,
    )

    obs_line = ax.axvline(
        obs_trend, color="black", linestyle="--", linewidth=2.5,
        label=f"Berkeley Earth: {obs_trend:.3f} °C/decade",
    )
    med_line = ax.axvline(
        np.median(model_trends), color=median_color, linestyle=":", linewidth=2.5,
        label=f"Model median: {np.median(model_trends):.3f} °C/decade",
    )
    mean_line = ax.axvline(
        np.mean(model_trends), color=median_color, linestyle="-.", linewidth=1.5,
        label=f"Model mean:   {np.mean(model_trends):.3f} °C/decade",
    )

    pct = percentile_of_value(model_trends, obs_trend)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of ensemble members", fontsize=LABEL_SIZE)
    ax.set_title(
        f"{panel_title}\nObs at {pct_label(pct)} percentile  "
        f"(n = {len(model_trends)} members)",
        fontsize=TITLE_SIZE,
    )
    ax.legend(handles=[obs_line, med_line, mean_line],
              loc="upper right", fontsize=LEGEND_SIZE + 1)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


plot_panel(
    axes[0], global_trends, gcm_labels,
    obs_global_trend, median_color="#2471a3",
    xlabel=f"Global annual mean tas trend (°C/decade, rel. 1850–1899)",
    panel_title=f"(a) Global annual mean tas, {TREND_Y0}–{TREND_Y1}",
)
plot_panel(
    axes[1], cna_trends, gcm_labels,
    obs_cna_trend, median_color="#922b21",
    xlabel=f"CNA annual mean tas trend (°C/decade, rel. 1850–1899)",
    panel_title=f"(b) CNA annual mean tas, {TREND_Y0}–{TREND_Y1}",
)

# Consistent y-axis
y_max = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Shared GCM legend (two columns to fit many GCMs)
gcm_patches = [mpatches.Patch(color=GCM_COLORS[g], label=g) for g in all_gcms_sorted]
fig.legend(
    handles=gcm_patches,
    loc="lower center",
    ncol=min(len(all_gcms_sorted), 8),
    title="GCM  (SSP1-2.6, SSP2-4.5, SSP3-7.0 pooled)",
    bbox_to_anchor=(0.5, -0.12),
    fontsize=GCM_LEG_SIZE,
    title_fontsize=GCM_LEG_SIZE,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

FIG_OUT = "trend_bias_mapp_ssp126_245_370_1970_2023.png"
plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved {FIG_OUT}")
