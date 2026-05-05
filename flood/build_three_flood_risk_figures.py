from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

OUT_DIR = ROOT / "outputs" / "gini_income_map_periods" / "flood_risk_three_figures"
GNI_CACHE = ROOT / "outputs" / "gini_income_map_periods" / "correlation_analysis" / "worldbank_gni_pc.csv"
WB_GNI_URL = "https://api.worldbank.org/v2/country/all/indicator/NY.GNP.PCAP.CD?format=json&per_page=20000"

GADM_GPKG = Path("/Users/qijiayun/cursor project/era5-tools/output/gadm_admin1_dissolved.gpkg")
RAIN_PARQUET = Path("/Users/qijiayun/cursor project/era5-tools/output/gadm_admin1_monthly_precip_stats.parquet")
GINI_GPKG = Path("/Users/qijiayun/Downloads/polyg_adm1_gini_disp_1990_2023.gpkg")
FLOOD_CSV = Path("/Users/qijiayun/cursor project/era5-tools/output/emdat_gadm_match/matched.csv")
GINI_LAYER = "polyg_adm1_gini_disp_1990_2023"

YEARS = list(range(2000, 2026))
Gini_YEARS = [str(year) for year in range(2000, 2024)]
RAIN_LAYERS = ["<p90", "p90-p95", ">=p95"]
RAIN_BIN_PERCENTILES = [50, 60, 70, 80, 90, 95, 97, 99]
RAIN_BINS = ["p50-p60", "p60-p70", "p70-p80", "p80-p90", "p90-p95", "p95-p97", "p97-p99", "p99+"]


def load_gni_data() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if GNI_CACHE.exists():
        return pd.read_csv(GNI_CACHE)
    with urllib.request.urlopen(WB_GNI_URL, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for rec in payload[1]:
        iso3 = rec.get("countryiso3code")
        year = rec.get("date")
        value = rec.get("value")
        if iso3 and year and value is not None:
            rows.append({"iso3": iso3, "year": int(year), "gni_pc": float(value)})
    gni = pd.DataFrame(rows).drop_duplicates(["iso3", "year"])
    GNI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gni.to_csv(GNI_CACHE, index=False)
    return gni


def gni_lookup_table(gni: pd.DataFrame) -> dict[tuple[str, int], tuple[float | None, int | None]]:
    grouped = {iso3: sub.sort_values("year") for iso3, sub in gni.groupby("iso3")}
    lookup = {}
    for iso3, sub in grouped.items():
        for year in YEARS:
            target = min(year, 2024)
            candidates = sub[sub["year"] <= target]
            if candidates.empty:
                lookup[(iso3, year)] = (None, None)
            else:
                row = candidates.iloc[-1]
                lookup[(iso3, year)] = (float(row["gni_pc"]), int(row["year"]))
    return lookup


def load_rainfall_year() -> pd.DataFrame:
    rain = pd.read_parquet(
        RAIN_PARQUET,
        columns=["time", "country_code", "country_name", "admin1_code", "admin1_name", "max_mm_month"],
    )
    rain["year"] = rain["time"].dt.year
    rain = rain[rain["year"].isin(YEARS)].copy()
    annual = (
        rain.groupby(["admin1_code", "country_code", "country_name", "admin1_name", "year"], as_index=False)
        .agg(rain_max_month_mm=("max_mm_month", "max"))
    )
    rain_thresholds = {pct: annual["rain_max_month_mm"].quantile(pct / 100) for pct in RAIN_BIN_PERCENTILES}
    p90 = rain_thresholds[90]
    p95 = rain_thresholds[95]
    annual["rain_layer"] = np.select(
        [annual["rain_max_month_mm"] < p90, annual["rain_max_month_mm"] < p95],
        ["<p90", "p90-p95"],
        default=">=p95",
    )
    annual["rain_bin"] = pd.cut(
        annual["rain_max_month_mm"],
        bins=[
            rain_thresholds[50],
            rain_thresholds[60],
            rain_thresholds[70],
            rain_thresholds[80],
            rain_thresholds[90],
            rain_thresholds[95],
            rain_thresholds[97],
            rain_thresholds[99],
            np.inf,
        ],
        labels=RAIN_BINS,
        right=False,
    )
    return annual


def load_gadm_with_gini() -> gpd.GeoDataFrame:
    gadm = gpd.read_file(GADM_GPKG, engine="pyogrio").to_crs("EPSG:4326")
    gadm = gadm.rename(columns={"GID_1": "admin1_code", "GID_0": "country_code", "COUNTRY": "country_name", "NAME_1": "admin1_name"})
    gadm = gadm[gadm["admin1_code"].notna() & gadm["admin1_code"].ne("?")].copy()
    gadm["adm1_uid"] = np.arange(len(gadm))

    gini = gpd.read_file(GINI_GPKG, layer=GINI_LAYER, engine="pyogrio", encoding="latin1").to_crs("EPSG:4326")
    reps = gadm[["adm1_uid", "admin1_code", "geometry"]].copy()
    reps["geometry"] = reps.geometry.representative_point()
    joined = gpd.sjoin(
        reps,
        gini[["iso3", *Gini_YEARS, "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    joined = joined.drop_duplicates("adm1_uid")
    gadm = gadm.merge(joined.drop(columns=["geometry"]), on=["adm1_uid", "admin1_code"], how="left")
    return gadm


def clean_flood() -> pd.DataFrame:
    usecols = ["disasterno", "iso3", "disaster_type", "start_year", "total_affected", "adm1_id", "centroid_lon", "centroid_lat"]
    flood = pd.read_csv(FLOOD_CSV, usecols=usecols)
    flood = flood[flood["disaster_type"].astype(str).str.strip().eq("Flood")].copy()
    flood["year"] = pd.to_numeric(flood["start_year"], errors="coerce")
    flood["affected_num"] = pd.to_numeric(flood["total_affected"], errors="coerce")
    flood["lon"] = pd.to_numeric(flood["centroid_lon"], errors="coerce")
    flood["lat"] = pd.to_numeric(flood["centroid_lat"], errors="coerce")
    flood = flood.dropna(subset=["year", "affected_num", "lon", "lat", "adm1_id"]).copy()
    flood = flood[flood["lon"].between(-180, 180) & flood["lat"].between(-90, 90)].copy()
    flood["year"] = flood["year"].astype(int)
    return flood


def build_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    gni = load_gni_data()
    gni_lookup = gni_lookup_table(gni)
    rain = load_rainfall_year()
    gadm = load_gadm_with_gini()
    flood = clean_flood()

    panel = rain.merge(
        gadm.drop(columns="geometry"),
        on=["admin1_code", "country_code", "country_name", "admin1_name"],
        how="left",
    )
    gni_pairs = [gni_lookup.get((iso3, int(year)), (None, None)) for iso3, year in zip(panel["country_code"], panel["year"], strict=True)]
    panel["gni_pc"] = [item[0] for item in gni_pairs]
    panel["gni_year_used"] = [item[1] for item in gni_pairs]
    panel["log_gni_pc"] = np.log(panel["gni_pc"])
    panel["gini_year_used"] = panel["year"].clip(upper=2023)
    panel["gini"] = [row.get(str(int(row["gini_year_used"])), np.nan) for _, row in panel.iterrows()]

    flood_in_years = flood[flood["year"].isin(YEARS)].copy()
    counts = (
        flood_in_years.groupby(["adm1_id", "year"])["disasterno"]
        .agg(flood_point_count="size", unique_disasterno_count="nunique")
        .reset_index()
        .rename(columns={"adm1_id": "admin1_code"})
    )
    panel = panel.merge(counts, on=["admin1_code", "year"], how="left")
    panel["flood_point_count"] = panel["flood_point_count"].fillna(0).astype(int)
    panel["unique_disasterno_count"] = panel["unique_disasterno_count"].fillna(0).astype(int)
    panel["flood_any"] = (panel["flood_point_count"] > 0).astype(int)

    qa = {
        "gni_rows": len(gni),
        "rain_rows": len(rain),
        "gadm_rows": len(gadm),
        "panel_rows": len(panel),
        "flood_rows_clean": len(flood),
        "flood_rows_in_panel_years": len(flood_in_years),
        "flood_rows_excluded_no_rain_year": int((~flood["year"].isin(YEARS)).sum()),
        "panel_unique_admin1_year": bool(panel[["admin1_code", "year"]].duplicated().sum() == 0),
        "flood_any_check": bool((panel["flood_any"].eq(panel["flood_point_count"].gt(0).astype(int))).all()),
        "gni_match_rate": float(panel["gni_pc"].notna().mean()),
        "gini_match_rate": float(panel["gini"].notna().mean()),
        "rain_match_rate": float(panel["rain_max_month_mm"].notna().mean()),
        "rain_layer_counts": panel["rain_layer"].value_counts().to_dict(),
        "rain_bin_counts": panel["rain_bin"].value_counts().reindex(RAIN_BINS, fill_value=0).to_dict(),
        "rain_bin_thresholds": {f"p{pct}": float(panel["rain_max_month_mm"].quantile(pct / 100)) for pct in RAIN_BIN_PERCENTILES},
        "rain_p90": float(panel["rain_max_month_mm"].quantile(0.90)),
        "rain_p95": float(panel["rain_max_month_mm"].quantile(0.95)),
    }
    return panel, qa


def wilson_ci(successes: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return center - half, center + half


def decile_summary(df: pd.DataFrame, var: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    frames = []
    if group_cols:
        iterator = df.groupby(group_cols, dropna=False)
    else:
        iterator = [((), df)]
    for keys, sub in iterator:
        sub = sub.dropna(subset=[var, "flood_any"]).copy()
        if sub[var].nunique() < 10:
            continue
        sub["decile"] = pd.qcut(sub[var], 10, labels=False, duplicates="drop") + 1
        grouped = (
            sub.groupby("decile", as_index=False)
            .agg(x_mean=(var, "mean"), n=("flood_any", "size"), flood_sum=("flood_any", "sum"), p_flood=("flood_any", "mean"))
        )
        lo, hi = wilson_ci(grouped["flood_sum"], grouped["n"])
        grouped["ci_low"] = lo
        grouped["ci_high"] = hi
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col, key in zip(group_cols, keys, strict=True):
                grouped[col] = key
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_global_decile(df: pd.DataFrame, var: str, decile_col: str) -> pd.DataFrame:
    out = df.copy()
    valid = out[var].notna() & out["flood_any"].notna()
    out.loc[valid, decile_col] = pd.qcut(out.loc[valid, var], 10, labels=False, duplicates="drop") + 1
    return out


def rainfall_decile_summary(df: pd.DataFrame, var: str, decile_col: str, global_summary: pd.DataFrame, rain_col: str = "rain_layer") -> pd.DataFrame:
    sub = df.dropna(subset=[var, "flood_any", rain_col, decile_col]).copy()
    sub[decile_col] = sub[decile_col].astype(int)
    grouped = (
        sub.groupby([rain_col, decile_col], as_index=False, observed=True)
        .agg(n=("flood_any", "size"), flood_sum=("flood_any", "sum"), p_flood=("flood_any", "mean"))
        .rename(columns={decile_col: "decile"})
    )
    lo, hi = wilson_ci(grouped["flood_sum"], grouped["n"])
    grouped["ci_low"] = lo
    grouped["ci_high"] = hi
    grouped = grouped.merge(global_summary[["decile", "x_mean"]], on="decile", how="left", validate="many_to_one")
    return grouped[[rain_col, "decile", "x_mean", "n", "flood_sum", "p_flood", "ci_low", "ci_high"]]


def pooled_probability(summary: pd.DataFrame, deciles: list[int]) -> tuple[float, int, int]:
    sub = summary[summary["decile"].isin(deciles)]
    n = int(sub["n"].sum())
    flood_sum = int(sub["flood_sum"].sum())
    return (float(flood_sum / n) if n else np.nan), flood_sum, n


def slope_by_bin(summary: pd.DataFrame, layer_col: str) -> pd.Series:
    slopes = {}
    for layer, sub in summary.groupby(layer_col, observed=True):
        sub = sub.sort_values("decile")
        x = sub["decile"].to_numpy(dtype=float)
        y = sub["p_flood"].to_numpy(dtype=float)
        if len(sub) < 2 or np.allclose(x, x[0]):
            slopes[layer] = np.nan
        else:
            slopes[layer] = float(np.polyfit(x, y, 1)[0])
    return pd.Series(slopes)


def lowess_endpoint_gap_by_bin(summary: pd.DataFrame, layer_col: str) -> pd.Series:
    gaps = {}
    for layer, sub in summary.groupby(layer_col, observed=True):
        sub = sub.sort_values("decile")
        if len(sub) < 3:
            gaps[layer] = np.nan
            continue
        xline, yline = lowess_line(sub["decile"].to_numpy(dtype=float), sub["p_flood"].to_numpy(dtype=float))
        gaps[layer] = float(yline[-1] - yline[0])
    return pd.Series(gaps)


def disparity_summary(income_bins: pd.DataFrame, gini_bins: pd.DataFrame) -> pd.DataFrame:
    income_slope = slope_by_bin(income_bins, "rain_bin")
    gini_slope = slope_by_bin(gini_bins, "rain_bin")
    income_lowess_gap = lowess_endpoint_gap_by_bin(income_bins, "rain_bin")
    gini_lowess_gap = lowess_endpoint_gap_by_bin(gini_bins, "rain_bin")
    rows = []
    for rain_bin in RAIN_BINS:
        income_sub = income_bins[income_bins["rain_bin"].eq(rain_bin)]
        gini_sub = gini_bins[gini_bins["rain_bin"].eq(rain_bin)]
        income_low_p, income_low_flood, income_low_n = pooled_probability(income_sub, [1, 2, 3])
        income_high_p, income_high_flood, income_high_n = pooled_probability(income_sub, [8, 9, 10])
        gini_low_p, gini_low_flood, gini_low_n = pooled_probability(gini_sub, [1, 2, 3])
        gini_high_p, gini_high_flood, gini_high_n = pooled_probability(gini_sub, [8, 9, 10])
        rows.append(
            {
                "rain_bin": rain_bin,
                "income_low_decile_1_3_p_flood": income_low_p,
                "income_low_decile_1_3_n": income_low_n,
                "income_low_decile_1_3_flood_sum": income_low_flood,
                "income_high_decile_8_10_p_flood": income_high_p,
                "income_high_decile_8_10_n": income_high_n,
                "income_high_decile_8_10_flood_sum": income_high_flood,
                "income_gap_low_minus_high": income_low_p - income_high_p,
                "income_linear_slope_per_decile": income_slope.get(rain_bin, np.nan),
                "income_lowess_endpoint_gap_d10_minus_d1": income_lowess_gap.get(rain_bin, np.nan),
                "gini_low_decile_1_3_p_flood": gini_low_p,
                "gini_low_decile_1_3_n": gini_low_n,
                "gini_low_decile_1_3_flood_sum": gini_low_flood,
                "gini_high_decile_8_10_p_flood": gini_high_p,
                "gini_high_decile_8_10_n": gini_high_n,
                "gini_high_decile_8_10_flood_sum": gini_high_flood,
                "gini_gap_high_minus_low": gini_high_p - gini_low_p,
                "gini_linear_slope_per_decile": gini_slope.get(rain_bin, np.nan),
                "gini_lowess_endpoint_gap_d10_minus_d1": gini_lowess_gap.get(rain_bin, np.nan),
            }
        )
    return pd.DataFrame(rows)


def lowess_line(x: np.ndarray, y: np.ndarray, frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = len(x)
    if n < 3:
        return x, y
    span = max(3, math.ceil(frac * n))
    yfit = np.zeros(n)
    for i in range(n):
        distances = np.abs(x - x[i])
        width = np.partition(distances, min(span - 1, n - 1))[min(span - 1, n - 1)]
        if width == 0:
            weights = (distances == 0).astype(float)
        else:
            u = np.clip(distances / width, 0, 1)
            weights = (1 - u**3) ** 3
        x_centered = x - x[i]
        design = np.column_stack([np.ones(n), x_centered])
        try:
            beta = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
            yfit[i] = beta[0]
        except np.linalg.LinAlgError:
            yfit[i] = np.average(y, weights=weights)
    return x, yfit


def size_from_n(n: pd.Series, min_size: float = 35, max_size: float = 240) -> pd.Series:
    root = np.sqrt(n.astype(float))
    if root.max() == root.min():
        return pd.Series((min_size + max_size) / 2, index=n.index)
    return min_size + (root - root.min()) / (root.max() - root.min()) * (max_size - min_size)


def plot_single(summary: pd.DataFrame, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    sizes = size_from_n(summary["n"])
    ax.errorbar(summary["x_mean"], summary["p_flood"], yerr=[summary["p_flood"] - summary["ci_low"], summary["ci_high"] - summary["p_flood"]], fmt="none", ecolor="#555555", elinewidth=1, capsize=3, zorder=1)
    ax.scatter(summary["x_mean"], summary["p_flood"], s=sizes, color="#4c78a8", alpha=0.85, edgecolor="white", linewidth=0.8, zorder=2)
    xline, yline = lowess_line(summary["x_mean"].to_numpy(), summary["p_flood"].to_numpy())
    ax.plot(xline, yline, color="#1f1f1f", linewidth=2, label="LOWESS trend")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(Flood)")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    ax.text(0, -0.20, "P(Flood) = mean(flood_any) within each decile. Error bars are Wilson 95% CI. Point size scales with sqrt(n). Descriptive, not causal.", transform=ax.transAxes, fontsize=8.2, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_rainfall(summary: pd.DataFrame, path: Path, title: str, xlabel: str, reverse_x: bool = False, layer_col: str = "rain_layer", layer_order: list[str] | None = None, colors: dict[str, str] | None = None, legend_title: str = "Annual rainfall intensity", footnote: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    layer_order = layer_order or RAIN_LAYERS
    colors = colors or {"<p90": "#6b8fb3", "p90-p95": "#d18f32", ">=p95": "#8f2d56"}
    for layer in layer_order:
        sub = summary[summary[layer_col].eq(layer)].copy()
        if sub.empty:
            continue
        sizes = size_from_n(sub["n"], 35, 190)
        ax.errorbar(sub["x_mean"], sub["p_flood"], yerr=[sub["p_flood"] - sub["ci_low"], sub["ci_high"] - sub["p_flood"]], fmt="none", ecolor=colors[layer], alpha=0.55, elinewidth=1, capsize=2, zorder=1)
        ax.scatter(sub["x_mean"], sub["p_flood"], s=sizes, color=colors[layer], alpha=0.78, edgecolor="white", linewidth=0.7, label=layer, zorder=2)
        xline, yline = lowess_line(sub["x_mean"].to_numpy(), sub["p_flood"].to_numpy())
        ax.plot(xline, yline, color=colors[layer], linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(Flood)")
    if reverse_x:
        ax.invert_xaxis()
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(title=legend_title, frameon=False, ncols=2 if len(layer_order) > 4 else 1)
    if footnote is None:
        footnote = "Rainfall layers are mutually exclusive. P(Flood) = mean(flood_any). Deciles are defined globally, not separately within rainfall layers.\nWilson 95% CI; point size scales with sqrt(n). Descriptive, not causal."
    ax.text(0, -0.24, footnote, transform=ax.transAxes, fontsize=8.2, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.24)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_disparity(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(summary))
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--", zorder=1)
    ax.plot(x, summary["income_gap_low_minus_high"], marker="o", linewidth=2.4, color="#4c78a8", label="Income gap: low-income d1-3 minus high-income d8-10", zorder=2)
    ax.plot(x, summary["gini_gap_high_minus_low"], marker="o", linewidth=2.4, color="#b0527a", label="Gini gap: high-inequality d8-10 minus low-inequality d1-3", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["rain_bin"], rotation=30, ha="right")
    ax.set_title("Socioeconomic Flood-Risk Gap by Rainfall Percentile Bin", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rainfall percentile bin")
    ax.set_ylabel("P(Flood) gap")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    ax.text(0, -0.28, "Positive income gap means lower-income deciles have higher P(Flood). Positive Gini gap means higher-inequality deciles have higher P(Flood).\nRainfall bins are global ADM1-year percentiles over annual maximum monthly rainfall. Descriptive, not causal.", transform=ax.transAxes, fontsize=8.2, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def x_positions_match(layered: pd.DataFrame, global_summary: pd.DataFrame) -> bool:
    merged = layered[["decile", "x_mean"]].merge(global_summary[["decile", "x_mean"]], on="decile", how="left", suffixes=("_layered", "_global"), validate="many_to_one")
    return bool(np.allclose(merged["x_mean_layered"], merged["x_mean_global"], equal_nan=False))


def ci_in_unit_interval(*summaries: pd.DataFrame) -> bool:
    return bool(all(summary["ci_low"].between(0, 1).all() and summary["ci_high"].between(0, 1).all() for summary in summaries))


def write_qa(panel: pd.DataFrame, qa: dict[str, object], income: pd.DataFrame, gini: pd.DataFrame, income_rain: pd.DataFrame, gini_rain: pd.DataFrame, income_bins: pd.DataFrame, gini_bins: pd.DataFrame, disparity: pd.DataFrame) -> None:
    rain_layer_coverage = bool(panel.loc[panel["rain_max_month_mm"].notna(), "rain_layer"].notna().all())
    rain_layer_known = bool(panel["rain_layer"].dropna().isin(RAIN_LAYERS).all())
    p50 = qa["rain_bin_thresholds"]["p50"]
    p50_panel = panel[panel["rain_max_month_mm"].ge(p50)].copy()
    rain_bin_coverage = bool(p50_panel["rain_bin"].notna().all())
    rain_bin_known = bool(panel["rain_bin"].dropna().isin(RAIN_BINS).all())
    rain_layer_counts = panel.loc[panel["rain_max_month_mm"].notna(), "rain_layer"].value_counts().reindex(RAIN_LAYERS, fill_value=0)
    rain_bin_counts = panel.loc[panel["rain_bin"].notna(), "rain_bin"].value_counts().reindex(RAIN_BINS, fill_value=0)
    income_counts = income_rain.pivot(index="decile", columns="rain_layer", values="n").reindex(columns=RAIN_LAYERS)
    gini_counts = gini_rain.pivot(index="decile", columns="rain_layer", values="n").reindex(columns=RAIN_LAYERS)
    income_bin_counts = income_bins.pivot(index="decile", columns="rain_bin", values="n").reindex(columns=RAIN_BINS)
    gini_bin_counts = gini_bins.pivot(index="decile", columns="rain_bin", values="n").reindex(columns=RAIN_BINS)
    threshold_lines = [f"  {pct}: {value:.3f}" for pct, value in qa["rain_bin_thresholds"].items()]
    lines = [
        "Flood risk figures QA",
        "=" * 72,
        f"ADM1-year panel rows: {len(panel)}",
        f"Unique admin1_code-year: {qa['panel_unique_admin1_year']}",
        f"flood_any check: {qa['flood_any_check']}",
        f"GNI match rate: {qa['gni_match_rate']:.3f}",
        f"Gini match rate: {qa['gini_match_rate']:.3f}",
        f"Rainfall match rate: {qa['rain_match_rate']:.3f}",
        f"Flood rows clean: {qa['flood_rows_clean']}",
        f"Flood rows in 2000-2025 panel years: {qa['flood_rows_in_panel_years']}",
        f"Flood rows excluded because rainfall lacks year: {qa['flood_rows_excluded_no_rain_year']}",
        f"Rain p90: {qa['rain_p90']:.3f}",
        f"Rain p95: {qa['rain_p95']:.3f}",
        f"Rain layer counts: {qa['rain_layer_counts']}",
        f"Rain bin counts p50+: {qa['rain_bin_counts']}",
        "",
        "Rain percentile thresholds:",
        *threshold_lines,
        "",
        "Decile summaries:",
        f"  income deciles rows: {len(income)}; min n={income['n'].min()}, max n={income['n'].max()}",
        f"  gini deciles rows: {len(gini)}; min n={gini['n'].min()}, max n={gini['n'].max()}",
        f"  income by rainfall rows: {len(income_rain)}; min n={income_rain['n'].min()}, max n={income_rain['n'].max()}",
        f"  gini by rainfall rows: {len(gini_rain)}; min n={gini_rain['n'].min()}, max n={gini_rain['n'].max()}",
        f"  income by rainfall bins rows: {len(income_bins)}; min n={income_bins['n'].min()}, max n={income_bins['n'].max()}",
        f"  gini by rainfall bins rows: {len(gini_bins)}; min n={gini_bins['n'].min()}, max n={gini_bins['n'].max()}",
        f"  rainfall bin disparity rows: {len(disparity)}",
        "",
        "QA checks:",
        f"  Figure 3 decile x positions match decile_summary_income.csv: {x_positions_match(income_rain, income)}",
        f"  Figure 4 decile x positions match decile_summary_gini.csv: {x_positions_match(gini_rain, gini)}",
        f"  Figure 5 decile x positions match decile_summary_income.csv: {x_positions_match(income_bins, income)}",
        f"  Figure 6 decile x positions match decile_summary_gini.csv: {x_positions_match(gini_bins, gini)}",
        f"  Rainfall layers mutually exclusive and known: {rain_layer_known}",
        f"  Rainfall layers cover all ADM1-years with rainfall data: {rain_layer_coverage}",
        f"  Rainfall bins p50+ mutually exclusive and known: {rain_bin_known}",
        f"  Rainfall bins cover all p50+ ADM1-years with rainfall data: {rain_bin_coverage}",
        f"  Wilson CI in [0, 1]: {ci_in_unit_interval(income, gini, income_rain, gini_rain, income_bins, gini_bins)}",
        "",
        "Rain layer counts among ADM1-years with rainfall data:",
        rain_layer_counts.to_string(),
        "",
        "Rain bin counts among p50+ ADM1-years with rainfall data:",
        rain_bin_counts.to_string(),
        "",
        "Sample size by rain_layer x income decile:",
        income_counts.to_string(),
        "",
        "Sample size by rain_layer x gini decile:",
        gini_counts.to_string(),
        "",
        "Sample size by rain_bin x income decile:",
        income_bin_counts.to_string(),
        "",
        "Sample size by rain_bin x gini decile:",
        gini_bin_counts.to_string(),
        "",
        "Rainfall bin disparity summary:",
        disparity.to_string(index=False),
        "",
        "Notes:",
        "  P(Flood) = mean(flood_any), where flood_any = flood_point_count > 0.",
        "  Deciles are defined globally, not separately within rainfall layers.",
        "  Rainfall bins are based on global ADM1-year rainfall percentiles over annual maximum monthly rainfall.",
        "  The rainfall-bin mechanism analysis keeps p50+ bins only.",
        "  Point size scales with sqrt(n).",
        "  Error bars are Wilson 95% CI.",
        "  Trend lines use a small dependency-free LOWESS implementation.",
        "  p99+ and high-percentile-by-decile cells have smaller n and wider uncertainty.",
        "  Results are descriptive, not causal.",
    ]
    (OUT_DIR / "qa_three_figures.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, qa = build_panel()
    panel = add_global_decile(panel, "log_gni_pc", "gni_decile_global")
    panel = add_global_decile(panel, "gini", "gini_decile_global")
    panel.to_csv(OUT_DIR / "adm1_year_panel.csv", index=False)

    income_summary = decile_summary(panel, "log_gni_pc")
    gini_summary = decile_summary(panel, "gini")
    income_rain_summary = rainfall_decile_summary(panel, "log_gni_pc", "gni_decile_global", income_summary)
    gini_rain_summary = rainfall_decile_summary(panel, "gini", "gini_decile_global", gini_summary)
    income_rain_bin_summary = rainfall_decile_summary(panel, "log_gni_pc", "gni_decile_global", income_summary, "rain_bin")
    gini_rain_bin_summary = rainfall_decile_summary(panel, "gini", "gini_decile_global", gini_summary, "rain_bin")
    rain_bin_disparity = disparity_summary(income_rain_bin_summary, gini_rain_bin_summary)
    income_summary.to_csv(OUT_DIR / "decile_summary_income.csv", index=False)
    gini_summary.to_csv(OUT_DIR / "decile_summary_gini.csv", index=False)
    income_rain_summary.to_csv(OUT_DIR / "decile_summary_income_by_rainfall.csv", index=False)
    gini_rain_summary.to_csv(OUT_DIR / "decile_summary_gini_by_rainfall.csv", index=False)
    income_rain_bin_summary.to_csv(OUT_DIR / "decile_summary_income_by_rainfall_bins.csv", index=False)
    gini_rain_bin_summary.to_csv(OUT_DIR / "decile_summary_gini_by_rainfall_bins.csv", index=False)
    rain_bin_disparity.to_csv(OUT_DIR / "rainfall_bin_disparity_summary.csv", index=False)

    plot_single(income_summary, OUT_DIR / "figure1_income_vs_flood_risk.png", "Income vs Flood Risk", "Mean log(GNI per capita) within decile")
    plot_single(gini_summary, OUT_DIR / "figure2_gini_vs_flood_risk.png", "Gini vs Flood Risk", "Mean Gini within decile")
    plot_rainfall(income_rain_summary, OUT_DIR / "figure3_income_vs_flood_risk_by_rainfall.png", "Income vs Flood Risk by Rainfall Intensity", "Mean log(GNI per capita) within global decile")
    plot_rainfall(gini_rain_summary, OUT_DIR / "figure4_gini_vs_flood_risk_by_rainfall.png", "Gini vs Flood Risk by Rainfall Intensity", "Mean Gini within global decile (lower inequality to the right)", reverse_x=True)

    rain_bin_colors = {"p50-p60": "#4c78a8", "p60-p70": "#72b7b2", "p70-p80": "#54a24b", "p80-p90": "#eeca3b", "p90-p95": "#f58518", "p95-p97": "#e45756", "p97-p99": "#b279a2", "p99+": "#7f3c8d"}
    rain_bin_footnote = "Rainfall bins are mutually exclusive p50+ global ADM1-year percentiles. P(Flood) = mean(flood_any). Deciles are defined globally, not separately within rainfall bins.\nWilson 95% CI; point size scales with sqrt(n). p99+ cells have smaller n and wider uncertainty. Descriptive, not causal."
    plot_rainfall(income_rain_bin_summary, OUT_DIR / "figure5_income_vs_flood_risk_by_rainfall_bins.png", "Income vs Flood Risk by Detailed Rainfall Percentile Bin", "Mean log(GNI per capita) within global decile", layer_col="rain_bin", layer_order=RAIN_BINS, colors=rain_bin_colors, legend_title="Rainfall percentile bin", footnote=rain_bin_footnote)
    plot_rainfall(gini_rain_bin_summary, OUT_DIR / "figure6_gini_vs_flood_risk_by_rainfall_bins.png", "Gini vs Flood Risk by Detailed Rainfall Percentile Bin", "Mean Gini within global decile (lower inequality to the right)", reverse_x=True, layer_col="rain_bin", layer_order=RAIN_BINS, colors=rain_bin_colors, legend_title="Rainfall percentile bin", footnote=rain_bin_footnote)
    plot_disparity(rain_bin_disparity, OUT_DIR / "figure7_rainfall_bin_disparity_gap.png")
    write_qa(panel, qa, income_summary, gini_summary, income_rain_summary, gini_rain_summary, income_rain_bin_summary, gini_rain_bin_summary, rain_bin_disparity)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
