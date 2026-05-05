# Flood Risk Figures

This directory contains the code used to build the seven flood-risk figures and supporting QA tables.

## What it produces

The script writes outputs under:

```text
outputs/gini_income_map_periods/flood_risk_three_figures
```

Main figures:

- `figure1_income_vs_flood_risk.png`
- `figure2_gini_vs_flood_risk.png`
- `figure3_income_vs_flood_risk_by_rainfall.png`
- `figure4_gini_vs_flood_risk_by_rainfall.png`
- `figure5_income_vs_flood_risk_by_rainfall_bins.png`
- `figure6_gini_vs_flood_risk_by_rainfall_bins.png`
- `figure7_rainfall_bin_disparity_gap.png`

Supporting CSV/QA outputs include the ADM1-year panel, decile summaries, rainfall-bin summaries, disparity diagnostics, and `qa_three_figures.txt`.

## Data definitions

- Analysis unit: ADM1-year.
- Flood outcome: `flood_any = flood_point_count > 0`.
- Income: World Bank GNI per capita, indicator `NY.GNP.PCAP.CD`, plotted as `log(GNI per capita)`.
- Inequality: same-year Gini; 2024-2026 use 2023 Gini.
- Rainfall intensity: annual maximum monthly rainfall within each ADM1-year.
- Main rainfall layers: `<p90`, `p90-p95`, `>=p95`.
- Detailed rainfall bins for mechanism analysis: `p50-p60`, `p60-p70`, `p70-p80`, `p80-p90`, `p90-p95`, `p95-p97`, `p97-p99`, `p99+`.

All income and Gini deciles are defined globally over the full available sample, not separately within rainfall layers or rainfall bins.

## Run

From the repository root:

```bash
python flood/build_three_flood_risk_figures.py
```

The script expects the source geospatial, rainfall, Gini, and flood-match datasets at the absolute paths declared near the top of `build_three_flood_risk_figures.py`. Update those constants if running on another machine.

## Python dependencies

See `requirements.txt`. A geopandas stack with `pyogrio` and parquet support is required.
