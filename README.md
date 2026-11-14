# Data-Driven Wind Turbine Performance Monitoring to Reduce Clean Energy Losses (SDG 7)

** Machine Learning **

A machine learning pipeline that uses wind turbine SCADA data to separate abnormal operating states from normal behaviour, learn the turbine's realistic "healthy" power curve, and estimate how much clean energy (and CO₂) is lost so that maintenance can be prioritised.

## Problem

Wind turbines are expected to follow the manufacturer's theoretical power curve, but in practice they often produce less because of faults, curtailment, shutdowns and gradual decline. SCADA records mix all of these with normal operation and carry **no fault labels**, so operators cannot easily tell which losses are recoverable.

For the turbine in this dataset (2018):

| Measure                                                   | Value                              |
| --------------------------------------------------------- | ---------------------------------- |
| Actual energy generated                                   | ~11,013 MWh                        |
| Expected from manufacturer curve                          | ~12,567 MWh                        |
| Shortfall (sum of below-curve intervals)                  | ~1,600 MWh (12.7 %)                |
| Zero/negative output with wind ≥ 3.5 m/s                  | 2,220 intervals (~370 h, ~456 MWh) |
| Equivalent CO₂ (0.740 tCO₂/MWh, Peninsular Malaysia grid) | ~1,180 t                           |

## Approach

1. **Clustering (unsupervised):** K-Means and DBSCAN group the 10-minute records into suspected operating states (normal, shutdown-like, curtailment-like, underperforming).
2. **Regression (supervised):** Random Forest and XGBoost learn the healthy power curve from normal-state records only, validated with month-blocked cross-validation (MAE, RMSE, R²).
3. **Loss quantification:** `max(expected − actual, 0) × 1/6 h` per record, reported in MWh and tonnes of CO₂ by state, month, wind speed bin and direction sector, using both the manufacturer and ML baselines.

States are inferred from data alone, so they are reported as **suspected states, not confirmed faults**.

## Dataset

- **Source:** [Wind Turbine SCADA Dataset — Berk Erisen, Kaggle](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)
- **Coverage:** 1 Jan – 31 Dec 2018, one 3.6 MW turbine in Turkey
- **Size:** 50,530 ten-minute records, 5 columns

| Column                          | Meaning                                                                 |
| ------------------------------- | ----------------------------------------------------------------------- |
| `Date/Time`                     | Timestamp (DD MM YYYY HH:MM)                                            |
| `LV ActivePower (kW)`           | Actual power output — regression target                                 |
| `Wind Speed (m/s)`              | Hub wind speed                                                          |
| `Theoretical_Power_Curve (KWh)` | Manufacturer expected output (treated as kW; max = 3,600 = rated power) |
| `Wind Direction (°)`            | Direction the wind blows from                                           |

## Repository structure

```
windturbine/
├── wind_turbine.csv               # Raw SCADA data (Kaggle)
├── notebook/
│   ├── 01_windturbine.ipynb       # Data understanding, preprocessing, feature engineering
│   ├── 02_clustering.ipynb        # K-Means vs DBSCAN, normal-operation cluster
│   ├── 03_regression.ipynb        # Random Forest vs XGBoost vs baselines, error analysis, explainability
│   └── 04_loss_analysis.ipynb     # Energy loss, stop events, Pareto ranking, CO₂ scenarios
├── out/
│   ├── 01_preprocessing/          # Cleaned feature table, scaled clustering input, scaler, quality summary
│   ├── 02_clustering/             # Clustered feature table, silhouette comparisons
│   ├── 03_regression/             # Expected power per record, model comparison, error tables, importances
│   └── 04_loss_analysis/          # Loss tables, stop events, Pareto list, CO₂ scenarios
└── figures/                       # One file per report figure
```

Every figure in the report is produced by a notebook and saved under the report's figure number:

| Report figure | File | Notebook |
| --- | --- | --- |
| 2.1 – 2.5 | `fig2-1_wind_speed_distribution.png`, `fig2-2_active_power_distribution.png`, `fig2-3_power_curve.png`, `fig2-4_monthly_mean_power.png`, `fig2-5_feature_scales.png` | 01 |
| 3.1 | `fig3-1_coverage_by_month.png` | 01 |
| 4.1, 4.2 | `fig4-1_wind_rose.png`, `fig4-2_rule_states.png` | 01 |
| 6.2 | `fig6-2_dbscan_groups.png` | 02 |
| 6.1, 6.3, 6.4 | `fig6-1_actual_vs_predicted.png`, `fig6-3_error_by_month.png`, `fig6-4_explainability.png` | 03 |
| 7.1 – 7.5 | `fig7-1_loss_by_state.png`, `fig7-2_pareto_loss.png`, `fig7-3_loss_by_month.png`, `fig7-4_loss_by_direction.png`, `fig7-5_loss_by_wind_speed.png` | 04 |

## Getting started

```bash
pip install numpy pandas matplotlib scikit-learn joblib xgboost jupyter
cd notebook
jupyter notebook
```

Run the notebooks in order, 01 to 04, from inside the `notebook/` folder. Each one reads the previous stage's files from `out/` and writes its own tables to `out/` and figures to `figures/`.

## Preprocessing summary

| Check                        | Result                     | Action                                        |
| ---------------------------- | -------------------------- | --------------------------------------------- |
| Unparseable timestamps       | 0                          | —                                             |
| Null values                  | 0                          | No imputation needed                          |
| Duplicate rows / timestamps  | 0 / 0                      | —                                             |
| Missing 10-minute slots      | 2,030 (3.9 %)              | **Not imputed**; flagged via `gap_before_min` |
| Largest gap                  | 625 slots (~4.3 days, Jan) | Reported as a limitation                      |
| Physically impossible values | 0                          | Range rules on speed, direction and power     |
| Negative power records       | 57 (min −2.47 kW)          | **Kept** — idle/shutdown signal               |

Key decisions:

- Only impossible values are removed. Zero output, curtailment plateaus and underperformance are what the project is trying to detect, so they stay.
- Cut-in speed is read from the manufacturer curve (3.0 m/s); the operating region is **≥ 3.5 m/s** (cut-in + 0.5 m/s buffer).
- Theoretical power is used only as a reference and for feature engineering, **never as a regression input**.
- Time features (`month`, `hour`) are for reporting and CV grouping only, to avoid leakage.

## Engineered features

| Feature                                                     | Role                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------ |
| `power_ratio` (actual ÷ theoretical, operating region only) | Clustering input                                       |
| `residual_kw` (theoretical − actual)                        | Clustering input                                       |
| `wd_sin`, `wd_cos` (circular wind direction)                | Regression input; optional clustering input            |
| `ws_bin` (0.5 m/s), `dir_sector` (30°)                      | Reporting                                              |
| `actual_kwh`, `theo_kwh`, `shortfall_kwh`                   | Loss quantification                                    |
| `in_operating_region`, `rule_state`                         | Subset selection and physical rule check (not a label) |

Full details are in [`out/01_preprocessing/feature_dictionary.csv`](out/01_preprocessing/feature_dictionary.csv).

### Rule-based reference states

A simple physical-rule reference used to sanity-check clusters (not used as labels):

| rule_state            | Records | Shortfall (MWh) | Share of shortfall |
| --------------------- | ------: | --------------: | -----------------: |
| near_curve            |  31,197 |           700.2 |             43.8 % |
| zero_output           |   2,220 |           456.1 |             28.5 % |
| underperforming       |   7,218 |           436.0 |             27.3 % |
| below_operating_range |   9,895 |             7.6 |              0.5 % |

![Rule-based reference states](figures/fig4-2_rule_states.png)

`near_curve` still holds a large share of the loss because it includes the curtailment-like plateau at ~3,450 kW above rated wind speed. Separating that is the job of clustering.
