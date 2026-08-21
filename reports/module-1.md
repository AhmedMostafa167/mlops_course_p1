# Module 1 Baseline

## Validation metrics

- **Dataset:** NYC TLC green taxi trips, 2024-01

- **Features:** `PU_DO`, `trip_distance`

- **Model:** `DictVectorizer` + `LinearRegression`

- **Validation split:** chronological final 20% of filtered records

- **Validation RMSE:** 6.4647 minutes

- **Validation MAE:** 3.8969 minutes

## Reproduction

Run `python -m prodml.train` to download the official Parquet file, reproduce these metrics, and overwrite `models/model.pkl`.