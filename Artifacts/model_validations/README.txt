This folder stores the output of the Inference Pipeline.

Every time you run inference, TWO files are auto-generated here with a timestamp:

  predictions_YYYY-MM-DD_HH-MM-SS.csv
  ├── engine_id       : Engine identifier from input CSV
  ├── cycle           : Cycle number
  ├── true_RUL        : Proxy RUL computed from input data (max_cycle - cycle)
  └── predicted_RUL   : RUL predicted by the trained LightGBM model

  metrics_YYYY-MM-DD_HH-MM-SS.csv
  ├── engine_id       : Engine identifier (one row per engine + one OVERALL row)
  ├── num_cycles      : Number of cycles observed for that engine
  ├── MSE             : Mean Squared Error
  ├── MAE             : Mean Absolute Error
  └── RMSE            : Root Mean Squared Error

To run inference:
  python entrypoint/inference.py --input path/to/your_new_data.csv
