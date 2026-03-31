import sys
import argparse
from src.Pipelines.inference_pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(
        description="APU Predictive Maintenance — Inference Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to the input CSV file.\n"
            "Required columns: engine_id, cycle, op_setting_1-3, sensor_1-21\n"
            "Example: --input path/to/new_engines.csv"
        )
    )
    args = parser.parse_args()

    pipeline = InferencePipeline()
    results  = pipeline.run_inference(args.input)

    print(f"\nPredictions preview (first 10 rows):")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
