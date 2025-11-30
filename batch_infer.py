import os
import sys
import time
from datetime import datetime

import joblib
import pandas as pd


def batch_predict(input_file, output_file, model_path):
    print(f"[{datetime.now()}] Starting batch inference...")

    # 1. Validation
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Tip: Run 'python train_model.py' to generate the model artifact.")
        sys.exit(1)

    try:
        # 2. Load Resources
        start_time = time.time()
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)

        required_columns = ["x1", "x2"]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input CSV must contain columns: {required_columns}")
            sys.exit(1)

        # 3. Inference
        print(f"Processing {len(df)} rows...")
        features = df[required_columns].values

        # Using predict_proba for a continuous score (probability of class 1)
        scores = model.predict_proba(features)[:, 1]

        # 4. Save Results
        df["prediction"] = scores.round(4)
        df["model_version"] = "v1.0"

        print(f"Writing results to {output_file}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)

        # 5. Statistics
        duration = time.time() - start_time
        print("\n--- Batch Inference Statistics ---")
        print(f"Rows processed: {len(df)}")
        print(f"Time taken:     {duration:.4f} seconds")
        print(f"Avg time/row:   {(duration / len(df) * 1000):.2f} ms")
        print(f"Output saved:   {output_file}")
        print("----------------------------------")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_infer.py <input_csv_path> <output_csv_path>")
        print("Example: python batch_infer.py data/input.csv data/predictions.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = os.path.join("models", "baseline.joblib")

    batch_predict(input_path, output_path, model_path)
