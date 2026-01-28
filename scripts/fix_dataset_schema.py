from pathlib import Path

import numpy as np
import pandas as pd


def fix_parquet_file(path):
    try:
        # Read with pyarrow backend to properly load the complex types
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    modified = False
    for col in df.columns:
        # Check if it's an image statistic column
        # Pattern: stats/observation.images.{camera_name}/{stat_name}
        is_image_stat = "observation.images" in col and col.startswith("stats/")

        if isinstance(df[col].dtype, pd.ArrowDtype) or is_image_stat:
            try:
                # Convert to python list first
                if isinstance(df[col].dtype, pd.ArrowDtype):
                    # Handle Arrow-backed columns
                    arrow_array = df[col].array
                    data_list = arrow_array._pa_array.to_pylist()
                else:
                    # Handle standard object columns that might contain numpy arrays or lists
                    data_list = df[col].tolist()

                # If it is an image stat, we need to ensure it matches the (3, 1, 1) structure
                # expected by the LeRobot schema (list<list<list<double>>>).
                if is_image_stat:
                    new_data = []
                    for item in data_list:
                        # Convert numpy array to list if needed
                        if isinstance(item, np.ndarray):
                            item = item.tolist()

                        # If it's a flat list of 3 elements (RGB), reshape to [[[r]], [[g]], [[b]]]
                        if isinstance(item, list) and len(item) == 3 and not isinstance(item[0], list):
                            # Reshape [r, g, b] -> [[[r]], [[g]], [[b]]]
                            new_item = [[[float(x)]] for x in item]
                            new_data.append(new_item)
                        else:
                            new_data.append(item)

                    df[col] = new_data
                    modified = True

                # Non-image stat Arrow columns -> just convert to list
                elif isinstance(df[col].dtype, pd.ArrowDtype):
                    df[col] = data_list
                    modified = True

            except Exception as e:
                print(f"  Error converting column {col}: {e}")

    if modified:
        print(f"Fixing schema for {path}")
        # Saving as parquet now will use the standard Pandas->Parquet inference
        # compatible with LeRobot's standard read_parquet calls
        df.to_parquet(path)


def main():
    # Update this path to your dataset
    dataset_path = Path("/home/tamer/.cache/huggingface/lerobot/ravtscheev/three-piece-assembly-UR5e")

    if not dataset_path.exists():
        print("Dataset path not found.")
        return

    print(f"Processing dataset at: {dataset_path}")
    files = list(dataset_path.glob("meta/episodes/**/*.parquet"))
    print(f"Found {len(files)} episode metadata files.")

    for f in files:
        fix_parquet_file(f)

    print("Done.")


if __name__ == "__main__":
    main()
