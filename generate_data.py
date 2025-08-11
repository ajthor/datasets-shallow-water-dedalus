#!/usr/bin/env python3
"""
Generate shallow water PDE dataset and save to parquet files in chunks.
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shallow_water_dataset import ShallowWaterDataset


def generate_dataset_split(
    split_name="train", num_samples=1000, chunk_size=100, output_dir="data"
):
    """Generate a dataset split and save as chunked parquet files."""

    os.makedirs(output_dir, exist_ok=True)

    # Create shallow water dataset
    dataset = ShallowWaterDataset(
        Nphi=128,  # Smaller grid for faster generation
        Ntheta=64,
        stop_sim_time=600,  # Shorter simulations for dataset generation
        save_interval=1,  # Save every 1 hours
    )

    num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Generating {num_samples} {split_name} samples in {num_chunks} chunks...")

    dataset_iter = iter(dataset)
    chunk_data = None

    for i in range(num_samples):
        sample = next(dataset_iter)

        if chunk_data is None:
            # Initialize chunk data on first sample
            chunk_data = {key: [] for key in sample.keys()}

        # Add sample to current chunk
        for key, value in sample.items():
            chunk_data[key].append(value)

        # Save chunk when full or at end
        if (i + 1) % chunk_size == 0 or i == num_samples - 1:
            chunk_idx = i // chunk_size

            # Convert numpy arrays to lists for PyArrow compatibility
            table_data = {}
            for key, values in chunk_data.items():
                if isinstance(values[0], np.ndarray):
                    table_data[key] = [
                        arr.tolist() if hasattr(arr, "tolist") else arr
                        for arr in values
                    ]
                else:
                    table_data[key] = values

            # Convert to PyArrow table
            table = pa.table(table_data)

            # Save chunk
            filename = f"{split_name}-{chunk_idx:05d}-of-{num_chunks:05d}.parquet"
            filepath = os.path.join(output_dir, filename)
            pq.write_table(table, filepath)

            print(f"Saved chunk {chunk_idx + 1}/{num_chunks}: {filepath}")

            # Reset for next chunk
            chunk_data = {key: [] for key in sample.keys()}

    print(f"Generated {num_samples} {split_name} samples")
    return num_samples


if __name__ == "__main__":
    np.random.seed(42)

    # Generate train split
    generate_dataset_split("train", num_samples=1000, chunk_size=10)

    # Generate test split
    generate_dataset_split("test", num_samples=200, chunk_size=10)
