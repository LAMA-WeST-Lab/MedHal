import os
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Dataset:
    """
    Utility class for handling datasets, particularly for partitioning CSV files.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the Dataset with a path to a CSV file or dataset directory.

        Args:
            dataset_path: Path to the CSV file containing the dataset
        """
        self.dataset_path = dataset_path
        self.df = None
        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset from CSV file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        if self.dataset_path.endswith('.csv'):
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset from CSV: {self.dataset_path}")
        else:
            raise ValueError(f"Unsupported dataset format. Expected CSV file, got: {self.dataset_path}")

    def partition(self, output_folder_path: str, size_of_partition: int = 1000):
        """
        Partition the dataset into smaller chunks and save them to separate files.

        Args:
            output_folder_path: Directory where partitioned files will be saved
            size_of_partition: Number of rows per partition file
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Cannot partition.")

        os.makedirs(output_folder_path, exist_ok=True)

        num_partitions = (len(self.df) + size_of_partition - 1) // size_of_partition
        logger.info(f"Partitioning {len(self.df)} rows into {num_partitions} files of size {size_of_partition}")

        for i in range(num_partitions):
            start_idx = i * size_of_partition
            end_idx = min((i + 1) * size_of_partition, len(self.df))

            partition_df = self.df.iloc[start_idx:end_idx]
            partition_path = os.path.join(output_folder_path, f"partition_{i:05d}.csv")

            partition_df.to_csv(partition_path, index=False)
            logger.info(f"Saved partition {i} ({len(partition_df)} rows) to {partition_path}")
