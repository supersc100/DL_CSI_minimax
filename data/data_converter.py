"""
Data converter for CSI data normalization and format handling.

This module handles CSI data normalization, compatible with Sionna 2.0
which generates data directly in NumPy/PyTorch format.
"""
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional


class CSIDataConverter:
    """
    Handles CSI data normalization and format transformation.

    Sionna 2.0 generates data directly in NumPy format, so this converter
    focuses on normalization and data format operations.
    """

    def __init__(self, normalization_method: str = "standard"):
        """
        Args:
            normalization_method: 'standard' (mean=0, std=1) or 'minmax' (range [0,1])
        """
        self.normalization_method = normalization_method
        self.mean = None
        self.std = None

    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSI data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            dl_csi = f['dl_csi'][:]
            ul_csi = f['ul_csi'][:]
        return dl_csi, ul_csi

    def compute_normalization(self, dl_csi: np.ndarray, ul_csi: Optional[np.ndarray] = None):
        """Compute normalization parameters from data."""
        if self.normalization_method == "standard":
            self.mean = dl_csi.mean(axis=(0, 1))
            self.std = dl_csi.std(axis=(0, 1))
            self.std = np.clip(self.std, 1e-8, None)
        elif self.normalization_method == "minmax":
            self.mean = dl_csi.min(axis=(0, 1))
            self.std = dl_csi.max(axis=(0, 1)) - self.mean

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using computed parameters."""
        if self.normalization_method == "standard":
            return (data - self.mean) / self.std
        elif self.normalization_method == "minmax":
            return (data - self.mean) / (self.std + 1e-8)
        return data

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.normalization_method == "standard":
            return data * self.std + self.mean
        elif self.normalization_method == "minmax":
            return data * (self.std + 1e-8) + self.mean
        return data

    def convert(
        self,
        input_path: str,
        output_path: str,
        compute_norm: bool = True,
    ):
        """
        Normalize CSI data and save to HDF5.

        Args:
            input_path: Path to input HDF5 file
            output_path: Path to output HDF5 file
            compute_norm: Whether to compute normalization from this data
        """
        dl_csi, ul_csi = self.load_dataset(input_path)

        if compute_norm:
            self.compute_normalization(dl_csi, ul_csi)

        dl_csi_norm = self.normalize(dl_csi)
        ul_csi_norm = self.normalize(ul_csi)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('dl_csi', data=dl_csi_norm)
            f.create_dataset('ul_csi', data=ul_csi_norm)
            f.attrs['mean'] = self.mean
            f.attrs['std'] = self.std
            f.attrs['normalization'] = self.normalization_method

        print(f"Converted {input_path} -> {output_path}")
        print(f"  Shape: {dl_csi_norm.shape}")
        print(f"  Mean: {self.mean}, Std: {self.std}")

    def to_torch_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.from_numpy(data).float()

    def save_normalization_params(self, filepath: str):
        """Save normalization parameters for later use."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalization parameters not computed")

        with h5py.File(filepath, 'w') as f:
            f.attrs['mean'] = self.mean
            f.attrs['std'] = self.std
            f.attrs['normalization'] = self.normalization_method

    def load_normalization_params(self, filepath: str):
        """Load normalization parameters from file."""
        with h5py.File(filepath, 'r') as f:
            self.mean = f.attrs['mean']
            self.std = f.attrs['std']
            self.normalization_method = f.attrs['normalization']


def convert_directory(
    input_dir: str,
    output_dir: str,
    normalization_method: str = "standard",
) -> Tuple[str, str]:
    """
    Normalize all CSI datasets in a directory.

    Returns:
        Tuple of (train_output_path, test_output_path)
    """
    from glob import glob

    converter = CSIDataConverter(normalization_method=normalization_method)

    train_files = glob(f"{input_dir}/*_train.h5")
    test_files = glob(f"{input_dir}/*_test.h5")

    if not train_files or not test_files:
        raise FileNotFoundError(f"No datasets found in {input_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_out = f"{output_dir}/train_normalized.h5"
    test_out = f"{output_dir}/test_normalized.h5"

    converter.convert(train_files[0], train_out, compute_norm=True)
    converter.convert(test_files[0], test_out, compute_norm=False)

    norm_path = f"{output_dir}/normalization_params.h5"
    converter.save_normalization_params(norm_path)

    return train_out, test_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize CSI data")
    parser.add_argument("--input", required=True, help="Input HDF5 file")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    parser.add_argument("--method", default="standard", choices=["standard", "minmax"])

    args = parser.parse_args()

    converter = CSIDataConverter(normalization_method=args.method)
    converter.convert(args.input, args.output)