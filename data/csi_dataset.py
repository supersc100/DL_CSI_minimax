"""
CSI Dataset for PyTorch DataLoader.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Optional, Tuple, Callable


class CSIDataset(Dataset):
    """
    PyTorch Dataset for CSI data.

    Loads downlink/uplink CSI pairs from HDF5 files and provides
    them in the format expected by the model.
    """

    def __init__(
        self,
        h5_file: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        """
        Args:
            h5_file: Path to HDF5 file containing 'dl_csi' and 'ul_csi' datasets
            transform: Optional transform to apply to each sample
            normalize: Whether to normalize the CSI data
        """
        self.h5_file = h5_file
        self.transform = transform
        self.normalize = normalize

        with h5py.File(h5_file, 'r') as f:
            self.num_samples = f['dl_csi'].shape[0]
            self.seq_len = f['dl_csi'].shape[1]
            self.num_features = f['dl_csi'].shape[2]

            # Compute normalization statistics
            if normalize:
                dl_csi = f['dl_csi'][:1000]  # Use subset for efficiency
                self.mean = torch.from_numpy(dl_csi.mean(axis=(0, 1))).float()
                self.std = torch.from_numpy(dl_csi.std(axis=(0, 1))).float()
                # Avoid division by zero
                self.std = torch.clamp(self.std, min=1e-8)
            else:
                self.mean = None
                self.std = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single CSI sample pair."""
        with h5py.File(self.h5_file, 'r') as f:
            dl_csi = f['dl_csi'][idx]
            ul_csi = f['ul_csi'][idx]

        # Convert to torch tensors
        dl_csi = torch.from_numpy(dl_csi).float()
        ul_csi = torch.from_numpy(ul_csi).float()

        # Normalize
        if self.normalize and self.mean is not None:
            dl_csi = (dl_csi - self.mean) / self.std
            ul_csi = (ul_csi - self.mean) / self.std

        # Apply transform if provided
        if self.transform:
            dl_csi = self.transform(dl_csi)
            ul_csi = self.transform(ul_csi)

        return dl_csi, ul_csi

    def get_normalization_params(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return normalization parameters for later use."""
        return self.mean, self.std


class CSIDataLoader:
    """
    Convenience wrapper for creating PyTorch DataLoaders for CSI data.
    """

    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
    ):
        """
        Args:
            train_file: Path to training HDF5 file
            test_file: Path to test HDF5 file
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for data loading
            shuffle_train: Whether to shuffle training data
        """
        self.train_dataset = CSIDataset(train_file)
        self.test_dataset = CSIDataset(
            test_file,
            normalize=True,
        )

        # Use training normalization for test set
        self.test_dataset.mean = self.train_dataset.mean
        self.test_dataset.std = self.train_dataset.std

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    @property
    def normalization_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalization parameters from training set."""
        return self.train_dataset.mean, self.train_dataset.std


def create_csi_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders from default data directory.

    Args:
        data_dir: Directory containing CSI HDF5 files
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, test_loader)
    """
    import glob

    train_files = glob.glob(f"{data_dir}/*_train.h5")
    test_files = glob.glob(f"{data_dir}/*_test.h5")

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"No training/test files found in {data_dir}. "
            "Run data generation first: python scripts/generate_data.py"
        )

    # Use first matching files
    train_file = train_files[0]
    test_file = test_files[0]

    loader = CSIDataLoader(
        train_file=train_file,
        test_file=test_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return loader.train_loader, loader.test_loader
