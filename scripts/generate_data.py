"""
Data generation entry script for CSI data.

Generates realistic MIMO channel data using Sionna and saves it
in a format compatible with the PyTorch data pipeline.

Example usage:
    python scripts/generate_data.py --num_samples 10000 --output_dir ./data
    python scripts/generate_data.py --num_samples 5000 --config custom_config.yaml
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sionna_csi_generator import generate_csi_dataset, ChannelConfig


def main():
    parser = argparse.ArgumentParser(description="Generate CSI dataset using Sionna")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (0-1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data generation",
    )
    parser.add_argument(
        "--num_subcarriers",
        type=int,
        default=128,
        help="Number of OFDM subcarriers",
    )
    parser.add_argument(
        "--num_tx",
        type=int,
        default=64,
        help="Number of transmit antennas",
    )
    parser.add_argument(
        "--num_rx",
        type=int,
        default=16,
        help="Number of receive antennas",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Output sequence length for CSI data",
    )
    parser.add_argument(
        "--system_type",
        type=str,
        default="TDD",
        choices=["TDD", "FDD"],
        help="System type: TDD (Time Division Duplex) or FDD (Frequency Division Duplex)",
    )
    parser.add_argument(
        "--dl_frequency",
        type=float,
        default=3.5e9,
        help="Downlink carrier frequency in Hz (default: 3.5 GHz)",
    )
    parser.add_argument(
        "--ul_frequency",
        type=float,
        default=2.1e9,
        help="Uplink carrier frequency in Hz (only used for FDD, default: 2.1 GHz)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CSI Data Generation using Sionna")
    print("=" * 60)
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Subcarriers: {args.num_subcarriers}")
    print(f"TX/RX antennas: {args.num_tx}/{args.num_rx}")
    print(f"Output sequence length: {args.seq_len}")
    print(f"System type: {args.system_type}")

    # Configure channel
    config = ChannelConfig(
        system_type=args.system_type,
        num_subcarriers=args.num_subcarriers,
        num_tx_antennas=args.num_tx,
        num_rx_antennas=args.num_rx,
        batch_size=args.batch_size,
        output_seq_len=args.seq_len,
        carrier_frequency=args.dl_frequency,
        ul_carrier_frequency=args.ul_frequency,
    )

    print("\nChannel Configuration:")
    print(f"  System type: {config.system_type}")
    print(f"  Downlink carrier frequency: {config.carrier_frequency/1e9:.2f} GHz")
    if config.system_type.upper() == "FDD":
        print(f"  Uplink carrier frequency: {config.ul_carrier_frequency/1e9:.2f} GHz")
    print(f"  Number of paths: {config.num_paths}")
    print(f"  Delay spread: {config.delay_spread*1e9:.1f} ns")
    print(f"  SNR range: {config.snr_db_min}-{config.snr_db_max} dB")

    # Generate dataset
    print("\nStarting data generation...")
    train_file, test_file = generate_csi_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)
    print(f"Training data: {train_file}")
    print(f"Test data: {test_file}")

    # Verify files exist
    import os
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_size = os.path.getsize(train_file) / (1024 * 1024)
        test_size = os.path.getsize(test_file) / (1024 * 1024)
        print(f"File sizes: train={train_size:.1f}MB, test={test_size:.1f}MB")
    else:
        print("WARNING: Files not found!")

    print("\nTo use this data for training:")
    print(f"  python scripts/train.py --config config/csi_config.yaml --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
