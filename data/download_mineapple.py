#!/usr/bin/env python3
"""
MineApple Dataset Download Script

This script automates the download and setup of the MineApple orchard dataset
from the University of Minnesota Digital Conservancy. It handles download,
extraction, verification, and organization of the dataset files.

Usage:
    python data/download_mineapple.py [--output-dir DATA_DIR] [--skip-extraction] [--force]

Author: Applied AI Research Lab
Date: December 2025
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# Dataset configuration
DATASET_URL = "https://conservancy.umn.edu/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/download"
DATASET_NAME = "mineapple"
EXPECTED_SIZE_MB = 2500
CHUNK_SIZE = 8192  # 8KB chunks for streaming download


class MineAppleDownloader:
    """
    Handles downloading and organizing the MineApple dataset.
    
    This class manages the entire dataset acquisition pipeline, including
    downloading from the remote source, verifying file integrity, extracting
    compressed archives, and organizing files into the expected directory structure.
    """
    
    def __init__(self, output_dir: str = "data", force: bool = False, skip_extraction: bool = False):
        """
        Initialize the downloader with configuration parameters.
        
        Args:
            output_dir: Root directory where dataset will be stored
            force: If True, re-download even if files already exist
            skip_extraction: If True, skip extraction and only download
        """
        self.output_dir = Path(output_dir)
        self.force = force
        self.skip_extraction = skip_extraction
        
        # Define directory structure
        self.raw_dir = self.output_dir / "mineapple_raw"
        self.processed_dir = self.output_dir / "mineapple_processed"
        self.sample_dir = self.output_dir / "sample_images"
        self.download_path = self.output_dir / "mineapple_dataset.zip"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directory structure for the dataset."""
        directories = [
            self.output_dir,
            self.raw_dir,
            self.raw_dir / "images" / "train",
            self.raw_dir / "images" / "val",
            self.raw_dir / "images" / "test",
            self.raw_dir / "annotations",
            self.processed_dir,
            self.sample_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def _check_existing_dataset(self) -> bool:
        """
        Check if dataset already exists and is valid.
        
        Returns:
            True if valid dataset exists and force flag is False
        """
        if not self.force:
            train_images = list((self.raw_dir / "images" / "train").glob("*.jpg"))
            if len(train_images) > 50:
                print(f"Found existing dataset with {len(train_images)} training images")
                response = input("Dataset already exists. Re-download? (y/n): ")
                if response.lower() != 'y':
                    print("Skipping download. Use --force to override.")
                    return True
        return False
    
    def download_dataset(self) -> bool:
        """
        Download the MineApple dataset from the remote URL.
        
        Returns:
            True if download successful, False otherwise
        """
        if self._check_existing_dataset():
            return True
        
        print(f"\nDownloading MineApple dataset from:")
        print(f"{DATASET_URL}")
        print(f"This may take several minutes depending on your connection...\n")
        
        try:
            response = requests.get(DATASET_URL, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Initialize progress bar
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc="Downloading"
            )
            
            # Download with progress tracking
            with open(self.download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        size = f.write(chunk)
                        progress_bar.update(size)
            
            progress_bar.close()
            
            # Verify download
            file_size_mb = self.download_path.stat().st_size / (1024 * 1024)
            print(f"\nDownload complete: {file_size_mb:.2f} MB")
            
            if total_size > 0 and abs(file_size_mb - (total_size / (1024 * 1024))) > 10:
                print("Warning: Downloaded file size differs from expected size")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return False
        except KeyboardInterrupt:
            print("\nDownload interrupted by user")
            if self.download_path.exists():
                self.download_path.unlink()
            return False
    
    def _compute_checksum(self, filepath: Path, algorithm: str = 'sha256') -> str:
        """
        Compute checksum of a file for integrity verification.
        
        Args:
            filepath: Path to file
            algorithm: Hashing algorithm to use
            
        Returns:
            Hexadecimal checksum string
        """
        hash_func = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def extract_dataset(self) -> bool:
        """
        Extract the downloaded dataset archive.
        
        Returns:
            True if extraction successful, False otherwise
        """
        if self.skip_extraction:
            print("Skipping extraction (--skip-extraction flag set)")
            return True
        
        if not self.download_path.exists():
            print("Error: Downloaded file not found")
            return False
        
        print("\nExtracting dataset...")
        
        try:
            # Determine archive type and extract accordingly
            if self.download_path.suffix == '.zip':
                with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
            elif self.download_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(self.download_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.raw_dir)
            else:
                print(f"Unsupported archive format: {self.download_path.suffix}")
                return False
            
            print("Extraction complete")
            return True
            
        except (zipfile.BadZipFile, tarfile.TarError) as e:
            print(f"Error extracting archive: {e}")
            return False
        except KeyboardInterrupt:
            print("\nExtraction interrupted by user")
            return False
    
    def organize_files(self):
        """
        Organize extracted files into the expected directory structure.
        
        This method handles various possible archive structures and
        reorganizes files into our standardized layout.
        """
        print("\nOrganizing files...")
        
        # Find all image files recursively in raw directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(self.raw_dir.rglob(ext))
        
        print(f"Found {len(image_files)} image files")
        
        # Find annotation files
        annotation_files = list(self.raw_dir.rglob('*.json'))
        print(f"Found {len(annotation_files)} annotation files")
        
        # If files are already organized, skip
        train_imgs = list((self.raw_dir / "images" / "train").glob("*.jpg"))
        if len(train_imgs) > 0:
            print("Files already organized")
            return
        
        # Basic organization: split files into train/val/test
        # This is a placeholder - actual organization depends on archive structure
        print("Note: File organization may need manual adjustment based on archive structure")
        print("Please verify the directory structure in data/mineapple_raw/")
    
    def create_sample_images(self, num_samples: int = 5):
        """
        Copy a few representative images to sample_images directory.
        
        Args:
            num_samples: Number of sample images to create
        """
        print(f"\nCreating {num_samples} sample images...")
        
        # Find test images (prefer test set for samples)
        test_images = list((self.raw_dir / "images" / "test").glob("*.jpg"))
        
        if not test_images:
            # Fall back to all images if test set not found
            test_images = list(self.raw_dir.rglob("*.jpg"))
        
        if not test_images:
            print("Warning: No images found to create samples")
            return
        
        # Select evenly spaced samples
        step = max(1, len(test_images) // num_samples)
        sample_images = test_images[::step][:num_samples]
        
        for idx, img_path in enumerate(sample_images, 1):
            target_path = self.sample_dir / f"img_{idx:03d}.jpg"
            shutil.copy2(img_path, target_path)
            print(f"Created sample: {target_path.name}")
    
    def generate_metadata(self):
        """Generate metadata file with dataset statistics."""
        print("\nGenerating metadata...")
        
        metadata = {
            "dataset_name": DATASET_NAME,
            "source_url": DATASET_URL,
            "download_date": str(Path(self.download_path).stat().st_mtime) if self.download_path.exists() else "unknown",
            "splits": {},
        }
        
        # Count files in each split
        for split in ['train', 'val', 'test']:
            split_dir = self.raw_dir / "images" / split
            if split_dir.exists():
                images = list(split_dir.glob("*.jpg"))
                metadata["splits"][split] = {
                    "num_images": len(images),
                    "directory": str(split_dir)
                }
        
        # Save metadata
        metadata_path = self.output_dir / "mineapple_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
        print(f"\nDataset statistics:")
        for split, info in metadata["splits"].items():
            print(f"  {split}: {info['num_images']} images")
    
    def cleanup(self, keep_archive: bool = False):
        """
        Clean up temporary files after successful extraction.
        
        Args:
            keep_archive: If True, keep the downloaded archive file
        """
        if not keep_archive and self.download_path.exists():
            print(f"\nRemoving archive: {self.download_path.name}")
            self.download_path.unlink()
    
    def run(self, keep_archive: bool = False) -> bool:
        """
        Execute the complete download and setup pipeline.
        
        Args:
            keep_archive: If True, keep the downloaded archive
            
        Returns:
            True if all steps successful, False otherwise
        """
        print("=" * 60)
        print("MineApple Dataset Download and Setup")
        print("=" * 60)
        
        steps = [
            ("Downloading dataset", self.download_dataset),
            ("Extracting archive", self.extract_dataset),
            ("Organizing files", self.organize_files),
            ("Creating samples", lambda: self.create_sample_images()),
            ("Generating metadata", self.generate_metadata),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'=' * 60}")
            print(f"Step: {step_name}")
            print(f"{'=' * 60}")
            
            try:
                result = step_func()
                if result is False:
                    print(f"Failed: {step_name}")
                    return False
            except Exception as e:
                print(f"Error during {step_name}: {e}")
                return False
        
        self.cleanup(keep_archive=keep_archive)
        
        print("\n" + "=" * 60)
        print("Dataset setup complete!")
        print("=" * 60)
        print(f"\nDataset location: {self.raw_dir}")
        print(f"Sample images: {self.sample_dir}")
        print("\nNext steps:")
        print("1. Verify the dataset structure")
        print("2. Run preprocessing: python data/preprocess_mineapple.py")
        print("3. Start experiments: bash experiments/sam2_baseline_mineapple.sh")
        
        return True


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download and setup the MineApple dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for dataset (default: data)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extraction step (download only)'
    )
    
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='Keep the downloaded archive after extraction'
    )
    
    args = parser.parse_args()
    
    # Create downloader instance and run
    downloader = MineAppleDownloader(
        output_dir=args.output_dir,
        force=args.force,
        skip_extraction=args.skip_extraction
    )
    
    success = downloader.run(keep_archive=args.keep_archive)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
