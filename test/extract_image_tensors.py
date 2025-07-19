"""
Extract example of downloading parquet files from Hugging Face
and extracting images to tensors - no bitmind dependencies.
"""

import asyncio
import base64
import json
import os
import random
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
import hashlib

# Optional imports - these may not be available in all environments
try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HF_AVAILABLE = True
    print("✓ huggingface_hub available")
except ImportError:
    HF_AVAILABLE = False
    print("✗ huggingface_hub not available. Install with: pip install huggingface_hub")

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
    print("✓ datasets available")
except ImportError:
    DATASETS_AVAILABLE = False
    print("✗ datasets not available. Install with: pip install datasets")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


class ExtractImageProcessor:
    """
    Extract processor for downloading parquet files from Hugging Face datasets
    and extracting images to tensors.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the image processor.

        Args:
            cache_dir: Directory to cache downloaded files. Defaults to ~/.cache/image_processor
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "image_processor"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Output directory for saving .pt files
        self.output_dir = Path(__file__).parent  # bitmind/cache/datasets

        # Default tensor configuration
        self.default_tensor_config = {
            "dtype": torch.float32,
            "normalize": False,  # Normalize to [0, 1]
            "channels_first": True,  # (C, H, W) format
            "resize": None,  # Optional resize (width, height)
        }

    def list_parquet_files(
        self, dataset_path: str, repo_type: str = "dataset"
    ) -> List[str]:
        """
        List all available parquet files in a Hugging Face dataset.

        Args:
            dataset_path: Hugging Face dataset path
            repo_type: Repository type ("dataset", "model", etc.)

        Returns:
            List of parquet filenames
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        try:
            print(f"Listing files in {dataset_path}...")
            files = list_repo_files(repo_id=dataset_path, repo_type=repo_type)
            parquet_files = [f for f in files if f.endswith(".parquet")]

            if not parquet_files:
                raise ValueError(f"No parquet files found in {dataset_path}")

            print(f"Found {len(parquet_files)} parquet files")
            return parquet_files

        except Exception as e:
            print(f"✗ Error listing files in {dataset_path}: {e}")
            raise

    async def download_parquet_from_hf(
        self,
        dataset_path: str,
        filename: Optional[str] = None,
        repo_type: str = "dataset",
    ) -> Path:
        """
        Download a parquet file from Hugging Face dataset.

        Args:
            dataset_path: Hugging Face dataset path (e.g., "bitmind/bm-eidon-image")
            filename: Specific parquet filename to download. If None, downloads a random parquet file
            repo_type: Repository type ("dataset", "model", etc.)

        Returns:
            Path to the downloaded parquet file
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        try:
            # Create dataset-specific cache directory
            dataset_name = dataset_path.split("/")[-1]
            dataset_cache_dir = self.cache_dir / dataset_name
            dataset_cache_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                # List available parquet files and select a random one
                parquet_files = self.list_parquet_files(dataset_path, repo_type)
                filename = random.choice(parquet_files)
                print(f"Selected random parquet file: {filename}")

            # Check if file already exists
            local_path = (
                dataset_cache_dir / str(filename)
                if filename
                else dataset_cache_dir / "data.parquet"
            )
            if local_path.exists():
                print(f"Parquet file already exists: {local_path}")
                return local_path

            # Download the file
            print(f"Downloading {filename} from {dataset_path}...")
            downloaded_path = hf_hub_download(
                repo_id=dataset_path,
                filename=filename,
                repo_type=repo_type,
                cache_dir=str(dataset_cache_dir),
                local_dir=str(dataset_cache_dir),
            )

            print(f"✓ Successfully downloaded: {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            print(f"✗ Error downloading parquet from {dataset_path}: {e}")
            print(traceback.format_exc())
            raise

    def load_parquet_data(self, parquet_path: Path) -> pd.DataFrame:
        """
        Load data from a parquet file.

        Args:
            parquet_path: Path to the parquet file

        Returns:
            DataFrame containing the parquet data
        """
        try:
            print(f"Loading parquet data from: {parquet_path}")

            # Read parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            print(f"✓ Loaded {len(df)} rows from parquet file")
            print(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"✗ Error loading parquet data: {e}")
            print(traceback.format_exc())
            raise

    def find_image_column(self, df: pd.DataFrame) -> str:
        """
        Find the image column in the dataframe.

        Args:
            df: DataFrame containing the data

        Returns:
            Name of the image column
        """
        # Look for common image column names
        image_keywords = [
            "image",
            "img",
            "photo",
            "picture",
            "data",
            "bytes",
            "content",
        ]

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in image_keywords):
                print(f"Found image column: {col}")
                return col

        # If no obvious image column found, return the first column
        print(f"⚠ No obvious image column found, using first column: {df.columns[0]}")
        return df.columns[0]

    def extract_image_from_row(self, row_data: Any) -> Optional[bytes]:
        """
        Extract image bytes from a row of data.

        Args:
            row_data: Image data from a dataframe row

        Returns:
            Image bytes or None if extraction fails
        """
        try:
            if isinstance(row_data, bytes):
                return row_data
            elif isinstance(row_data, str):
                # Try to decode base64
                try:
                    return base64.b64decode(row_data)
                except Exception:
                    # If not base64, try to encode as bytes
                    return row_data.encode("utf-8")
            elif isinstance(row_data, dict):
                # Look for image data in dictionary
                for key in ["bytes", "data", "image", "content"]:
                    if key in row_data:
                        return self.extract_image_from_row(row_data[key])
                # If no obvious key, try the first value
                first_value = next(iter(row_data.values()))
                return self.extract_image_from_row(first_value)
            else:
                # Try to convert to bytes
                return bytes(row_data)

        except Exception as e:
            print(f"⚠ Failed to extract image from row data: {e}")
            return None

    def image_to_tensor(
        self, image_bytes: bytes, config: Optional[Dict[str, Any]] = None, device: torch.device = device
    ) -> torch.Tensor:
        """
        Convert image bytes to tensor.

        Args:
            image_bytes: Raw image bytes
            config: Tensor configuration (dtype, normalize, channels_first, resize)

        Returns:
            Image tensor
        """
        config = config or self.default_tensor_config

        try:
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if specified
            if config.get("resize"):
                width, height = config["resize"]
                image = image.resize((width, height), Image.Resampling.LANCZOS)

            # Convert to numpy array
            image_array = np.array(image)

            # Normalize to [0, 1] if requested
            if config.get("normalize", True):
                image_array = image_array.astype(np.float32) / 255.0

            # Convert to tensor
            tensor = torch.from_numpy(image_array)

            # Change format if requested
            if config.get("channels_first", True):
                # Convert from (H, W, C) to (C, H, W)
                tensor = tensor.permute(2, 0, 1)

            # Set dtype
            tensor = tensor.to(config.get("dtype", torch.float32))

            # Move to device (CUDA or CPU)
            tensor = tensor.to(device)

            return tensor

        except Exception as e:
            print(f"✗ Error converting image to tensor: {e}")
            raise

    def extract_images_to_tensors(
        self,
        parquet_path: Path,
        num_images: int = 10,
        seed: Optional[int] = None,
        tensor_config: Optional[Dict[str, Any]] = None,
        save_metadata: bool = True,
        device: torch.device = device,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Extract images from parquet file and convert to tensors.

        Args:
            parquet_path: Path to the parquet file
            num_images: Number of images to extract
            seed: Random seed for sampling
            tensor_config: Configuration for tensor conversion
            save_metadata: Whether to save metadata for extracted images

        Returns:
            Tuple of (list of image tensors, list of metadata dictionaries)
        """
        try:
            # Set random seed
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            # Load parquet data
            df = self.load_parquet_data(parquet_path)

            # Find image column
            image_col = self.find_image_column(df)

            # Sample random rows
            sample_size = min(num_images, len(df))
            sample_df = df.sample(n=sample_size, random_state=seed)

            tensors = []
            metadata_list = []

            print(f"Extracting {sample_size} images...")

            for idx, (row_idx, row) in enumerate(sample_df.iterrows()):
                try:
                    # Extract image data
                    image_data = row[image_col]
                    image_bytes = self.extract_image_from_row(image_data)

                    if image_bytes is None:
                        print(f"⚠ Failed to extract image from row {row_idx}")
                        continue

                    # Convert to tensor
                    tensor = self.image_to_tensor(image_bytes, tensor_config, device=device)
                    tensors.append(tensor)

                    # Create metadata
                    metadata = {
                        "dataset": parquet_path.parent.name,
                        "source_parquet": str(parquet_path),
                        "original_index": int(row_idx),
                        "extraction_index": idx,
                        "tensor_shape": list(tensor.shape),
                        "tensor_dtype": str(tensor.dtype),
                        "extraction_time": datetime.now().isoformat(),
                    }

                    # Add other columns as metadata
                    for col in row.index:
                        if col != image_col:
                            try:
                                # Try to serialize the value
                                json.dumps({col: row[col]})
                                metadata[col] = row[col]
                            except (TypeError, OverflowError):
                                metadata[col] = str(row[col])

                    metadata_list.append(metadata)

                    if (idx + 1) % 10 == 0:
                        print(f"  Processed {idx + 1}/{sample_size} images")

                except Exception as e:
                    print(f"⚠ Failed to process row {row_idx}: {e}")
                    continue

            print(f"✓ Successfully extracted {len(tensors)} images to tensors")

            # Save metadata if requested
            if save_metadata and metadata_list:
                metadata_path = (
                    parquet_path.parent / f"{parquet_path.stem}_metadata.json"
                )
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(
                        metadata_list, f, indent=2, ensure_ascii=False, default=str
                    )
                print(f"✓ Saved metadata to: {metadata_path}")

            return tensors, metadata_list

        except Exception as e:
            print(f"✗ Error extracting images to tensors: {e}")
            print(traceback.format_exc())
            raise

    async def process_multiple_parquet_files(
        self,
        dataset_path: str,
        total_images: int = 10000,
        max_images_per_file: int = 10000,
        tensor_config: Optional[Dict[str, Any]] = None,
        save_tensors: bool = True,
        output_filename: Optional[str] = None,
        device: torch.device = device,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Process multiple parquet files to extract a total number of images.

        Args:
            dataset_path: Hugging Face dataset path
            total_images: Total number of images to extract
            max_images_per_file: Maximum images to extract from each parquet file
            tensor_config: Configuration for tensor conversion
            save_tensors: Whether to save tensors to .pt file
            output_filename: Output filename for tensors (without extension)

        Returns:
            Tuple of (list of image tensors, list of metadata dictionaries)
        """
        try:
            # List all available parquet files
            parquet_files = self.list_parquet_files(dataset_path)
            print(f"Found {len(parquet_files)} parquet files in {dataset_path}")

            all_tensors = []
            all_metadata = []
            processed_files = 0

            # Shuffle parquet files for random selection
            random.shuffle(parquet_files)

            seen_hashes = set()

            for filename in parquet_files:
                if len(all_tensors) >= total_images:
                    break
                try:
                    print(
                        f"\n📁 Processing file {processed_files + 1}/{len(parquet_files)}: {filename}"
                    )
                    # Calculate how many images to extract from this file
                    remaining_images = total_images - len(all_tensors)
                    images_to_extract = min(max_images_per_file, remaining_images)
                    # Download and process this parquet file
                    parquet_path = await self.download_parquet_from_hf(
                        dataset_path, filename
                    )
                    # Extract images from this file
                    tensors, metadata = self.extract_images_to_tensors(
                        parquet_path=parquet_path,
                        num_images=images_to_extract,
                        tensor_config=tensor_config,
                        save_metadata=False,  # We'll save combined metadata later
                        device=device,
                    )
                    for i, tensor in enumerate(tensors):
                        image_hash = hash_image_bytes(tensor.numpy().tobytes())
                        if image_hash in seen_hashes:
                            continue  # skip duplicate
                        seen_hashes.add(image_hash)
                        all_tensors.append(tensor)
                        all_metadata.append(metadata[i])
                    processed_files += 1
                    print(f"✓ Extracted {len(tensors)} images from {filename}")
                    print(
                        f"📊 Total images collected: {len(all_tensors)}/{total_images}"
                    )
                except Exception as e:
                    print(f"⚠ Failed to process {filename}: {e}")
                    continue
            print(
                f"\n✅ Successfully extracted {len(all_tensors)} images from {processed_files} parquet files"
            )
            if all_tensors:
                print(all_tensors[0].shape)
            else:
                print("No tensors were extracted.")
            # Save tensors to .pt file if requested
            if save_tensors and all_tensors:
                import os
                import json as _json

                if output_filename is None:
                    dataset_name = dataset_path.split("/")[-1]
                    output_filename = f"{dataset_name}_{len(all_tensors)}images"

                # Stack all tensors into a single tensor
                stacked_tensors = torch.stack(all_tensors, dim=0)
                # Move to CPU before saving
                stacked_tensors = stacked_tensors.cpu()
                output_path = self.output_dir / f"{output_filename}.pt"
                metadata_path = self.output_dir / f"{output_filename}_metadata.json"

                # --- APPEND TO EXISTING DATA IF FILES EXIST ---
                if os.path.exists(output_path):
                    print(f"Appending to existing tensor file: {output_path}")
                    existing_tensors = torch.load(output_path, weights_only=True)
                    stacked_tensors = torch.cat(
                        [existing_tensors, stacked_tensors], dim=0
                    )
                if os.path.exists(metadata_path):
                    print(f"Appending to existing metadata file: {metadata_path}")
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        existing_metadata = _json.load(f)
                    all_metadata = existing_metadata + all_metadata

                print(f"💾 Saving {stacked_tensors.shape[0]} tensors to {output_path}")
                torch.save(stacked_tensors, output_path)
                print(f"✓ Saved tensors to: {output_path}")
                print(f"📊 Tensor shape: {stacked_tensors.shape}")
                print(f"📊 Tensor dtype: {stacked_tensors.dtype}")

                # Save metadata
                with open(metadata_path, "w", encoding="utf-8") as f:
                    _json.dump(
                        all_metadata, f, indent=2, ensure_ascii=False, default=str
                    )
                print(f"✓ Saved metadata to: {metadata_path}")

            return all_tensors, all_metadata

        except Exception as e:
            print(f"✗ Error processing multiple parquet files: {e}")
            print(traceback.format_exc())
            raise

    def get_tensor_stats(self, tensors: List[torch.Tensor], device: torch.device = device) -> Dict[str, Any]:
        """
        Get statistics about a list of tensors.

        Args:
            tensors: List of image tensors

        Returns:
            Dictionary containing tensor statistics
        """
        if not tensors:
            return {}

        # Stack all tensors for analysis
        stacked = torch.stack(tensors, dim=0).to(device)

        stats = {
            "count": len(tensors),
            "shapes": [list(t.shape) for t in tensors],
            "dtypes": [str(t.dtype) for t in tensors],
            "min_value": float(torch.min(stacked)),
            "max_value": float(torch.max(stacked)),
            "mean_value": float(torch.mean(stacked)),
            "std_value": float(torch.std(stacked)),
        }

        return stats


def create_tensor_config(
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    channels_first: bool = True,
    resize: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Create a tensor configuration dictionary.

    Args:
        dtype: Tensor data type
        normalize: Whether to normalize to [0, 1]
        channels_first: Whether to use (C, H, W) format
        resize: Optional resize dimensions (width, height)

    Returns:
        Tensor configuration dictionary
    """
    return {
        "dtype": dtype,
        "normalize": normalize,
        "channels_first": channels_first,
        "resize": resize,
    }


async def main():
    """Main example function."""
    print("=" * 60)
    print("Extract Image Dataset Processor Example")
    print("Downloading 10,000 images from Hugging Face parquets")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\n❌ huggingface_hub is required to run this example.")
        print("Install it with: pip install huggingface_hub")
        return

    # Initialize processor
    processor = ExtractImageProcessor()

    # Train dataset path (change as needed)
    # dataset_path = "bitmind/bm-real"  # Real
    dataset_path = "bitmind/celeb-a-hq"  # Real

    # dataset_path = "bitmind/bm-aura-imagegen"  # Synthetic
    # dataset_path = "bitmind/face-swap"  # Semi-Synthetic
    # dataset_path = "bitmind/idoc-mugshots-images"  # Real
    # dataset_path = "bitmind/JourneyDB"  # Synthetic

    # Test dataset path (change as needed)

    try:
        print(f"\n📥 Processing dataset: {dataset_path}")
        print("🎯 Target: 10,000 images from Hugging Face parquets")

        # Create tensor configuration
        tensor_config = create_tensor_config(
            dtype=torch.float32,
            normalize=True,
            channels_first=True,
            resize=(256, 256),  # Resize all images to 256x256 for consistency
        )

        output_filename = "bm_train_images"
        # Process multiple parquet files to get 10,000 images
        tensors, metadata = await processor.process_multiple_parquet_files(
            dataset_path=dataset_path,
            total_images=20000,
            max_images_per_file=10000,  # Extract up to 10000 images per parquet file for efficiency
            tensor_config=tensor_config,
            save_tensors=True,
            output_filename=output_filename,
            device=device,
        )

        print(f"\n✅ Success! Extracted {len(tensors)} images")
        print(f"💾 Tensors saved to: {output_filename}.pt in current directory")
        print(
            f"📋 Metadata saved to: {output_filename}_metadata.json in current directory"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
