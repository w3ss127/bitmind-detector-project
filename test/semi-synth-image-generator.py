import asyncio
from pathlib import Path
from PIL import Image
import torch
import os
import re
from io import BytesIO
import numpy as np
import warnings
import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import ExtractImageProcessor from test/extract_image_tensors.py
from extract_image_tensors import ExtractImageProcessor

# Import GenerationPipeline and ModelTask from bitmind/generation
from bitmind.generation.generation_pipeline import GenerationPipeline
from bitmind.types import ModelTask

warnings.filterwarnings(
    "ignore", message="The config attributes.*were passed to UNet2DConditionModel.*"
)

# Settings
DATASET_PATH = "bitmind/bm-real"
START_FROM = 0  # index to start downloading from
EXTRACT_COUNT = 25000  # total number of images to extract
BATCH_SIZE = 5000  # number of images per .pt file
MODEL_NAMES = [
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Lykon/dreamshaper-8-inpainting",
]
OUTPUT_DIR = Path("test/semi_synth_image_batches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_tensor = torch.device(device)


async def main():
    processor = ExtractImageProcessor()
    parquet_files = processor.list_parquet_files(DATASET_PATH)
    tensor_batch = []

    # Check for existing .pt files and set batch_idx accordingly
    existing_pt_files = list(OUTPUT_DIR.glob("images_*.pt"))
    max_idx = -1
    pt_pattern = re.compile(r"images_(\d+)\.pt")
    for f in existing_pt_files:
        m = pt_pattern.match(f.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    batch_idx = max_idx + 1

    images_extracted = 0
    rows_seen = 0
    for parquet_file in parquet_files:
        if images_extracted >= EXTRACT_COUNT:
            break
        parquet_path = await processor.download_parquet_from_hf(
            DATASET_PATH, parquet_file
        )
        df = processor.load_parquet_data(parquet_path)
        image_col = str(processor.find_image_column(df))
        for idx, row in df.iterrows():
            if rows_seen < START_FROM:
                rows_seen += 1
                continue
            if images_extracted >= EXTRACT_COUNT:
                break
            rows_seen += 1
            image_bytes = processor.extract_image_from_row(row[image_col])
            if image_bytes is None:
                continue
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue

            image_array = np.array(image)
            image_samples = [{"image": image_array, "path": f"row_{rows_seen}"}]
            pipeline = GenerationPipeline(output_dir=OUTPUT_DIR, device=device)
            selected_model = random.choice(MODEL_NAMES)
            results = pipeline.generate(
                image_samples=image_samples,
                tasks=[ModelTask.IMAGE_TO_IMAGE.value],  # Pass as a list
                model_names=[selected_model],
            )
            if not results:
                continue

            # If results is a single image, wrap it in a list for uniformity
            if isinstance(results, (Image.Image, np.ndarray)):
                results = [results]

            for result in results:
                # Convert numpy array to PIL Image if needed
                if isinstance(result, np.ndarray):
                    image = Image.fromarray(result)
                else:
                    image = result
                image = image.resize((256, 256), Image.LANCZOS)
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(device_tensor)
                tensor_batch.append(tensor)
                images_extracted += 1

                # Save the current state of the tensor batch after every append
                save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                stacked = torch.stack(tensor_batch).cpu()
                torch.save(stacked, save_path)
                print(f"Saved {len(tensor_batch)} tensors to {save_path}")
                print(f"Tensor shape: {stacked.shape}")
                print(f"Tensor dtype: {stacked.dtype}")

                # Existing logic for full batches
                while len(tensor_batch) >= BATCH_SIZE:
                    save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                    stacked = torch.stack(tensor_batch[:BATCH_SIZE]).cpu()
                    torch.save(stacked, save_path)
                    print(f"Saved {BATCH_SIZE} tensors to {save_path}")
                    print(f"Tensor shape: {stacked.shape}")
                    print(f"Tensor dtype: {stacked.dtype}")
                    tensor_batch = tensor_batch[BATCH_SIZE:]
                    batch_idx += 1
                if images_extracted >= EXTRACT_COUNT:
                    break
    # Save any remaining tensors
    if tensor_batch:
        save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
        stacked = torch.stack(tensor_batch).cpu()
        torch.save(stacked, save_path)
        print(f"=========== All done! ===========")
        print(f"Saved {len(tensor_batch)} tensors to {save_path}")
        print(f"Tensor shape: {stacked.shape}")
        print(f"Tensor dtype: {stacked.dtype}")

if __name__ == "__main__":
    asyncio.run(main())
