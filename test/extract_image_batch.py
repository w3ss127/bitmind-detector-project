import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from io import BytesIO
import re
import asyncio
import sys
from pathlib import Path as SysPath
sys.path.append(str(SysPath(__file__).resolve().parent.parent))

from extract_image_tensors import ExtractImageProcessor

# Real: bm-real, MS-COCO-unique-256, open-image-v7-256, celeb-a-hq, dtd, caltech-101
# Synthetic: bm-aura-imagegen, GenImage_MidJourney, JourneyDB
# Semi-Synthetic: face-swap
DATASET_PATH = "bitmind/MS-COCO-unique-256"
START_FROM = 50000  # index to start downloading from
EXTRACT_COUNT = 20000  # total number of images to extract
BATCH_SIZE = 5000  # number of images per .pt file
# DIR: test/real_image_batches, test/synth_image_batches, semi_synth_image_batches
OUTPUT_DIR = Path("test/real_image_batches") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        parquet_path = await processor.download_parquet_from_hf(DATASET_PATH, parquet_file)
        df = processor.load_parquet_data(parquet_path)
        image_col = processor.find_image_column(df)
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
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            except Exception:
                continue
            # Resize to 256x256
            image = image.resize((256, 256), Image.LANCZOS)
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(device)
            # tensor = tensor.float() / 255.0  # Normalize to [0, 1]
            tensor_batch.append(tensor)
            images_extracted += 1
            if len(tensor_batch) == BATCH_SIZE:
                save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                # Stack on CUDA, then move to CPU before saving
                stacked = torch.stack(tensor_batch).cpu()
                torch.save(stacked, save_path)
                print(f"Saved {len(tensor_batch)} tensors to {save_path}")
                tensor_batch = []
                batch_idx += 1
    # Save any remaining tensors
    if tensor_batch:
        save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
        stacked = torch.stack(tensor_batch).cpu()
        torch.save(stacked, save_path)
        print(f"Saved {len(tensor_batch)} tensors to {save_path}")

if __name__ == "__main__":
    asyncio.run(main()) 