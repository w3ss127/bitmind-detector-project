import asyncio
from pathlib import Path
from PIL import Image
import torch
import os
from io import BytesIO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import ExtractImageProcessor from test/extract_image_tensors.py
from extract_image_tensors import ExtractImageProcessor

# Import GenerationPipeline and ModelTask from bitmind/generation
from bitmind.generation.generation_pipeline import GenerationPipeline
from bitmind.types import ModelTask

# Settings
DATASET_PATH = "bitmind/bm-real"
NUM_IMAGES = 10
MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
OUTPUT_DIR = Path("test/semi_synth_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    # 1. Load real images from HuggingFace (as PIL Images)
    processor = ExtractImageProcessor()
    parquet_files = processor.list_parquet_files(DATASET_PATH)
    parquet_path = await processor.download_parquet_from_hf(DATASET_PATH, parquet_files[0])
    df = processor.load_parquet_data(parquet_path)
    image_col = processor.find_image_column(df)
    sample_df = df.sample(n=NUM_IMAGES, random_state=42)

    image_samples = []
    for idx, row in sample_df.iterrows():
        image_bytes = processor.extract_image_from_row(row[image_col])
        if image_bytes is None:
            continue
        try:
            image = Image.open(torch.io.BytesIO(image_bytes)).convert('RGB')
        except Exception:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_samples.append({
            "image": image,
            "path": f"row_{idx}"
        })

    print(f"Loaded {len(image_samples)} real images from HuggingFace.")

    # 2. Run Bitmind GenerationPipeline for image-to-image generation
    pipeline = GenerationPipeline(output_dir=OUTPUT_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
    results = pipeline.generate(
        image_samples=image_samples,
        tasks=ModelTask.IMAGE_TO_IMAGE.value,
        model_names=MODEL_NAME
    )

    print(f"Generated semi-synthetic images. Output paths:")
    for path in results:
        print(path)

if __name__ == "__main__":
    asyncio.run(main()) 