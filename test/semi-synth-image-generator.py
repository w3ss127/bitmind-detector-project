import asyncio
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import os
import re
from io import BytesIO
import numpy as np
import warnings
import sys
from pathlib import Path
import random
import gc
import traceback
import base64
import json
import time
from typing import List, Optional, Tuple, Union, Dict, Any

# Optional imports - these may not be available in all environments
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
    print("✓ huggingface_hub available")
except ImportError:
    HF_AVAILABLE = False
    print("✗ huggingface_hub not available. Install with: pip install huggingface_hub")

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
    print("✓ pandas and pyarrow available")
except ImportError:
    PANDAS_AVAILABLE = False
    print("✗ pandas/pyarrow not available. Install with: pip install pandas pyarrow")

# Try to import transformers for VLM/LLM
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Blip2ForConditionalGeneration,
        Blip2Processor,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
    print("✓ transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("✗ transformers not available. Install with: pip install transformers")

# Try to import diffusers for image generation
try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("✓ diffusers available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("✗ diffusers not available. Install with: pip install diffusers")

warnings.filterwarnings(
    "ignore", message="The config attributes.*were passed to UNet2DConditionModel.*"
)

# Settings - Optimized for VRAM usage
DATASET_PATH = "bitmind/celeb-a-hq"
START_FROM = 0  # index to start downloading from
EXTRACT_COUNT = 10  # total number of images to extract
BATCH_SIZE = 5000  # number of images per .pt file (reduced for VRAM optimization)

# Smaller models for VRAM efficiency - using 4-bit quantized models
IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-6.7b-coco"  # Smaller BLIP2 model
TEXT_MODERATION_MODEL = "microsoft/DialoGPT-medium"  # Much smaller LLM
INPAINTING_MODEL = "stabilityai/stable-diffusion-2-inpainting"  # More reliable inpainting model

# Custom prompts that start with specific text
CUSTOM_PROMPTS = [
    "A semi-synthetic image of",
    "The enhanced setting is",
    "The improved background is", 
    "The synthetic image type/style is",
]

OUTPUT_DIR = Path("test/semi_synth_image_batches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_tensor = torch.device(device)

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def clear_gpu_memory():
    """Clear GPU memory to prevent VRAM overflow."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_random_mask(
    size: Tuple[int, int],
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.5,
    allow_multiple: bool = True,
    allowed_shapes: list = ["rectangle", "circle", "ellipse", "triangle"],
) -> Image.Image:
    """
    Create a random mask (or masks) for i2i/inpainting with more variety.
    Returns a single-channel ("L" mode) mask image.
    """
    w, h = size
    allowed_shapes = [s for s in allowed_shapes]
    max_retries = 5
    for attempt in range(max_retries):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        n_masks = np.random.randint(1, 5) if allow_multiple else 1
        for _ in range(n_masks):
            shape = np.random.choice(allowed_shapes)
            min_dim = min(w, h)
            min_pixel_size = 64
            min_mask_size = max(int(min_size_ratio * min_dim), min_pixel_size)
            max_mask_size = max(int(max_size_ratio * min_dim), min_pixel_size)
            if min_mask_size >= max_mask_size:
                width = min_mask_size
                height = min_mask_size
            else:
                width = np.random.randint(min_mask_size, max_mask_size)
                height = np.random.randint(min_mask_size, max_mask_size)
            width = min(width, w)
            height = min(height, h)
            if shape == "circle":
                r = min(width, height) // 2
                if r < 1:
                    r = 1
                cx = np.random.randint(r, w - r + 1)
                cy = np.random.randint(r, h - r + 1)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            elif shape == "rectangle":
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape == "ellipse":
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                x0, y0, x1, y1 = x, y, x + width, y + height
                draw.ellipse([x0, y0, x1, y1], fill=255)
            elif shape == "triangle":
                min_triangle_size = max(96, min_mask_size)
                max_triangle_size = max(128, max_mask_size)
                min_triangle_size = min(min_triangle_size, w, h)
                max_triangle_size = min(max_triangle_size, w, h)
                if min_triangle_size >= max_triangle_size:
                    width = max_triangle_size
                    height = max_triangle_size
                else:
                    width = np.random.randint(min_triangle_size, max_triangle_size)
                    height = np.random.randint(min_triangle_size, max_triangle_size)
                x = np.random.randint(0, w - width + 1)
                y = np.random.randint(0, h - height + 1)
                jitter = lambda v, maxv: max(
                    0,
                    min(v + np.random.randint(-width // 10, width // 10 + 1), maxv - 1),
                )
                pt1 = (jitter(x, w), jitter(y, h))
                pt2 = (jitter(x + width - 1, w), jitter(y, h))
                pt3 = (jitter(x, w), jitter(y + height - 1, h))
                pts = [pt1, pt2, pt3]
                draw.polygon(pts, fill=255)
        if np.array(mask).max() > 0:
            return mask
    return mask


class StandaloneImageProcessor:
    """Standalone image processor for downloading and extracting images from parquet files."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "image_processor"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_parquet_files(self, dataset_path: str, repo_type: str = "dataset") -> List[str]:
        """List all available parquet files in a Hugging Face dataset."""
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

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

    async def download_parquet_from_hf(self, dataset_path: str, filename: Optional[str] = None, repo_type: str = "dataset") -> Path:
        """Download a parquet file from Hugging Face dataset."""
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

        try:
            dataset_name = dataset_path.split("/")[-1]
            dataset_cache_dir = self.cache_dir / dataset_name
            dataset_cache_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                parquet_files = self.list_parquet_files(dataset_path, repo_type)
                filename = random.choice(parquet_files)
                print(f"Selected random parquet file: {filename}")

            local_path = dataset_cache_dir / str(filename) if filename else dataset_cache_dir / "data.parquet"
            if local_path.exists():
                print(f"Parquet file already exists: {local_path}")
                return local_path

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

    def load_parquet_data(self, parquet_path: Path):
        """Load data from a parquet file."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and pyarrow are required. Install with: pip install pandas pyarrow")

        try:
            print(f"Loading parquet data from: {parquet_path}")
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            print(f"✓ Loaded {len(df)} rows from parquet file")
            print(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"✗ Error loading parquet data: {e}")
            print(traceback.format_exc())
            raise

    def find_image_column(self, df):
        """Find the image column in the dataframe."""
        image_keywords = ["image", "img", "photo", "picture", "data", "bytes", "content"]

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in image_keywords):
                print(f"Found image column: {col}")
                return col

        print(f"⚠ No obvious image column found, using first column: {df.columns[0]}")
        return df.columns[0]

    def extract_image_from_row(self, row_data: Any) -> Optional[bytes]:
        """Extract image bytes from a row of data."""
        try:
            if isinstance(row_data, bytes):
                return row_data
            elif isinstance(row_data, str):
                try:
                    return base64.b64decode(row_data)
                except Exception:
                    return row_data.encode("utf-8")
            elif isinstance(row_data, dict):
                for key in ["bytes", "data", "image", "content"]:
                    if key in row_data:
                        return self.extract_image_from_row(row_data[key])
                first_value = next(iter(row_data.values()))
                return self.extract_image_from_row(first_value)
            else:
                return bytes(row_data)

        except Exception as e:
            print(f"⚠ Failed to extract image from row data: {e}")
            return None


class StandalonePromptGenerator:
    """Standalone prompt generator using VLM and LLM models."""
    
    def __init__(self, vlm_name: str, llm_name: str, device: str = "cuda"):
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.vlm_processor = None
        self.vlm = None
        self.llm = None
        self.device = device

    def load_vlm(self):
        """Load the vision-language model for image annotation."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
            
        print(f"Loading VLM model: {self.vlm_name}")
        # Use lower precision and CPU offloading for memory efficiency
        self.vlm_processor = Blip2Processor.from_pretrained(self.vlm_name, torch_dtype=torch.float16)
        self.vlm = Blip2ForConditionalGeneration.from_pretrained(
            self.vlm_name, 
            torch_dtype=torch.float16,
            device_map="auto",  # Automatic device mapping
            low_cpu_mem_usage=True
        )
        print(f"✓ Loaded VLM model: {self.vlm_name}")

    def load_llm(self):
        """Load the language model for text moderation."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
            
        print(f"Loading LLM model: {self.llm_name}")
        # Use smaller model and CPU offloading
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, 
            torch_dtype=torch.float16,
            device_map="auto",  # <--- add this line
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.llm = pipeline("text-generation", model=llm, tokenizer=tokenizer)
        print(f"✓ Loaded LLM model: {self.llm_name}")

    def load_models(self):
        """Load both VLM and LLM models."""
        if self.vlm is None:
            self.load_vlm()
        if self.llm is None:
            self.load_llm()

    def clear_gpu(self):
        """Clear GPU memory."""
        print("Clearing GPU memory...")
        if self.vlm:
            del self.vlm
            self.vlm = None
        if self.llm:
            del self.llm
            self.llm = None
        if self.vlm_processor:
            del self.vlm_processor
            self.vlm_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    def generate(self, image: Image.Image, custom_prompts: List[str], max_new_tokens: int = 20) -> str:
        """Generate a description using custom prompts."""
        if self.vlm is None or self.vlm_processor is None:
            self.load_vlm()

        description = ""
        for prompt in custom_prompts:
            description += prompt + " "
            inputs = self.vlm_processor(image, text=description, return_tensors="pt").to(self.device, torch.float16)

            with torch.no_grad():  # Disable gradient computation
                generated_ids = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
                answer = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if answer:
                answer = answer.rstrip(" ,;!?")
                if not answer.endswith("."):
                    answer += "."
                description += answer + " "
            else:
                description = description[: -len(prompt) - 1]

        if description.startswith(custom_prompts[0]):
            description = description[len(custom_prompts[0]) :]

        description = description.strip()
        if not description.endswith("."):
            description += "."

        return self.moderate(description)

    def moderate(self, description: str, max_new_tokens: int = 80) -> str:
        """Moderate the description using LLM."""
        if self.llm is None:
            self.load_llm()

        # Simplified moderation for smaller model
        try:
            # For smaller models, use a simpler approach
            if "DialoGPT" in self.llm_name:
                # Use a simple text cleaning approach for DialoGPT
                cleaned = description.replace("  ", " ").strip()
                if len(cleaned) > 200:  # Truncate if too long
                    cleaned = cleaned[:200] + "..."
                return cleaned
            else:
                # Original moderation for larger models
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "[INST]You always concisely rephrase given descriptions, "
                            "eliminate redundancy, and remove all specific references to "
                            "individuals by name. You do not respond with anything other "
                            "than the revised description.[/INST]"
                        ),
                    },
                    {"role": "user", "content": description},
                ]
                moderated_text = self.llm(
                    messages,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                    return_full_text=False,
                )
                return moderated_text[0]["generated_text"]
        except Exception as e:
            print(f"⚠ Error during moderation: {e}")
            return description


class StandaloneImageGenerator:
    """Standalone image generator using diffusion models."""
    
    def __init__(self, model_name: str, device: str = "cuda", mask_config: Dict[str, Any] = None):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.is_inpainting = "inpainting" in model_name.lower()
        
        # Default mask configuration
        self.mask_config = mask_config or {
            "min_size_ratio": 0.15,
            "max_size_ratio": 0.5,
            "allow_multiple": True,
            "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"]
        }

    def load_model(self):
        """Load the diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required. Install with: pip install diffusers")
            
        print(f"Loading diffusion model: {self.model_name}")
        
        try:
            # Try inpainting model first
            if self.is_inpainting:
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                # Fallback to regular text-to-image model
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
        except Exception as e:
            print(f"⚠ Failed to load inpainting model: {e}")
            print("Falling back to regular text-to-image model...")
            # Fallback to a simpler model
            fallback_model = "runwayml/stable-diffusion-v1-5"
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.is_inpainting = False
        
        # Enable memory optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
        print(f"✓ Loaded diffusion model: {self.model_name}")

    def generate_image(self, prompt: str, image: Image.Image, mask: Optional[Image.Image] = None) -> Image.Image:
        """Generate a semi-synthetic image using inpainting or text-to-image."""
        if self.pipeline is None:
            self.load_model()

        # Resize image to smaller size to save memory
        target_size = (512, 512)  # Smaller size for memory efficiency
        image = image.resize(target_size, Image.LANCZOS)

        # Generate the image with memory optimizations
        with torch.no_grad():  # Disable gradient computation
            try:
                if self.is_inpainting:
                    # Use the advanced random mask generation if none provided
                    if mask is None:
                        print("Generating random mask...")
                        mask = create_random_mask(
                            size=image.size,
                            **self.mask_config
                        )
                        print(f"Generated mask with shapes: {self.mask_config['allowed_shapes']}")
                    
                    # Ensure image is in RGB format
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # Use inpainting
                    result = self.pipeline(
                        prompt=prompt,
                        image=image,
                        mask_image=mask,
                        num_inference_steps=15,  # Reduced for speed and memory
                        guidance_scale=7.0,  # Slightly reduced
                    ).images[0]
                else:
                    # Use text-to-image generation
                    result = self.pipeline(
                        prompt=prompt,
                        num_inference_steps=15,  # Reduced for speed and memory
                        guidance_scale=7.0,  # Slightly reduced
                    ).images[0]
                    
            except Exception as e:
                print(f"⚠ Inpainting failed: {e}")
                print("Falling back to text-to-image generation...")
                
                # Fallback to text-to-image generation
                try:
                    result = self.pipeline(
                        prompt=prompt,
                        num_inference_steps=15,
                        guidance_scale=7.0,
                    ).images[0]
                except Exception as e2:
                    print(f"⚠ Text-to-image also failed: {e2}")
                    # Return original image as last resort
                    return image

        return result

    def clear_gpu(self):
        """Clear GPU memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()


async def main():
    """Main function to run the standalone semi-synthetic image generator."""
    print("=" * 60)
    print("Standalone Semi-Synthetic Image Generator with Random Masks")
    print("Loading parquets from Hugging Face and generating semi-synthetic images")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\n❌ huggingface_hub is required to run this example.")
        print("Install it with: pip install huggingface_hub")
        return

    if not PANDAS_AVAILABLE:
        print("\n❌ pandas and pyarrow are required to run this example.")
        print("Install them with: pip install pandas pyarrow")
        return

    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ transformers is required to run this example.")
        print("Install it with: pip install transformers")
        return

    if not DIFFUSERS_AVAILABLE:
        print("\n❌ diffusers is required to run this example.")
        print("Install it with: pip install diffusers")
        return

    # Clear GPU memory at start
    clear_gpu_memory()
    
    # Initialize components with custom mask configuration
    processor = StandaloneImageProcessor()
    prompt_generator = StandalonePromptGenerator(
        vlm_name=IMAGE_ANNOTATION_MODEL,
        llm_name=TEXT_MODERATION_MODEL,
        device=device
    )
    
    # Configure mask generation settings
    mask_config = {
        "min_size_ratio": 0.2,  # Slightly larger minimum masks
        "max_size_ratio": 0.6,  # Larger maximum masks
        "allow_multiple": True,  # Allow multiple mask regions
        "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"]  # All shapes
    }
    
    image_generator = StandaloneImageGenerator(
        model_name=INPAINTING_MODEL,
        device=device,
        mask_config=mask_config
    )

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

            # Complete standalone workflow with aggressive memory management
            try:
                print(f"Processing image {images_extracted + 1}/{EXTRACT_COUNT}")
                
                # Step 1: Load models and generate prompt
                print("Loading models and generating prompt...")
                prompt_generator.load_models()
                
                prompt = prompt_generator.generate(
                    image=image,
                    custom_prompts=CUSTOM_PROMPTS,
                    max_new_tokens=20
                )
                print(f"Generated prompt: {prompt}")
                
                # Clear GPU after prompt generation
                prompt_generator.clear_gpu()
                clear_gpu_memory()
                
                # Step 2: Generate semi-synthetic image with random mask
                print("Generating semi-synthetic image with random mask...")
                result_image = image_generator.generate_image(
                    prompt=prompt,
                    image=image
                )
                
                # Clear GPU after image generation
                image_generator.clear_gpu()
                clear_gpu_memory()
                
                # Step 3: Convert to tensor and add to batch
                result_image = result_image.resize((224, 224), Image.LANCZOS)
                tensor = torch.from_numpy(np.array(result_image)).permute(2, 0, 1).contiguous().to(device_tensor)
                tensor_batch.append(tensor)
                images_extracted += 1

                print(f"✓ Generated semi-synthetic image {images_extracted}/{EXTRACT_COUNT}")
                print(f"📦 Batch size: {len(tensor_batch)}/{BATCH_SIZE}")

                # Save batch only when it reaches BATCH_SIZE
                if len(tensor_batch) == BATCH_SIZE:
                    save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                    stacked = torch.stack(tensor_batch).cpu()
                    torch.save(stacked, save_path)
                    print(f"💾 Saved {len(tensor_batch)} tensors to {save_path}")
                    print(f"📊 Tensor shape: {stacked.shape}")
                    print(f"📊 Tensor dtype: {stacked.dtype}")
                    tensor_batch = []
                    batch_idx += 1
                    clear_gpu_memory()
                
            except Exception as e:
                print(f"⚠ Failed to generate semi-synthetic image for row {rows_seen}: {e}")
                print(traceback.format_exc())
                # Fall back to original image
                image = image.resize((224, 224), Image.LANCZOS)
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(device_tensor)
                tensor_batch.append(tensor)
                images_extracted += 1
                print(f"✓ Used original image {images_extracted}/{EXTRACT_COUNT}")
                
                # Save batch only when it reaches BATCH_SIZE
                if len(tensor_batch) == BATCH_SIZE:
                    save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                    stacked = torch.stack(tensor_batch).cpu()
                    torch.save(stacked, save_path)
                    print(f"💾 Saved {len(tensor_batch)} tensors to {save_path}")
                    print(f"📊 Tensor shape: {stacked.shape}")
                    print(f"📊 Tensor dtype: {stacked.dtype}")
                    tensor_batch = []
                    batch_idx += 1
                    clear_gpu_memory()
                
            if images_extracted >= EXTRACT_COUNT:
                break

    # Final cleanup
    prompt_generator.clear_gpu()
    image_generator.clear_gpu()
    clear_gpu_memory()
    
    # Save any remaining tensors in the final batch (if any)
    if tensor_batch:
        save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
        stacked = torch.stack(tensor_batch).cpu()
        torch.save(stacked, save_path)
        print(f"💾 Saved final batch of {len(tensor_batch)} tensors to {save_path}")
        print(f"📊 Final tensor shape: {stacked.shape}")
        print(f"📊 Final tensor dtype: {stacked.dtype}")
    
    print(f"=========== All done! ===========")
    print(f"📊 Total images processed: {images_extracted}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print(f"🎭 Mask configuration used:")
    print(f"   - Size ratio: {mask_config['min_size_ratio']:.1f} - {mask_config['max_size_ratio']:.1f}")
    print(f"   - Multiple masks: {mask_config['allow_multiple']}")
    print(f"   - Shapes: {', '.join(mask_config['allowed_shapes'])}")

if __name__ == "__main__":
    asyncio.run(main())