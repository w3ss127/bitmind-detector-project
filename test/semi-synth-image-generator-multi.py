import asyncio
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

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

# Settings - Optimized for multi-GPU usage
DATASET_PATH = "bitmind/open-images-v7-subset"
START_FROM = 0
EXTRACT_COUNT = 10000  # Total images to process
BATCH_SIZE = 5000   # Images per .pt file
NUM_GPUS = 8        # Number of GPUs available

# Model settings
IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL = "microsoft/DialoGPT-medium"
INPAINTING_MODEL = "stabilityai/stable-diffusion-2-inpainting"

# Custom prompts
CUSTOM_PROMPTS = [
    "A semi-synthetic image of",
    "The enhanced setting is",
    "The improved background is", 
    "The synthetic image type/style is",
]

OUTPUT_DIR = Path("test/semi_synth_image_batches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for multi-GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NUM_GPUS))

@dataclass
class ImageData:
    """Container for image data and metadata."""
    image_bytes: bytes
    image_id: int
    prompt: Optional[str] = None

@dataclass
class PromptTask:
    """Task for prompt generation."""
    image_bytes: bytes
    image_id: int

@dataclass
class ImageGenTask:
    """Task for image generation."""
    image_bytes: bytes
    image_id: int
    prompt: str

def clear_gpu_memory(device_id: Optional[int] = None):
    """Clear GPU memory for specific device or all devices."""
    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    else:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    gc.collect()

def create_random_mask(
    size: Tuple[int, int],
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.5,
    allow_multiple: bool = True,
    allowed_shapes: list = ["rectangle", "circle", "ellipse", "triangle"],
) -> Image.Image:
    """Create a random mask for inpainting."""
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

def prompt_generation_worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                           progress_queue: mp.Queue):
    """Worker process for prompt generation - loads models once and keeps them."""
    try:
        # Set up GPU for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        print(f"[Prompt GPU {gpu_id}] Loading models...")
        
        # Load VLM models once
        vlm_processor = Blip2Processor.from_pretrained(IMAGE_ANNOTATION_MODEL, torch_dtype=torch.float16)
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            IMAGE_ANNOTATION_MODEL, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Load LLM models once  
        llm_model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODERATION_MODEL, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        llm_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODERATION_MODEL)
        llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=gpu_id)
        
        print(f"[Prompt GPU {gpu_id}] ✓ Models loaded, ready for processing")
        
        processed_count = 0
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=5)
                if task is None:  # Shutdown signal
                    break
                
                image_bytes = task.image_bytes
                image_id = task.image_id
                
                # Convert bytes to image
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
                # Generate prompt using loaded models
                description = ""
                for prompt in CUSTOM_PROMPTS:
                    description += prompt + " "
                    inputs = vlm_processor(image, text=description, return_tensors="pt").to(device, torch.float16)

                    with torch.no_grad():
                        generated_ids = vlm.generate(**inputs, max_new_tokens=20)
                        answer = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    if answer:
                        answer = answer.rstrip(" ,;!?")
                        if not answer.endswith("."):
                            answer += "."
                        description += answer + " "
                    else:
                        description = description[: -len(prompt) - 1]

                if description.startswith(CUSTOM_PROMPTS[0]):
                    description = description[len(CUSTOM_PROMPTS[0]) :]

                description = description.strip()
                if not description.endswith("."):
                    description += "."

                # Moderate with LLM
                try:
                    if "DialoGPT" in TEXT_MODERATION_MODEL:
                        cleaned = description.replace("  ", " ").strip()
                        if len(cleaned) > 200:
                            cleaned = cleaned[:200] + "..."
                        final_prompt = cleaned
                    else:
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
                        moderated_text = llm_pipeline(
                            messages,
                            max_new_tokens=80,
                            pad_token_id=llm_pipeline.tokenizer.eos_token_id,
                            return_full_text=False,
                        )
                        final_prompt = moderated_text[0]["generated_text"]
                except Exception as e:
                    print(f"[Prompt GPU {gpu_id}] ⚠ Moderation failed: {e}")
                    final_prompt = description
                
                # Put result in queue
                result_queue.put((image_id, final_prompt))
                processed_count += 1
                
                if processed_count % 5 == 0:
                    progress_queue.put(f"[Prompt GPU {gpu_id}] Generated {processed_count} prompts")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Prompt GPU {gpu_id}] Error processing task {image_id}: {e}")
                result_queue.put((image_id, "fallback prompt"))
        
        print(f"[Prompt GPU {gpu_id}] Processed {processed_count} prompts, shutting down")
        
        # Clean up models
        del vlm, vlm_processor, llm_model, llm_tokenizer, llm_pipeline
        clear_gpu_memory(gpu_id)
        
    except Exception as e:
        print(f"[Prompt GPU {gpu_id}] Worker error: {e}")
        print(traceback.format_exc())

def image_generation_worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                          progress_queue: mp.Queue, mask_config: Dict[str, Any]):
    """Worker process for image generation - loads models once and keeps them."""
    try:
        # Set up GPU for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        print(f"[Image GPU {gpu_id}] Loading diffusion model...")
        
        # Load diffusion model once
        try:
            pipeline_model = StableDiffusionInpaintPipeline.from_pretrained(
                INPAINTING_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            is_inpainting = True
        except Exception as e:
            print(f"[Image GPU {gpu_id}] ⚠ Failed to load inpainting model: {e}")
            fallback_model = "runwayml/stable-diffusion-v1-5"
            pipeline_model = StableDiffusionPipeline.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            is_inpainting = False
        
        pipeline_model = pipeline_model.to(device)
        pipeline_model.enable_attention_slicing()
        pipeline_model.enable_vae_slicing()
        
        print(f"[Image GPU {gpu_id}] ✓ Diffusion model loaded, ready for processing")
        
        processed_count = 0
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=5)
                if task is None:  # Shutdown signal
                    break
                
                image_bytes = task.image_bytes
                image_id = task.image_id
                prompt = task.prompt
                
                # Convert bytes to image
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                target_size = (512, 512)
                image = image.resize(target_size, Image.LANCZOS)

                # Generate image using loaded model
                with torch.no_grad():
                    try:
                        if is_inpainting:
                            mask = create_random_mask(size=image.size, **mask_config)
                            
                            result = pipeline_model(
                                prompt=prompt,
                                image=image,
                                mask_image=mask,
                                num_inference_steps=15,
                                guidance_scale=7.0,
                            ).images[0]
                        else:
                            result = pipeline_model(
                                prompt=prompt,
                                num_inference_steps=15,
                                guidance_scale=7.0,
                            ).images[0]
                            
                    except Exception as e:
                        print(f"[Image GPU {gpu_id}] ⚠ Generation failed: {e}")
                        try:
                            result = pipeline_model(
                                prompt=prompt,
                                num_inference_steps=15,
                                guidance_scale=7.0,
                            ).images[0]
                        except Exception as e2:
                            print(f"[Image GPU {gpu_id}] ⚠ Fallback failed: {e2}")
                            result = image

                # Convert to tensor
                result = result.resize((224, 224), Image.LANCZOS)
                tensor = torch.from_numpy(np.array(result)).permute(2, 0, 1).contiguous()
                
                # Put result in queue
                result_queue.put((image_id, tensor))
                processed_count += 1
                
                if processed_count % 5 == 0:
                    progress_queue.put(f"[Image GPU {gpu_id}] Generated {processed_count} images")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Image GPU {gpu_id}] Error processing task {image_id}: {e}")
                # Use original image as fallback
                try:
                    image = Image.open(BytesIO(task.image_bytes)).convert("RGB")
                    image = image.resize((224, 224), Image.LANCZOS)
                    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
                    result_queue.put((image_id, tensor))
                except:
                    result_queue.put((image_id, None))
        
        print(f"[Image GPU {gpu_id}] Processed {processed_count} images, shutting down")
        
        # Clean up model
        del pipeline_model
        clear_gpu_memory(gpu_id)
        
    except Exception as e:
        print(f"[Image GPU {gpu_id}] Worker error: {e}")
        print(traceback.format_exc())

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

async def main():
    """Main function with two-phase processing: prompt generation, then image generation."""
    global NUM_GPUS
    
    print("=" * 80)
    print(f"Optimized Multi-GPU Semi-Synthetic Image Generator ({NUM_GPUS} GPUs)")
    print("Phase 1: Generate prompts | Phase 2: Generate images")
    print("=" * 80)

    # Check requirements
    if not all([HF_AVAILABLE, PANDAS_AVAILABLE, TRANSFORMERS_AVAILABLE, DIFFUSERS_AVAILABLE]):
        print("\n❌ Missing required dependencies. Please install:")
        if not HF_AVAILABLE:
            print("  pip install huggingface_hub")
        if not PANDAS_AVAILABLE:
            print("  pip install pandas pyarrow")
        if not TRANSFORMERS_AVAILABLE:
            print("  pip install transformers")
        if not DIFFUSERS_AVAILABLE:
            print("  pip install diffusers")
        return

    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < NUM_GPUS:
        print(f"⚠ Only {available_gpus} GPUs available, adjusting NUM_GPUS")
        NUM_GPUS = available_gpus

    print(f"🚀 Using {NUM_GPUS} GPUs for processing")
    
    # Clear all GPU memory at start
    for i in range(NUM_GPUS):
        clear_gpu_memory(i)
    
    # Initialize image processor
    processor = StandaloneImageProcessor()
    
    # Configure mask generation settings
    mask_config = {
        "min_size_ratio": 0.2,
        "max_size_ratio": 0.6,
        "allow_multiple": True,
        "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"]
    }

    # Load and extract image data first
    print("\n📥 Loading image data from parquet files...")
    image_data_list = []
    parquet_files = processor.list_parquet_files(DATASET_PATH)
    
    images_loaded = 0
    rows_seen = 0
    
    for parquet_file in parquet_files:
        if images_loaded >= EXTRACT_COUNT:
            break
            
        parquet_path = await processor.download_parquet_from_hf(DATASET_PATH, parquet_file)
        df = processor.load_parquet_data(parquet_path)
        image_col = str(processor.find_image_column(df))
        
        for idx, row in df.iterrows():
            if rows_seen < START_FROM:
                rows_seen += 1
                continue
            if images_loaded >= EXTRACT_COUNT:
                break
            rows_seen += 1
            
            image_bytes = processor.extract_image_from_row(row[image_col])
            if image_bytes is None:
                continue
                
            try:
                # Verify image can be opened
                test_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                test_image.close()
                
                image_data_list.append(ImageData(
                    image_bytes=image_bytes,
                    image_id=images_loaded
                ))
                images_loaded += 1
                
                if images_loaded % 10 == 0:
                    print(f"📥 Loaded {images_loaded}/{EXTRACT_COUNT} images")
                
            except Exception as e:
                print(f"⚠ Failed to load image at row {rows_seen}: {e}")
                continue

    print(f"✅ Loaded {len(image_data_list)} valid images")

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)

    # Progress monitoring function
    def progress_monitor(progress_queue, stop_event):
        while not stop_event.is_set():
            try:
                msg = progress_queue.get(timeout=1)
                print(msg)
            except queue.Empty:
                continue
            except:
                break

    # PHASE 1: Generate all prompts using all GPUs
    print(f"\n🎭 PHASE 1: Generating prompts using {NUM_GPUS} GPUs...")
    
    prompt_task_queue = mp.Queue(maxsize=NUM_GPUS * 4)
    prompt_result_queue = mp.Queue()
    prompt_progress_queue = mp.Queue()
    
    # Start prompt generation workers
    prompt_processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=prompt_generation_worker,
            args=(gpu_id, prompt_task_queue, prompt_result_queue, prompt_progress_queue)
        )
        p.start()
        prompt_processes.append(p)
        time.sleep(1)  # Stagger starts

    # Start progress monitor
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(prompt_progress_queue, stop_progress), 
        daemon=True
    )
    progress_thread.start()

    # Submit all prompt generation tasks
    for image_data in image_data_list:
        task = PromptTask(image_bytes=image_data.image_bytes, image_id=image_data.image_id)
        prompt_task_queue.put(task)

    print(f"✅ Submitted {len(image_data_list)} prompt generation tasks")

    # Collect all prompts
    prompt_results = {}
    prompts_collected = 0
    
    while prompts_collected < len(image_data_list):
        try:
            image_id, prompt = prompt_result_queue.get(timeout=30)
            prompt_results[image_id] = prompt
            prompts_collected += 1
            
            if prompts_collected % 10 == 0:
                print(f"📝 Collected {prompts_collected}/{len(image_data_list)} prompts")
                
        except queue.Empty:
            print("⚠ Timeout waiting for prompt results")
            break

    # Shutdown prompt workers
    for _ in range(NUM_GPUS):
        prompt_task_queue.put(None)

    for p in prompt_processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()

    # Update image data with prompts
    for image_data in image_data_list:
        image_data.prompt = prompt_results.get(image_data.image_id, "fallback prompt")

    print(f"\n✅ PHASE 1 COMPLETE: Generated {len(prompt_results)} prompts")

    # Small delay to let GPU memory clear
    time.sleep(2)
    for i in range(NUM_GPUS):
        clear_gpu_memory(i)

    # PHASE 2: Generate all images using all GPUs
    print(f"\n🖼️  PHASE 2: Generating images using {NUM_GPUS} GPUs...")
    
    image_task_queue = mp.Queue(maxsize=NUM_GPUS * 4)
    image_result_queue = mp.Queue()
    image_progress_queue = mp.Queue()
    
    # Start image generation workers
    image_processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=image_generation_worker,
            args=(gpu_id, image_task_queue, image_result_queue, image_progress_queue, mask_config)
        )
        p.start()
        image_processes.append(p)
        time.sleep(1)  # Stagger starts

    # Restart progress monitor for phase 2
    stop_progress.set()  # Stop previous monitor
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(image_progress_queue, stop_progress), 
        daemon=True
    )
    progress_thread.start()

    # Submit all image generation tasks
    for image_data in image_data_list:
        if image_data.prompt:  # Only process if we have a prompt
            task = ImageGenTask(
                image_bytes=image_data.image_bytes, 
                image_id=image_data.image_id,
                prompt=image_data.prompt
            )
            image_task_queue.put(task)

    print(f"✅ Submitted {len(image_data_list)} image generation tasks")

    # Collect results and save in batches
    shared_tensor_batch = []
    batch_idx = 0
    
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

    results_collected = 0
    failed_results = 0
    
    while results_collected < len(image_data_list):
        try:
            image_id, tensor = image_result_queue.get(timeout=30)
            
            if tensor is not None:
                shared_tensor_batch.append(tensor)
                results_collected += 1
                
                # Save batch when it reaches BATCH_SIZE
                if len(shared_tensor_batch) >= BATCH_SIZE:
                    save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
                    stacked = torch.stack(shared_tensor_batch)
                    torch.save(stacked, save_path)
                    print(f"💾 Saved batch {batch_idx} with {len(shared_tensor_batch)} tensors to {save_path}")
                    print(f"📊 Tensor shape: {stacked.shape}, dtype: {stacked.dtype}")
                    
                    # Reset batch and increment batch index
                    shared_tensor_batch = []
                    batch_idx += 1
                    
                    # Clear some memory
                    del stacked
                    gc.collect()
            else:
                failed_results += 1
                results_collected += 1
                
            if results_collected % 10 == 0:
                print(f"📥 Collected {results_collected}/{len(image_data_list)} results")
                
        except queue.Empty:
            print("⚠ Timeout waiting for image results")
            break

    # Shutdown image workers
    for _ in range(NUM_GPUS):
        image_task_queue.put(None)

    for p in image_processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()

    # Save any remaining tensors in the final batch
    if shared_tensor_batch:
        save_path = OUTPUT_DIR / f"images_{batch_idx}.pt"
        stacked = torch.stack(shared_tensor_batch)
        torch.save(stacked, save_path)
        print(f"💾 Saved final batch {batch_idx} with {len(shared_tensor_batch)} tensors to {save_path}")
        print(f"📊 Final tensor shape: {stacked.shape}, dtype: {stacked.dtype}")

    # Stop progress monitoring
    stop_progress.set()

    print("\n" + "=" * 80)
    print("🎉 OPTIMIZED MULTI-GPU PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"📊 Total images processed: {results_collected}")
    print(f"📊 Failed images: {failed_results}")
    print(f"📊 Success rate: {((results_collected - failed_results) / results_collected * 100):.1f}%")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print(f"🖥️  GPUs used: {NUM_GPUS}")
    print(f"🎭 Processing method: Two-phase (prompts → images)")
    print(f"⚡ Model loading: Once per phase per GPU (major speedup!)")
    
    # Show performance improvements
    estimated_old_time = len(image_data_list) * 2 * 30  # 30s per load * 2 models * num images
    estimated_new_time = NUM_GPUS * 2 * 30  # 30s per load * 2 models * num GPUs
    time_saved = estimated_old_time - estimated_new_time
    print(f"⏱️  Estimated model loading time saved: ~{time_saved//60:.0f} minutes")
    
    print(f"\n🎭 Mask configuration:")
    print(f"   - Size ratio: {mask_config['min_size_ratio']:.1f} - {mask_config['max_size_ratio']:.1f}")
    print(f"   - Multiple masks: {mask_config['allow_multiple']}")
    print(f"   - Shapes: {', '.join(mask_config['allowed_shapes'])}")
    
    # Show consolidated file information
    print(f"\n📈 Output Files:")
    output_files = list(OUTPUT_DIR.glob("images_*.pt"))
    total_tensors = 0
    for f in sorted(output_files):
        try:
            tensors = torch.load(f, map_location='cpu')
            file_count = tensors.shape[0]
            total_tensors += file_count
            print(f"   {f.name}: {file_count} images, shape {tensors.shape}")
        except Exception as e:
            print(f"   {f.name}: Error loading - {e}")
    print(f"   Total: {len(output_files)} files, {total_tensors} images")

    print(f"\n✨ Key Optimizations Implemented:")
    print(f"   ✓ Two-phase processing (prompts → images)")
    print(f"   ✓ Models loaded once per phase per GPU")
    print(f"   ✓ No repeated model loading/unloading")
    print(f"   ✓ Efficient memory management")
    print(f"   ✓ Parallel processing across all GPUs")
    print(f"   ✓ Batch saving for memory efficiency")

if __name__ == "__main__":
    # Enable multiprocessing for CUDA
    torch.multiprocessing.set_sharing_strategy('file_system')
    asyncio.run(main())