import os
from pathlib import Path
from PIL import Image
import torch
import torch.multiprocessing as mp
import numpy as np
from io import BytesIO
import re
import asyncio
import sys
from pathlib import Path as SysPath
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
from typing import List, Tuple
import time

sys.path.append(str(SysPath(__file__).resolve().parent.parent))

from extract_image_tensors import ExtractImageProcessor

# Real: bm-real, MS-COCO-unique-256, open-image-v7-256, celeb-a-hq, dtd, caltech-101
# Synthetic: bm-aura-imagegen, GenImage_MidJourney, JourneyDB
# Semi-Synthetic: face-swap
DATASET_PATH = "bitmind/JourneyDB"
START_FROM = 60000  # index to start downloading from
EXTRACT_COUNT = 40000  # total number of images to extract
BATCH_SIZE = 5000  # number of images per .pt file
# DIR: test/real_image_batches, test/synth_image_batches, semi_synth_image_batches
OUTPUT_DIR = Path("test/synth_image_batches") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_available_gpus():
    """Get list of available GPU devices"""
    if not torch.cuda.is_available():
        return []
    return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

def process_images_worker(image_data_list: List[Tuple[bytes, int]], device: torch.device) -> List[Tuple[torch.Tensor, int]]:
    """Worker function to process images on a specific GPU"""
    torch.cuda.set_device(device)
    results = []
    
    for image_bytes, original_idx in image_data_list:
        if image_bytes is None:
            continue
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(device)
            # Move to CPU immediately to avoid GPU memory issues
            tensor_cpu = tensor.cpu()
            results.append((tensor_cpu, original_idx))
        except Exception as e:
            print(f"Error processing image {original_idx}: {e}")
            continue
    
    return results

class MultiGPUImageProcessor:
    def __init__(self):
        self.devices = get_available_gpus()
        if not self.devices:
            self.devices = [torch.device('cpu')]
        print(f"Using {len(self.devices)} device(s): {self.devices}")
        
        self.processor = ExtractImageProcessor()
        self.tensor_accumulator = []  # Accumulate tensors here
        self.batch_idx = self._get_next_batch_idx()
        
    def _get_next_batch_idx(self):
        """Get the next batch index based on existing files"""
        existing_pt_files = list(OUTPUT_DIR.glob("images_*.pt"))
        max_idx = -1
        pt_pattern = re.compile(r"images_(\d+)\.pt")
        for f in existing_pt_files:
            m = pt_pattern.match(f.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1
    
    def _save_batch_if_ready(self):
        """Save batch if we have BATCH_SIZE tensors"""
        if len(self.tensor_accumulator) >= BATCH_SIZE:
            # Take exactly BATCH_SIZE tensors
            batch_tensors = self.tensor_accumulator[:BATCH_SIZE]
            self.tensor_accumulator = self.tensor_accumulator[BATCH_SIZE:]
            
            # Stack and save
            stacked = torch.stack(batch_tensors)
            save_path = OUTPUT_DIR / f"images_{self.batch_idx}.pt"
            torch.save(stacked, save_path)
            
            print(f"✅ Saved batch {self.batch_idx}: {len(batch_tensors)} tensors to {save_path}")
            print(f"   Progress: {self.batch_idx * BATCH_SIZE} tensors saved total")
            print(f"   Remaining in accumulator: {len(self.tensor_accumulator)} tensors")
            
            self.batch_idx += 1
            return True
        return False
    
    def _save_final_batch(self):
        """Save any remaining tensors in the final batch"""
        if self.tensor_accumulator:
            stacked = torch.stack(self.tensor_accumulator)
            save_path = OUTPUT_DIR / f"images_{self.batch_idx}.pt"
            torch.save(stacked, save_path)
            
            print(f"✅ Saved final batch {self.batch_idx}: {len(self.tensor_accumulator)} tensors to {save_path}")
            print(f"   Total tensors saved: {(self.batch_idx * BATCH_SIZE) + len(self.tensor_accumulator)}")
            
            self.tensor_accumulator = []
    
    async def extract_images_parallel(self):
        """Main extraction method using multiple GPUs with continuous saving"""
        parquet_files = self.processor.list_parquet_files(DATASET_PATH)
        
        print(f"🚀 Starting parallel extraction:")
        print(f"   Target: {EXTRACT_COUNT} images")
        print(f"   Batch size: {BATCH_SIZE} tensors per .pt file")
        print(f"   Starting from batch index: {self.batch_idx}")
        print(f"   Available devices: {len(self.devices)}")
        
        images_processed = 0
        rows_seen = 0
        
        for parquet_idx, parquet_file in enumerate(parquet_files):
            if images_processed >= EXTRACT_COUNT:
                break
            
            print(f"\n📁 Processing parquet {parquet_idx + 1}/{len(parquet_files)}: {parquet_file}")
            
            # Download and load parquet
            parquet_path = await self.processor.download_parquet_from_hf(DATASET_PATH, parquet_file)
            df = self.processor.load_parquet_data(parquet_path)
            image_col = self.processor.find_image_column(df)
            
            # Collect image data from this parquet
            image_data_batch = []
            for idx, row in df.iterrows():
                if rows_seen < START_FROM:
                    rows_seen += 1
                    continue
                if images_processed >= EXTRACT_COUNT:
                    break
                
                rows_seen += 1
                image_bytes = self.processor.extract_image_from_row(row[image_col])
                if image_bytes is not None:
                    image_data_batch.append((image_bytes, images_processed))
                    images_processed += 1
            
            if not image_data_batch:
                print(f"   ⚠️  No valid images found in this parquet")
                continue
            
            print(f"   📊 Found {len(image_data_batch)} valid images")
            
            # Process images in parallel across GPUs
            processed_tensors = await self._process_batch_parallel(image_data_batch)
            
            # Add processed tensors to accumulator
            self.tensor_accumulator.extend(processed_tensors)
            
            print(f"   ✨ Processed {len(processed_tensors)} tensors (accumulator: {len(self.tensor_accumulator)})")
            
            # Save batches as they become ready
            while self._save_batch_if_ready():
                pass  # Keep saving until we don't have enough for a full batch
        
        # Save any remaining tensors
        self._save_final_batch()
        
        print(f"\n🎉 Extraction complete!")
        print(f"   Total images processed: {len(self.tensor_accumulator) + (self.batch_idx * BATCH_SIZE)}")
    
    async def _process_batch_parallel(self, image_data_batch: List[Tuple[bytes, int]]) -> List[torch.Tensor]:
        """Process a batch of images in parallel across GPUs"""
        num_gpus = len(self.devices)
        chunk_size = max(1, len(image_data_batch) // num_gpus)
        
        # Split data across GPUs
        gpu_chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            if i == num_gpus - 1:  # Last GPU gets remaining images
                end_idx = len(image_data_batch)
            else:
                end_idx = (i + 1) * chunk_size
            
            if start_idx < len(image_data_batch):
                gpu_chunks.append((image_data_batch[start_idx:end_idx], self.devices[i]))
        
        # Process chunks in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=len(gpu_chunks)) as executor:
            # Submit all GPU tasks
            futures = []
            for chunk_data, device in gpu_chunks:
                if chunk_data:  # Only submit if there's data
                    future = executor.submit(process_images_worker, chunk_data, device)
                    futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"   ❌ Error in GPU worker: {e}")
        
        # Sort by original index to maintain order
        all_results.sort(key=lambda x: x[1])
        
        # Extract just the tensors
        tensors = [tensor for tensor, _ in all_results]
        
        return tensors

# Alternative simpler approach using threading for smaller memory footprint
class ThreadedBatchProcessor:
    def __init__(self):
        self.devices = get_available_gpus()
        if not self.devices:
            self.devices = [torch.device('cpu')]
        
        self.processor = ExtractImageProcessor()
        self.tensor_accumulator = []
        self.batch_idx = self._get_next_batch_idx()
        self.lock = threading.Lock()
        
    def _get_next_batch_idx(self):
        existing_pt_files = list(OUTPUT_DIR.glob("images_*.pt"))
        max_idx = -1
        pt_pattern = re.compile(r"images_(\d+)\.pt")
        for f in existing_pt_files:
            m = pt_pattern.match(f.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1
    
    def _save_batch_if_ready(self):
        """Thread-safe batch saving"""
        with self.lock:
            if len(self.tensor_accumulator) >= BATCH_SIZE:
                batch_tensors = self.tensor_accumulator[:BATCH_SIZE]
                self.tensor_accumulator = self.tensor_accumulator[BATCH_SIZE:]
                
                stacked = torch.stack(batch_tensors)
                save_path = OUTPUT_DIR / f"images_{self.batch_idx}.pt"
                torch.save(stacked, save_path)
                
                print(f"✅ Saved batch {self.batch_idx}: {BATCH_SIZE} tensors to {save_path}")
                print(f"   Progress: {self.batch_idx * BATCH_SIZE} tensors saved total")
                
                self.batch_idx += 1
                return True
            return False
    
    def process_images_on_gpu(self, image_data_list: List[Tuple[bytes, int]], device: torch.device):
        """Process images on a specific GPU (thread worker)"""
        torch.cuda.set_device(device)
        
        for image_bytes, original_idx in image_data_list:
            if image_bytes is None:
                continue
            try:
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                image = image.resize((224, 224), Image.LANCZOS)
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(device)
                tensor_cpu = tensor.cpu()
                
                # Thread-safe addition to accumulator
                with self.lock:
                    self.tensor_accumulator.append(tensor_cpu)
                
                # Check if we can save a batch
                self._save_batch_if_ready()
                
            except Exception as e:
                print(f"Error processing image {original_idx}: {e}")

async def main():
    """Main function"""
    print("=== Multi-GPU Image Processing with Continuous Batch Saving ===")
    
    # Set multiprocessing start method for CUDA compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    processor = MultiGPUImageProcessor()
    await processor.extract_images_parallel()

if __name__ == "__main__":
    asyncio.run(main())