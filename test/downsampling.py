import torch
import torch.nn.functional as F
import os
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock
import time

class MultiGPUTensorProcessor:
    def __init__(self, backup=True, batch_size=None):
        self.backup = backup
        self.device_count = torch.cuda.device_count()
        self.devices = [f'cuda:{i}' for i in range(self.device_count)] if self.device_count > 0 else ['cpu']
        self.batch_size = batch_size or min(10, max(1, 100 // self.device_count))  # Adaptive batch size
        self.lock = Lock()
        
        print(f"Available devices: {self.devices}")
        print(f"Processing batch size per GPU: {self.batch_size}")
    
    def resize_tensor_cpu_fallback(self, tensor, new_size=(224, 224)):
        """
        CPU fallback for tensor resizing with proper dtype handling
        """
        original_dtype = tensor.dtype
        
        # Convert to float32 if tensor is uint8/byte type
        if tensor.dtype == torch.uint8:
            tensor_float = tensor.float()
            resized = F.interpolate(tensor_float, size=new_size, mode='bilinear', align_corners=False)
            # Convert back to uint8
            result = torch.clamp(resized, 0, 255).byte()
        else:
            result = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
        
        return result
    
    def resize_tensor_on_device(self, tensor, device, new_size=(224, 224)):
        """
        Resize tensor on specific device with proper dtype handling
        """
        try:
            original_dtype = tensor.dtype
            
            # Convert to float32 if tensor is uint8/byte type for interpolation
            if tensor.dtype == torch.uint8:
                tensor_float = tensor.float()
                needs_conversion_back = True
            else:
                tensor_float = tensor
                needs_conversion_back = False
            
            # Move tensor to device
            tensor_gpu = tensor_float.to(device, non_blocking=True)
            
            # Resize using bilinear interpolation
            resized = F.interpolate(
                tensor_gpu, 
                size=new_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Convert back to original dtype if needed
            if needs_conversion_back:
                # Clamp values to valid uint8 range and convert back
                resized = torch.clamp(resized, 0, 255).byte()
            
            # Move back to CPU for saving
            result = resized.cpu()
            
            # Clear GPU memory
            del tensor_gpu, resized
            if needs_conversion_back:
                del tensor_float
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            with self.lock:
                print(f"Error on device {device}: {str(e)}")
            # Fallback to CPU with proper dtype handling
            return self.resize_tensor_cpu_fallback(tensor, new_size)
    
    def process_single_file(self, pt_file, device_id):
        """
        Process a single .pt file on specified device
        """
        device = self.devices[device_id % len(self.devices)]
        
        try:
            with self.lock:
                print(f"[{device}] Processing: {pt_file.name}")
            
            # Load tensor
            tensor = torch.load(pt_file, map_location='cpu')
            
            # Check tensor dtype and inform user
            with self.lock:
                print(f"[{device}] Tensor dtype: {tensor.dtype}")
            
            # Validate tensor shape
            if len(tensor.shape) != 4:
                with self.lock:
                    print(f"[{device}] Warning: {pt_file.name} has unexpected shape {tensor.shape}, skipping...")
                return False
            
            if tensor.shape[2:] != (256, 256):
                with self.lock:
                    print(f"[{device}] Warning: {pt_file.name} has spatial dimensions {tensor.shape[2:]}, not (256, 256), skipping...")
                return False
            
            original_shape = tensor.shape
            
            # Create backup if requested
            if self.backup:
                backup_file = pt_file.with_suffix('.pt.backup')
                if not backup_file.exists():
                    torch.save(tensor, backup_file)
                    with self.lock:
                        print(f"[{device}] Backup created: {backup_file.name}")
            
            # Process tensor in batches if it's very large
            if tensor.shape[0] > self.batch_size:
                resized_batches = []
                
                for i in range(0, tensor.shape[0], self.batch_size):
                    batch = tensor[i:i+self.batch_size]
                    resized_batch = self.resize_tensor_on_device(batch, device)
                    resized_batches.append(resized_batch)
                
                # Concatenate all batches
                resized_tensor = torch.cat(resized_batches, dim=0)
            else:
                # Process entire tensor at once
                resized_tensor = self.resize_tensor_on_device(tensor, device)
            
            # Save resized tensor
            torch.save(resized_tensor, pt_file)
            
            with self.lock:
                print(f"[{device}] Completed: {pt_file.name} - Shape: {original_shape} -> {resized_tensor.shape}")
            
            return True
            
        except Exception as e:
            with self.lock:
                print(f"[{device}] Error processing {pt_file.name}: {str(e)}")
            return False
    
    def process_files_parallel(self, pt_files, max_workers=None):
        """
        Process multiple files in parallel using multiple GPUs
        """
        if max_workers is None:
            max_workers = min(len(pt_files), len(self.devices) * 2)  # 2 threads per GPU
        
        print(f"Processing {len(pt_files)} files with {max_workers} workers across {len(self.devices)} devices")
        
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, pt_file, i): pt_file
                for i, pt_file in enumerate(pt_files)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                pt_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Error in future for {pt_file.name}: {str(e)}")
                    failed += 1
        
        end_time = time.time()
        
        print(f"\nProcessing Summary:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per file: {(end_time - start_time) / len(pt_files):.2f} seconds")

def process_pt_files_multigpu(directory_path, backup=True, batch_size=None, max_workers=None):
    """
    Process all .pt files in directory using multiple GPUs
    
    Args:
        directory_path: Path to directory containing .pt files
        backup: Whether to create backup files
        batch_size: Batch size for processing large tensors (None for auto)
        max_workers: Number of worker threads (None for auto)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Find all .pt files
    pt_files = list(directory.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {directory_path}")
        return
    
    print(f"Found {len(pt_files)} .pt files to process")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU processing")
    
    # Initialize processor
    processor = MultiGPUTensorProcessor(backup=backup, batch_size=batch_size)
    
    # Process files
    processor.process_files_parallel(pt_files, max_workers=max_workers)

def main():
    # Get user input
    directory_path = input("Enter the directory path containing .pt files: ").strip()
    
    # Backup option
    create_backup = input("Create backup files? (y/n, default: y): ").strip().lower()
    backup = create_backup != 'n'
    
    # Batch size option
    batch_size_input = input("Enter batch size for large tensors (default: auto): ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else None
    
    # Max workers option
    max_workers_input = input("Enter max worker threads (default: auto): ").strip()
    max_workers = int(max_workers_input) if max_workers_input.isdigit() else None
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"Directory: {directory_path}")
    print(f"Backup: {backup}")
    print(f"Batch size: {'auto' if batch_size is None else batch_size}")
    print(f"Max workers: {'auto' if max_workers is None else max_workers}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Confirm
    confirm = input(f"\nProceed with processing? (y/n): ").strip().lower()
    
    if confirm == 'y':
        process_pt_files_multigpu(
            directory_path, 
            backup=backup, 
            batch_size=batch_size,
            max_workers=max_workers
        )
    else:
        print("Operation cancelled")

# Direct usage function
def resize_pt_files_multigpu(directory_path, backup=True, batch_size=None, max_workers=None):
    """
    Direct function to resize .pt files using multiple GPUs
    
    Args:
        directory_path: Path to directory containing .pt files
        backup: Whether to create backup files
        batch_size: Batch size for processing (None for auto)
        max_workers: Number of worker threads (None for auto)
    """
    process_pt_files_multigpu(directory_path, backup, batch_size, max_workers)

if __name__ == "__main__":
    main()
    
    # Example of direct usage:
    # resize_pt_files_multigpu("/path/to/your/pt/files", backup=True, batch_size=10, max_workers=8)