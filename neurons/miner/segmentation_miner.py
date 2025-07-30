import io
import os
import traceback

import json
import bittensor as bt
import numpy as np
import torch
from fastapi import APIRouter, Depends, Request, Response
from PIL import Image

from neurons.miner.base_miner import BaseMiner, extract_testnet_metadata
from neurons.miner.segmenter import Segmenter
from bitmind.types import MinerType

def save_image_tensor(image_tensor, output_dir="tensors", max_tensors_per_file=5000, file_prefix="image_tensors"):
    """
    Save image_tensor to .pt files, with up to max_tensors_per_file tensors per file.
    Each tensor has shape [1, 3, 224, 224], stacked into [N, 3, 224, 224] in the file.
    Converts grayscale tensors ([1, 224, 224] or [224, 224]) to RGB ([3, 224, 224]).
    
    Args:
        image_tensor (torch.Tensor): Input tensor of shape [1, 3, 224, 224], [1, 224, 224], or [224, 224]
        output_dir (str): Directory to save .pt files
        max_tensors_per_file (int): Maximum number of tensors per .pt file
        file_prefix (str): Prefix for output file names
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if tensor is grayscale and convert to RGB
    if image_tensor.shape == (224, 224) or image_tensor.shape == (1, 224, 224):
        if image_tensor.shape == (224, 224):
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: [1, 224, 224]
        image_tensor = image_tensor.repeat(1, 3, 1, 1)  # Convert to [1, 3, 224, 224]
    elif image_tensor.shape != (1, 3, 224, 224):
        raise ValueError(f"Expected tensor shape [1, 3, 224, 224], [1, 224, 224], or [224, 224], got {image_tensor.shape}")
    
    # Initialize tensor list and file counter
    tensor_list = []
    file_counter = 0
    
    # Check if the tensor file already exists and load the current tensor count
    current_file = os.path.join(output_dir, f"{file_prefix}_{file_counter}.pt")
    if os.path.exists(current_file):
        saved_tensors = torch.load(current_file)
        tensor_list = saved_tensors.tolist()
        file_counter = len(tensor_list) // max_tensors_per_file
    
    # Append the new tensor
    tensor_list.append(image_tensor)
    
    # If tensor_list reaches max_tensors_per_file, save to .pt file
    if len(tensor_list) >= max_tensors_per_file:
        output_file = os.path.join(output_dir, f"{file_prefix}_{file_counter}.pt")
        torch.save(torch.stack(tensor_list), output_file)
        tensor_list = []  # Reset tensor list
        file_counter += 1
    
    # Save remaining tensors if any
    if tensor_list:
        output_file = os.path.join(output_dir, f"{file_prefix}_{file_counter}.pt")
        torch.save(torch.stack(tensor_list), output_file)

class SegmentationMiner(BaseMiner):
    """Miner specialized for image segmentation tasks."""

    def initialize_models(self):
        """Initialize the segmentation models."""
        self.segmenter = Segmenter(self.config)

    def get_miner_type(self):
        return MinerType.SEGMENTER.value

    def setup_routes(self, router: APIRouter):
        """Setup segmentation-specific routes."""
        router.add_api_route(
            "/segment_image",
            self.segment_image,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )

    async def segment_image(self, request: Request):
        """Handle image segmentation requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        image_data = await request.body()

        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            "\u2713",
            f"Received image for segmentation ({len(image_data)} bytes) from {signed_by}, type: {content_type}",
        )

        if content_type not in ("image/jpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected image/jpeg"
            )

        testnet_metadata, gt_mask = extract_testnet_metadata(request.headers)
        if len(testnet_metadata) > 0:
            bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            image_array = np.array(Image.open(io.BytesIO(image_data)))
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            save_image_tensor(image_tensor, output_dir="test/segmenter", max_tensors_per_file=5000)

            ### SEGMENT - update the Segmenter class with your own model and preprocessing
            heatmap = self.segmenter.segment(image_tensor)

            # If testnet mask is provided, compute IOU for validation
            if gt_mask is not None:
                pred_mask = (heatmap > 0.5).astype(np.uint8)

                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                bt.logging.info(f"Testnet mask IOU: {iou:.4f}")
            
            heatmap_bytes = heatmap.astype(np.float16).tobytes()
            
            headers = {
                "X-Mask-Shape": f"{heatmap.shape[0]},{heatmap.shape[1]}",
                "X-Mask-Dtype": str(heatmap.dtype),
                "Content-Type": "application/octet-stream"
            }

            return Response(
                content=heatmap_bytes,
                headers=headers,
                status_code=200
            )

        except Exception as e:
            bt.logging.error(f"Error processing image segmentation: {e}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    try:
        miner = SegmentationMiner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
