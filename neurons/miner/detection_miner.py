import base64
import io
import traceback
import tempfile
import os
import av
import json
import bittensor as bt
import numpy as np
import torch
from fastapi import APIRouter, Depends, Request
from PIL import Image
import torch.nn.functional as F

from neurons.miner.base_miner import BaseMiner, extract_testnet_metadata
from neurons.miner.detector import Detector
from bitmind.types import MinerType


class DetectionMiner(BaseMiner):
    """Miner specialized for image and video detection tasks."""

    def initialize_models(self):
        """Initialize the detection models."""
        self.detector = Detector(self.config)

    def get_miner_type(self):
        return MinerType.DETECTOR.value

    def setup_routes(self, router: APIRouter):
        """Setup detection-specific routes."""
        router.add_api_route(
            "/detect_image",
            self.detect_image,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )
        router.add_api_route(
            "/detect_video",
            self.detect_video,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )

    async def detect_image(self, request: Request):
        """Handle image detection requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        image_data = await request.body()

        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            "\u2713",
            f"Received image ({len(image_data)} bytes) from {signed_by}, type: {content_type}",
        )

        if content_type not in ("image/jpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected image/jpeg"
            )

        # testnet_metadata = extract_testnet_metadata(request.headers)
        # if len(testnet_metadata) > 0:
        #     bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            # Load image and convert to tensor format expected by detector
            image_array = np.array(Image.open(io.BytesIO(image_data)))
            # Convert to (C, H, W) format for the detector
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            bt.logging.info(image_tensor.shape)
            bt.logging.info(image_tensor[0].dtype)

            detect_tensor = image_tensor

            # --- Save tensor to request_image_set/image_*.pt ---
            request_dir = 'request_image_set'
            os.makedirs(request_dir, exist_ok=True)

            # Load and sort files
            files = [f for f in os.listdir(request_dir) if f.startswith('image_') and f.endswith('.pt')]
            files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else -1)

            if files:
                last_file = files[-1]
                last_file_path = os.path.join(request_dir, last_file)
                tensor = torch.load(last_file_path)  # shape: [T, C, H, W]
            else:
                last_file = None
                tensor = None

            # Convert to 3D tensor with 3 channels
            if image_tensor.dim() == 2:
                # Convert [H, W] to [3, H, W]
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)  # [H, W] -> [3, H, W]
            elif image_tensor.dim() == 3:
                current_channels = image_tensor.shape[0]
                if current_channels == 1:
                    # Convert [1, H, W] to [3, H, W]
                    image_tensor = image_tensor.repeat(3, 1, 1)
                elif current_channels != 3:
                    raise ValueError(f"Image tensor has {current_channels} channels, expected 1 or 3")
            else:
                raise ValueError(f"Image tensor has shape {image_tensor.shape}, expected 2D [H, W] or 3D [C, H, W]")

            # Resize to 224x224
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            image_tensor = image_tensor.squeeze(0)  # [1, 3, 224, 224] -> [3, 224, 224]

            # Concatenate into [T, C, H, W] format
            if tensor is not None:
                if tensor.dim() != 4 or tensor.shape[1] != 3:
                    raise ValueError(f"Expected existing tensor shape [T, 3, H, W], got {tensor.shape}")
                tensor = torch.cat([tensor, image_tensor.unsqueeze(0)], dim=0)  # [T, 3, 224, 224] + [1, 3, 224, 224]
            else:
                tensor = image_tensor.unsqueeze(0)  # [1, 3, 224, 224]

            # Save logic
            if tensor.shape[0] > 5000:
                if last_file:
                    torch.save(tensor[:-1], os.path.join(request_dir, last_file))
                    bt.logging.info(f"Saved {tensor.shape[0]-1} tensors to {last_file}")
                new_idx = int(last_file.split('_')[1].split('.')[0]) + 1 if last_file else 0
                new_file = f"image_{new_idx}.pt"
                torch.save(tensor[-1:], os.path.join(request_dir, new_file))  # Save as [1, 3, 224, 224]
                bt.logging.info(f"Saved 1 tensor to {new_file}")
            else:
                file_to_save = last_file or 'image_0.pt'
                torch.save(tensor, os.path.join(request_dir, file_to_save))
                bt.logging.info(f"Saved {tensor.shape[0]} tensors to {file_to_save}")

            # Before passing to conv2d (example)
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)  # [1, 3, 224, 224] -> [3, 224, 224] if unbatched conv2d
            elif tensor.dim() == 4:
                pass  # [N, 3, 224, 224] is fine for batched conv2d
            else:
                raise ValueError(f"Unexpected tensor shape for conv2d: {tensor.shape}")
            
            # --- End tensor saving logic ---
            
            ### PREDICT - using the updated Detector class with ConvNext+VIT model
            pred = self.detector.detect(detect_tensor, "image")
            bt.logging.success(pred)
            
            file_path = "probs_result.json"
            try:
                with open(os.path.join(request_dir, file_path), "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            except (FileNotFoundError, json.JSONDecodeError):
                data = []

            prob_json = {
                'index': tensor.shape[0],
                'prob': pred
            }
            for k, v in prob_json.items():
                if isinstance(v, bytes):
                    prob_json[k] = base64.b64encode(v).decode('utf-8')

            data.append(prob_json)

            with open(os.path.join(request_dir, file_path), "w") as f:
                json.dump(data, f, indent=2)

            return {"status": "success", "prediction": pred}

        except Exception as e:
            bt.logging.error(f"Error processing image: {e}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def detect_video(self, request: Request):
        """Handle video detection requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        video_data = await request.body()
        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            f"Received video ({len(video_data)} bytes) from {signed_by}, type: {content_type}",
        )
        if content_type not in ("video/mp4", "video/mpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected video/mp4 or video/mpeg"
            )

        # testnet_metadata = extract_testnet_metadata(request.headers)
        # if len(testnet_metadata) > 0:
        #     bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_data)
                temp_file.flush()

                with av.open(temp_path) as container:
                    video_stream = next(
                        (s for s in container.streams if s.type == "video"), None
                    )
                    if not video_stream:
                        raise ValueError("No video stream found")
                    try:
                        codec_info = (
                            f"name: {video_stream.codec.name}"
                            if hasattr(video_stream, "codec")
                            else "unknown"
                        )
                        bt.logging.info(f"Video codec: {codec_info}")
                    except Exception as codec_err:
                        bt.logging.warning(
                            f"Could not get codec info: {str(codec_err)}"
                        )
                    duration = container.duration / 1000000 if container.duration else 0
                    width = video_stream.width
                    height = video_stream.height
                    fps = video_stream.average_rate
                    bt.logging.info(
                        f"Video dimensions: ({width}, {height}), fps: {fps}, duration: {duration:.2f}s"
                    )
                    frames = []
                    for frame in container.decode(video=0):
                        img_array = frame.to_ndarray(format="rgb24")
                        frames.append(img_array)
                    bt.logging.info(f"Extracted {len(frames)} frames")
                    if not frames:
                        raise ValueError("No frames could be extracted from the video")
                    video_array = np.stack(frames)
                    # Convert to (C, T, H, W) format for the detector
                    video_tensor = torch.permute(
                        torch.from_numpy(video_array), (3, 0, 1, 2)  # (C, T, H, W)
                    )
                    bt.logging.info(video_tensor.shape)
                    bt.logging.info(video_tensor[0].dtype)
            ### PREDICT - using the updated Detector class with ConvNext+VIT model
            pred = self.detector.detect(video_tensor, "video")
            bt.logging.success(pred)
            return {"status": "success", "prediction": pred}
        except Exception as e:
            bt.logging.error(f"Error processing video: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    try:
        miner = DetectionMiner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()