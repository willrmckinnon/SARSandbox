import torch
from pathlib import Path
from ultralytics import YOLO






class Model:
    '''This class is meant to streamline model inferencing'''

    def __init__(self, model_path: str | Path, device=None):
        model_path = Path(model_path)

        if not model_path.is_file(): raise FileNotFoundError(model_path)

        # Set the device
        if device is None:
            if torch.cuda.is_available(): device = 0
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"

        self.device = device
        self.model = YOLO(str(model_path))







    def inference(self, image_path: str | Path, confidence: float = 0.25,) -> dict:
        """Return detections in the original image's pixel coordinates."""
        image_path = Path(image_path)

        if not image_path.is_file(): raise FileNotFoundError(image_path)

        result = self.model.predict(
            source=str(image_path),
            conf=confidence,
            device=self.device,
            verbose=False,
        )[0]

        return {
            "image_path": str(image_path),
            "predicted_boxes": result.boxes.xyxy.cpu().numpy(),
            "scores": result.boxes.conf.cpu().numpy(),
            "class_ids": result.boxes.cls.cpu().numpy().astype(int),
        }