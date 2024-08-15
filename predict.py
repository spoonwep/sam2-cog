# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import numpy as np
import json

class Predictor(BasePredictor):
    def __init__(self):
        self.predictor = None

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        points: str = Input(description="Input points")
    ) -> Path:
        """Run a single prediction on the model"""
        
        # Load the image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        point_list = json.loads(points)
        input_points = []
        input_labels = []
        for point in point_list:
            input_points.append(point['coordinate'])
            input_labels.append(point['type'])
        input_point = np.array(input_points)
        input_label = np.array(input_labels)

        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        best_mask = masks[-1]
        cv2.imwrite(f'output_mask.png', best_mask * 255)
        return Path("output_mask.png")
