from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import numpy as np
from bottle import get, post, request, run, response

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Load the image
image = cv2.imread('example-1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# coordinates
input_points = np.array([[577, 355]])
input_labels = np.array([1])  # Assuming all points are foreground

# Generate the mask
masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
# save all masks
#for i, mask in enumerate(masks):
#    cv2.imwrite(f'output_mask_{i}.png', mask * 255)


# Select the best mask (typically the last one)
#best_mask = masks[-1]

# Define your RGBA color (e.g., red with 50% transparency)
#rgba_color = [255, 0, 0, 128]  # Red with 50% opacity

# Create an empty RGBA image
#rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

# Apply the RGBA color to the selected area using the mask
#rgba_image[best_mask == 1] = rgba_color

# Save the RGBA image
#cv2.imwrite('output_rgba_image.png', rgba_image)