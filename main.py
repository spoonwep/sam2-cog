import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import copy
import os
from bottle import get, post, request, run, response
import json

checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def save_masked_image(image, filepath):
    if image.shape[-1] == 4:
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(filepath, image)

def apply_mask(image, mask, alpha_channel=True):  # 应用并且响应mask
    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])  # 制作掩体
        alpha[mask == 1] = 255  # 兴趣地方标记为1，且为白色
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))  # 融合图像
    else:
        image = np.where(mask[..., None] == 1, image, 0)
    return image


def mask_image(image, mask, crop_mode_=True):  # 保存掩盖部分的图像（感兴趣的图像）
    if crop_mode_:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image = apply_mask(cropped_image, cropped_mask)
    else:
        masked_image = apply_mask(image, mask)

    return masked_image


def save_anns(anns, image, path):
    if len(anns) == 0:
        return
    index = 1
    for ann in anns:
        mask_2d = ann
        segment_image = copy.copy(image)
        masked_image = mask_image(segment_image, mask_2d)
        filename = str(index) + '.png'
        filepath = os.path.join(path, filename)
        save_masked_image(masked_image, filepath)
        index = index + 1

@post('/rembg')
def rembg():
    # 设置返回类型为json
    response.content_type = 'application/json'
    # 接收参数
    input_image = request.json.get("input_image")
    output_dir = request.json.get("output_dir")
    points = request.json.get("points")
    # 条件判断
    print(input_image)
    print(output_dir)
    print(points)
    if input_image is None or output_dir is None or points is None:
        return json.dumps({"status": "n", "msg": "invalid parameters"})

    image = cv2.imread(input_image)
    predictor.set_image(image)
    point_list = json.loads(points)
    input_points = []
    input_labels = []
    for point in point_list:
        input_points.append(point['coordinate'])
        input_labels.append(point['type'])
    input_point = np.array(input_points)
    input_label = np.array(input_labels)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_anns(masks, image, output_dir)

    return "2333"

run(host='localhost', port=7656)