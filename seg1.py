import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import os
from bottle import get, post, request, run, response
import json

sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def save_masked_image(image, filepath):
    if image.shape[-1] == 4:
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(filepath, image)
    # print(f"Saved as {filepath}")


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
    input_image = request.forms.get("input_image")
    output_dir = request.forms.get("output_dir")
    points = request.forms.get("points")
    # 条件判断
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

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_anns(masks, image, output_dir)

    return "2333"


@post('/rembg_auto')
def rembgAuto():
    input_image = request.forms.get("input")
    output_image = request.forms.get("output")
    os.system('rembg i -m isnet-general-use ' + input_image + ' ' + output_image)
    return {"status": "y"}


run(host='localhost', port=8080)
