from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = f"./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)