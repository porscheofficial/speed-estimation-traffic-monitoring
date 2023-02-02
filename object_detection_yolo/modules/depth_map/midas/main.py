import os
import cv2
import torch
import numpy as np


def normalize_between_zero_one(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def generate_depth_map(folder: str, image: str, max_depth_o: int):
    # Compatibility to pixelformer depth map generation!
    max_depth = max_depth_o

    output_path = os.path.join(folder, f"depth_map_midas_{max_depth}.npy")
    image = os.path.join(folder, image)

    model_type = (
        "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    )
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    torch.hub.set_dir("/scratch2/torchhub_cache")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.imread(image)
    print("IMAGE")
    print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

    our_meters = normalize_between_zero_one(output) * max_depth
    with open(output_path, "wb") as f:
        np.save(f, our_meters)

    return output_path
