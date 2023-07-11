from __future__ import absolute_import, division, print_function

import configparser
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from .dataloaders.dataloader import NewDataLoader
from .networks.PixelFormer import PixelFormer
from .utils import post_process_depth, flip_lr

config = configparser.ConfigParser()
config.read("config.ini")

use_cpu = not torch.cuda.is_available()  # config.getboolean("device", "use_cpu")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


model_name = "pixelformer_kittieigen"
encoder = "large07"
dataset = "kitti"
input_height = 352
input_width = 1216
max_depth = 80
do_kb_crop = True
min_depth_eval = 1e-3
max_depth_eval = 80
checkpoint_path = "modules/depth_map/pixelformer/pretrained/kitti.pth"
# checkpoint_path='modules/depth_map/pixelformer/pretrained/kitti.pth'

args = {
    "model_name": model_name,
    "encoder": encoder,
    "dataset": dataset,
    "input_height": input_height,
    "input_width": input_width,
    "max_depth": max_depth,
    "do_kb_crop": do_kb_crop,
    "min_depth_eval": min_depth_eval,
    "max_depth_eval": max_depth_eval,
    "checkpoint_path": checkpoint_path,
    "mode": "test",
}

model_dir = os.path.dirname(checkpoint_path)
sys.path.append(model_dir)


def get_num_lines(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    return len(lines)


def generate_depth_map(data_folder: str, file_name: str, *, max_depth_o: int):
    """Test function."""

    if use_cpu:
        device = "cpu"
        torch.device(device)
    else:
        device = "cuda:0"
        torch.cuda.set_device(device)

    max_depth = args.max_depth if max_depth_o is None else max_depth_o
    output_path = os.path.join(data_folder, f"depth_map_{max_depth}.npy")
    dataloader = NewDataLoader(
        args,
        "test",
        file_list=[file_name],
        data_path=data_folder,
        do_kb_crop=do_kb_crop,
    )

    model = PixelFormer(version="large07", inv_depth=False, max_depth=max_depth)
    model = torch.nn.DataParallel(model, device_ids=[0])

    if use_cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    if not use_cpu:
        model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    def normalize_between_zero_one(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    pred_depths = []
    start_time = time.time()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataloader.data)):
            if use_cpu:
                image = Variable(sample["image"])
            else:
                image = Variable(sample["image"].cuda(device))

            # Predict
            depth_est = model(image)
            image_flipped = flip_lr(image)
            depth_est_flipped = model(image_flipped)
            depth_est = post_process_depth(depth_est, depth_est_flipped)

            pred_depth = depth_est.cpu().numpy().squeeze()

            if do_kb_crop:
                height, width = 352, 1216
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[
                    top_margin : top_margin + 352, left_margin : left_margin + 1216
                ] = pred_depth
                pred_depth = pred_depth_uncropped

            # max_depth = 100
            depth_map = normalize_between_zero_one(pred_depth) * max_depth
            # x: 1040 y: 370
            # x: 740 y:0
            # our_meters[214][375] - our_meters[20][528] session1_center
            # our_meters[230][426] - our_meters[75][720] session4_right
            # our_meters[213][751] - our_meters[132][511] session6_left

            # 1188, 266
            # 1113, 215

            # 494, 70
            # 494, 16

            return depth_map

    elapsed_time = time.time() - start_time
    print("Elapsed time: %s" % str(elapsed_time))
    print("Done.")
    return output_path


if __name__ == "__main__":
    raise Exception("This file should not be run as main file.")
