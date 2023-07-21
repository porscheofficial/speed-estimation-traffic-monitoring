#!/bin/bash

cd "$(dirname "$0")" || exit

cd ../speed_estimation/modules/depth_map || exit

git clone https://github.com/ashutosh1807/PixelFormer.git

cp custom_pixelformer/test.py PixelFormer/pixelformer
cp custom_pixelformer/utils.py PixelFormer/pixelformer
cp custom_pixelformer/load.py PixelFormer/pixelformer
cp custom_pixelformer/dataloader.py PixelFormer/pixelformer/dataloaders

mkdir PixelFormer/pretrained