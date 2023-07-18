# Traffic speed estimation for uncalibrated camera footage

Goal of this repository is to provide an easy way of estimating the speed of traffic from uncalibrated video footage.

## Structure

Our current approach is stored in the `speed_estimation` folder. It contains a `config.ini`, where some configuration
can be changed.

| Name                    | Description                                                                                                                                                                | Values   |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| fps                     | Default FPS to use, if they can't be detected from the provided video                                                                                                      |  integer |
| custom_object_detection | Wether to use the custom trained yolov5 model or pretrained yolov4 (default)                                                                                               | boolean  |
| avg_frame_count         | Output of meta statistics approach gets written here. Average frames a standard car was taking to drive through the CCTV segment (average tracked over a longer time frame |  float   |
|  speed_limit            |  Speed limit on the road segment shown in the video (in km/h)                                                                                                              |  integer |

The project is split into multiple modules, each handling a part of the total pipeline.
![](.github/modules.png)

The different modules of this project can be found inside the folder *speed_estimation/modules*
Currently, there are:
| Module Name | Folder | Description |
|-|-|-|
| Car Tracking | modules/object_detection | Contains yolov4 and v5 for detecting cars in a video frame |
| Depth map | modules/depth_map | Generates a depth map for a provided frame, using Pixelformer or Midas model |
| Hard Move Detection | modules/shake_detection | Detects if the frame moved |
| Stream-Conversion & Downsampler | modules/streaming | Reads a stream, caps it to 30FPS and provides the frames |
| Evaluation | modules/evaluation | Compares videos with the provided ground truth |

## Setup

Running the code can be done in two ways:

1. Locally
2. Docker (with CUDA support)

The advantage of the Docker container is that it supports CUDA acceleration out of the box. Locally, you'll have to set
it up yourself ;)

### Local Setup

0. (Have python virtual environments set up, e.g. through `conda`)
1. Install requirements from `environment.yml` or if you are using macOS from `environment_mac.yml`:\
`conda env create -f environment.yml`
2. `conda activate farsec`
3. Install [ffmpeg](https://ffmpeg.org/) for your machine.

```sh
# Mac
> brew install ffmpeg
# Ubuntu / Debian
> sudo apt install ffmpeg
```
4. Download the weights for the depth map from
   here: https://drive.google.com/file/d/1s7AdfwrV_6-svzfntBJih011u2IGkjf4/view?usp=share_link
5. Place the weights in that folder: `speed_estimation/modules/depth_map/pixelformer/pretrained`
6. Update the paths in `speed_estimation/paths.py`

### Docker Setup

0. (Have `docker` installed)
1. Go to `docker` directory in a terminal
2. Run `docker build .` Assign a tag, if you like.
3. Run the docker container with the following command:

```
docker run --rm \
        --gpus '"device=0"' -v $PATH_TO_REPO:/storage -v $PATH_TO_VIDEO_ROOT_FOLDER:/scratch2 \
        -t cv-cuda python3 /storage/speed_estimation/speed_estimation.py \
        "$PATH_TO_SESSION_DIRECTORY" "$PATH_TO_VIDEO_FILE_IN_DOCKER"
```

Replace `$PATH_TO_REPO`, `$PATH_TO_VIDEO_ROOT_FOLDER, "$PATH_TO_SESSION_DIRECTORY"` and `$PATH_TO_VIDEO_FILE_IN_DOCKER` with the paths on your
machine.

#### Note: This repository has a default configuration (`speed_estimation/config.ini`) that can be adjusted if necessary.

## Dataset

As a test dataset to run the estimation on, we provide you with a excerpt of the BrnoCompSpeed Dataset

1. Download the whole folder from here: https://1drv.ms/u/s!AmCOHF26iIAQgf1ladUQOKtY0an0dg?e=wa1iZX
2. Go to `speed_estimation/paths.py` and adjust the `session_path` accordingly

## Run

The path to the video should be given to `speed_estimation/speed_estimation.py` as argument.
If you do not give the path as argument adjust the `speed_estimation/paths.py` accordingly.

1. `cd speed_estimation`
2. `python speed_estimation.py /path/to/session /path/to/video.mp4` or `python speed_estimation.py` (this will use the default paths configured)
