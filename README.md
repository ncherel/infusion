# Infusion

This is the code for "Infusion: internal diffusion for inpainting of dynamic textures and complex motion " (https://arxiv.org/abs/2311.01090). You will find the basic elements for reproducing the results.

Website: https://infusion.telecom-paris.fr

The code has been simplified to split training and inference instead of running them together. It is indeed easier to share and debug this way.

![](teaser.gif)

## Interval training

We propose to use "interval training". Our models are trained on a single video and are used only a handful of times. To improve the performance, training and inference are mixed during the diffusion process, which is divided into intervals.
Contrary to the majority of diffusion models, the model is trained specifically for short intervals of similar noise levels. This significanly improves the results without increasing the size of the network.

![](teaser.png)

## Training the model

Requirements:

- cv2
- pytorch
- torchvision with PyAV (for video loading)
- numpy

Train with:

```
python train.py --video data/ants.mp4 --mask data/ants_mask.mp4 --steps 100 --interval-size 50 --small
```

/!\ Videos should be encoded losslessly (very important) so that they can be loaded with torchvision.
Encoding can be done with ffmpeg using the following command:

```
ffmpeg -pattern_type glob -i "frames/*.png" -c:v libx265 -x265-params lossless=1 video.mp4
```

Suggested parameters: 2000 steps for ants, fountain_1 with `--small` flag, 10000 steps for other videos

## Testing the model

A few trained models and data can be found here: [https://partage.imt.fr/index.php/s/ttZAEPBeHg5DBWp](https://partage.imt.fr/index.php/s/ttZAEPBeHg5DBWp) or on [Google Drive](https://drive.google.com/drive/folders/1Y_uYOiYKtKe6YABh7cnwzG67ZUaxfzg7?usp=drive_link).
After downloading the models and the data for testing, inference is run with:

```
python test.py --video data/ants.mp4 --mask data/ants_mask.mp4 --checkpoint model/ants/model_0000.pth --output ants_results --small
```

`--small` is only used for 256x256 videos from the DynTex databse (ants.mp4, fountain_1.mp4). The other videos are resized to 432x240 (jumping-girl-fire.mp4, young-jaws.mp4).


## Cite

```
@article{https://doi.org/10.1111/cgf.70070,
author = {Cherel, N. and Almansa, A. and Gousseau, Y. and Newson, A.},
title = {Infusion: Internal Diffusion for Inpainting of Dynamic Textures and Complex Motion},
journal = {Computer Graphics Forum},
volume = {n/a},
number = {n/a},
pages = {e70070},
keywords = {CCS Concepts, • Computing methodologies → Image processing},
doi = {https://doi.org/10.1111/cgf.70070},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70070},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.70070},
}
```
