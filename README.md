# Content-Adaptive Downsampling in Convolutional Neural Networks
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository accompanying the CVPR Workshop paper:

[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **Content-Adaptive Downsampling in Convolutional Neural Networks**. _CVPRW, The 6th Efficient Deep Learning for Computer Vision (ECV) Workshop_, 2023.

[Paper](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Hesse_Content-Adaptive_Downsampling_in_Convolutional_Neural_Networks_CVPRW_2023_paper.pdf) | [Preprint (arXiv)](https://arxiv.org/abs/2305.09504) | [Video](https://www.youtube.com/watch?v=E4iJPpWaJso) | [Poster](https://github.com/visinf/cad/blob/main/poster.jpeg) | [Supplemental](https://openaccess.thecvf.com/content/CVPR2023W/ECV/supplemental/Hesse_Content-Adaptive_Downsampling_in_CVPRW_2023_supplemental.pdf)

![Poster](https://github.com/visinf/cad/blob/main/poster.jpeg)

## Semantic Segmentation (Sec. 4.2)

### Pretrained Models

| Model       | mIoU Cityscapes  | Download |
| :---        |     :---:       | :---     |
| ResNet101+DeepLabv3 (OS=16)     | 0.762           | [best_deeplabv3_resnet101_cityscapes_os16_seed1.pth](https://download.visinf.tu-darmstadt.de/data/2023-cvpr-hesse-cad/best_deeplabv3_resnet101_cityscapes_os16_seed1.pth)| 
| ResNet101+DeepLabv3 (OS=8)   | 0.776           | [best_deeplabv3_resnet101_cityscapes_os8_seed1.pth](https://download.visinf.tu-darmstadt.de/data/2023-cvpr-hesse-cad/best_deeplabv3_resnet101_cityscapes_os8_seed1.pth) | 
| ResNet101+DeepLabv3 edge (OS=8->16)       | 0.773           | [best_deeplabv3_batch_ap_resnet101_cityscapes_os8_modeedges_os16till8_seed2_trimapwidth11_threshold0.15.pth](https://download.visinf.tu-darmstadt.de/data/2023-cvpr-hesse-cad/best_deeplabv3_batch_ap_resnet101_cityscapes_os8_modeedges_os16till8_seed2_trimapwidth11_threshold0.15.pth) | 
| ResNet101+DeepLabv3 learned (OS=8->16)     | 0.775           | [best_deeplabv3_ad_resnet101_cityscapes_modeend2end_seed0_default_tau1.0_lowresactive0.5_w_downsample_shared_andbatchnorm_shared.pth](https://download.visinf.tu-darmstadt.de/data/2023-cvpr-hesse-cad/best_deeplabv3_ad_resnet101_cityscapes_modeend2end_seed0_default_tau1.0_lowresactive0.5_w_downsample_shared_andbatchnorm_shared.pth) | 

### Available architectures

Specify the model architecture with '--model ARCH_NAME' and set the output stride using '--output_stride OUTPUT_STRIDE'. We here show example runs for **ResNet101+DeepLabv3**.

### Reproduce

#### 1. Install the required packages

Current channels:
- https://conda.anaconda.org/conda-forge/linux-64
- https://conda.anaconda.org/conda-forge/noarch
- https://conda.anaconda.org/pypi/linux-64
- https://conda.anaconda.org/pypi/noarch
- https://conda.anaconda.org/anaconda/linux-64
- https://conda.anaconda.org/anaconda/noarch
- https://conda.anaconda.org/pytorch/linux-64
- https://conda.anaconda.org/pytorch/noarch
- https://repo.anaconda.com/pkgs/main/linux-64
- https://repo.anaconda.com/pkgs/main/noarch
- https://repo.anaconda.com/pkgs/r/linux-64
- https://repo.anaconda.com/pkgs/r/noarch

conda create --name adaptive_downsampling --file requirements.txt
conda activate adaptive_downsampling

#### 2. Download cityscapes and extract it to 'datasets/cityscapes'

```
/datasets
  /cityscapes
  	/gtFine
		/leftImg8bit
```

#### 3. Train your models on Cityscapes

**For baseline models in Sec 4.2:**


```bash
python main.py --model deeplabv3_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 8 --output_stride 16 --data_root /datasets/cityscapes --random_seed 0

python main.py --model deeplabv3_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 8 --output_stride 8 --data_root /datasets/cityscapes --random_seed 0
```

**For content-adaptive downsampling models in Sec 4.2:**

Adaptive downsampling with edge mask:

```bash
python main.py --model deeplabv3_batch_ap_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 8 --output_stride 8 --data_root /datasets/cityscapes --trimap_width 11 --pooling_mask_mode edges_os16till8 --pooling_mask_edge_detection_treshold [0.15, 0.35, 0.95] --random_seed 0 --exp_name trimapwidth11_threshold[0.15, 0.35, 0.95]
```

Adaptive downsampling with learned mask:

```bash
python main_e2e_train.py --model deeplabv3_ad_resnet101 --dataset cityscapes --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 8 --data_root /datasets/cityscapes --random_seed 0 --exp_name default_tau1.0_lowresactive0.5_w_downsample_shared_andbatchnorm_shared --val_interval 100 --tau 1 --low_res_active 0.5

For evaluation:
python main_e2e_eval.py --model deeplabv3_ad_resnet101 --dataset cityscapes --gpu_id 0 --crop_size 768 --data_root /datasets/cityscapes --random_seed 0 --tau 1 --ckpt ./best_deeplabv3_ad_resnet101_cityscapes_modeend2end_seed0_default_tau1.0_lowresactive0.5_w_downsample_shared_andbatchnorm_shared.pth
```

#### 4. Evaluate your models

To evaluate your models run the respective training call (main.py) with the parameters ```--test_only``` and ```--ckpt```. 

#### 5. Get number of multiply-adds

Regular downsampling

```bash
python main_flops.py --model deeplabv3_resnet101 --dataset cityscapes --gpu_id 0 --output_stride [8,16] --data_root /datasets/cityscapes
```

Adaptive downsampling edge mask

```bash
python main_flops.py --model deeplabv3_ap_resnet101 --dataset cityscapes --gpu_id 0 --output_stride 8 --output_stride_from_trained 8 --data_root /datasets/cityscapes --pooling_mask_mode edges_os16till8 --trimap_width 11 --pooling_mask_edge_detection_treshold [0.15, 0.35, 0.95]
```

Adaptive downsampling learned mask

```bash
python main_e2e_flops.py --model deeplabv3_ad_resnet101 --dataset cityscapes --gpu_id 0 --crop_size 768 --data_root /datasets/cityscapes --random_seed 0 --ckpt ./best_deeplabv3_ad_resnet101_cityscapes_modeend2end_seed0_default_tau1.0_lowresactive0.5_w_downsample_shared_andbatchnorm_shared.pth
```

## Keypoints (Sec. 4.3)

This code is built on top of the official implementation of the following paper:

```text
"D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler. CVPR 2019.
```

[Paper on arXiv](https://arxiv.org/abs/1905.03561), [Project page](https://dsmn.ml/publications/d2-net.html)

### Downloading the models and datasets

For instruction on downloading the dataset please see the 'hpatches_sequences' folder

The model weights can be downloaded by running:

```bash
mkdir models
wget https://dusmanu.com/files/d2-net/d2_tf.pth -O models/d2_tf.pth
```

### Install the required packages

see ../segmentation

additionally install opencv:
pip install opencv-python

### Feature extraction

`extract_features.py` can be used to extract D2 features for a given list of images. 

Regular downsampling:
```bash
python extract_features.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --model_file models/d2_tf.pth --output_extension .sift_d2net_os[1,2,4,8]_512kpts --output_stride [1,2,4,8] --nr_keypoints 512
```

Adaptive downsampling (example for dilations 25 51 51):
```bash
python extract_features.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --model_file models/d2_tf.pth --output_extension .sift_apd2net_os1_512kpts_dils_25_51_51 --output_stride 1 --nr_keypoints 512 --des APD2Net --dilations 25 51 51
```

Adaptive downsampling (example for dilations 0 0 31):
```bash
python extract_features.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --model_file models/d2_tf.pth --output_extension .sift_apd2net_os4_512kpts_dils_0_0_31 --output_stride 4 --nr_keypoints 512 --des APD2Net --dilations 31
```

After extracting features, they can be evaluated by running hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb (add the methods that you want to evaluate)

### Estimate multiply-adds

Regular downsampling:
```bash
python eval_flops.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --output_stride [1,2,4,8] --nr_keypoints 512
```

Adaptive downsampling (example for dilations 25 51 51):
```bash
python eval_flops.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --output_stride 1 --nr_keypoints 512 --des APD2Net --dilations 25 51 51
```

Adaptive downsampling (example for dilations 0 0 31):
```bash
python eval_flops.py --gpu_id 0 --image_list_file image_list_hpatches_sequences.txt --output_stride 4 --nr_keypoints 512 --des APD2Net --dilations 0 0 31
```

## Acknowledgments

We would like to thank the contributors of the following repositories for using parts of their publicly available code:
- https://github.com/VainF/DeepLabV3Plus-Pytorch
- https://github.com/mihaidusmanu/d2-net

## Citation
If you find our work helpful please consider citing
```
@inproceedings{Hesse:2023:CAD,
  title     = {Content-Adaptive Downsampling in Convolutional Neural Networks},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), The 6$^\text{th}$ Efficient Deep Learning for Computer Vision (ECV) Workshop},
  year      = {2023}
}
```
