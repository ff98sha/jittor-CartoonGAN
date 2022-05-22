# jittor-CartoonGAN
[Jittor](https://github.com/Jittor/Jittor) implementation of CartoonGAN [1] (CVPR 2018), adpated from [pytorch-CartoonGAN](https://github.com/znxlwm/pytorch-CartoonGAN).

## Usage
### 1.Download VGG19
[VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)

### 2.Train
```
python CartoonGAN.py --name your_project_name --src_data src_data_path --tgt_data tgt_data_path --vgg_model pre_trained_VGG19_model_path
```

### Folder structure
The following shows basic folder structure.
```
├── data
│   ├── src_data # src data (not included in this repo)
│   │   ├── train 
│   │   └── test
│   └── tgt_data # tgt data (not included in this repo)
│       ├── train 
│       └── pair # edge-promoting results to be saved here
│
├── CartoonGAN.py # training code
├── edge_promoting.py
├── utils.py
├── networks.py
└── name_results # results to be saved here
```

## Development Environment

Tested in:
* python 3.7.13
* numpy  1.21.6
* pillow 9.1.1
* jittor 1.3.2.7
* imageio 2.19.2

## Reference
[1] Chen, Yang, Yu-Kun Lai, and Yong-Jin Liu. "CartoonGAN: Generative Adversarial Networks for Photo Cartoonization." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

(Full paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
