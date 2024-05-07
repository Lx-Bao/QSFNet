# QSFNet


This repo is an official implementation of the *QSFNet*.
**The code will be published after the paper is accepted.**

## Prerequisites

## Usage

### 1. Clone the repository

### 2. Training
Download the pretrained model **swin_base_patch4_window12_384_22k.pth** and **resnet34-333f7ec4.pth**. <br>

You can train the three stages entirely by using 
```
python Train_all.py
```
or train the three stages step by step, using
```
python Mtrain.py
python QAtrain.py
python Ttrain.py
```

### 3. Testing
```
python test_all.py
```

### 4. Evaluation

- We provide [saliency maps](https://pan.baidu.com/s/1iNippqmlOef_uHfWH33NZg) (fetch code: j9ko) of our QSFNet on VDT-2048 dataset.
- We also provide the [saliency maps](https://pan.baidu.com/s/1YLqu7LyulfPC3ZmrYidgYQ?pwd=rs87) (fetch code: rs87) of other comparison model in our paper on VDT-2048 dataset.
- The edge Ground Truth of the training set of VDT-2048 dataset can be download [here](https://pan.baidu.com/s/1T_zM6msG7e1Xg5bIzaWBxA?pwd=u450) (fetch code: u450)
## Citation
```
@article{bao2024quality,
  title={Quality-aware Selective Fusion Network for VDT Salient Object Detection},
  author={Bao, Liuxin and Zhou, Xiaofei and Lu, Xiankai and Sun, Yaoqi and Yin, Haibing and Hu, Zhenghui and Zhang, Jiyong and Yan, Chenggang},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```


- If you have any questions, feel free to contact me via: `lxbao@hdu.edu.cn` or `zxforchid@outlook.com`.
