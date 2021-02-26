SPADE_E2VID
=============================


Event to video reconstruction with spade module

- [Overview](##overview)
- [Getting Started](##getting-started)
- [Code](##code)
- [Evaluation datasets](##evaluation-datasets)
- [Non-polarity evaluation datasets](##non-polarity-evaluation-datasets)
- [Network weigths](##network-weigths)
- [Trainind dataset](##trainind-dataset)


## Overview

This repository contains the CODE for the paper:SPADE-E2VID: ![Spatially-Adaptive Denormalization for Event-Based Video Reconstruction](https://www.researchgate.net/publication/348821777_SPADE-E2VID_Spatially-Adaptive_Denormalization_for_Event-based_Video_Reconstruction)<br>. 
SPADE_E2VID uses a ConvLSTM and SPADE layers to reconstruct event-based videos. Our model compared with E2VID, have better reconstruction quality in early frames also has better contrast for all the reconstructios. We provide the code for training and testing.

Watch our video on youtube. <br>
[![Video](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/spade-e2vid.png)](https://www.youtube.com/watch?v=Dk1L0LeF7jQ)<br>

![SPADE_E2VID vs E2VID](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/compare.gif)<br>
A comparizon for SPADE_E2VID (our model) and E2VID.<br>

![SPADE_E2VID calendar](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/calendar.gif)<br>
Non-polarity Event-based video recontruction (Chinese Calendar).<br>

![SPADE_E2VID Shanghai Jiaotong Gate](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/jiaodamen.gif)<br>
Non-polarity Event-based video recontruction (the Shanghai Jiaotong Gate).<br>

## Getting Started
* Prerequisites<br>
Install PyTorch 1.3.0 (or higher), TorchVision, kornia, opencv, tqdm, pathlib, pandas, skimage, numpy, pytorch-msssim<br>


* Clone this repository <br>
```java
git clone https://github.com/RodrigoGantier/SPADE_E2VID.git

```
* Download the evaluation datasets and weigths. your directory tree should be as follows:<br>


├── SPADE_E2VID<br>
│   ├── cedric_firenet<br>
│   ├── dvs_datasets<br>
│   │   ├── bound_1<br>
│   │   ├── bound_2<br>
│   │   ├── bound_3<br>
│   │   ├── boxes_6dof<br>
│   │   ├── calibration<br>
│   │   ├── dynamic_6dof<br>
│   │   ├── office_zigzag<br>
│   │   ├── poster_6dof<br>
│   │   └── slider_depth<br>
│   ├── models<br>
│   │   ├── E2VID.pth.tar<br>
│   │   ├── E2VID_*.pth<br>
│   │   ├── E2VID_lightweight.pth.tar<br>
│   │   ├── firenet_1000.pth.tar<br>
│   │   ├── SPADE_E2VID.pth<br>
│   │   ├── SPADE_E2VID_2.pth<br>
│   │   └── SPADE_E2VID_ABS.pth<br>
│   ├── my_org_model<br>
│   ├── evs<br>
│   ├── org_e2vid<br>
│   ├── res<br>
│   └── spynet<br>

# Code
To run data evaluation with all models use the following code:
```java
python benchmark.py --root_dir /path/to/data/SPADE_E2VID

```
To run data evaluation with only one dataset and SPADE_E2VID, (you can choose fron 0 to 5):
```java
python test.py --root_dir /path/to/data/SPADE_E2VID --data_n 0

```
To train ESPADE_E2VID you can run:
```java
python train_e2v.py --root_dir /path/to/data/e2v_public --bs 2

```
Tested in ubuntu 18.04.4 LTS 

## Evaluation datasets


[DVS datasets](https://drive.google.com/file/d/1JH4QuJsrb2s67PYzueXRPQiCqJomnNuB/view?usp=sharing)<br>
if you want to download one by one, the individual links are below<br>

[calibration dataset](https://drive.google.com/drive/folders/1ctfatJRZlEMx0xdthKzhpjRU0PYu6QyS?usp=sharing)<br>
[boxes_6dof dataset](https://drive.google.com/drive/folders/1U6_6q1Rwn2S0_7OK_6m2o2XHexmdKsoR?usp=sharing)<br>
[slider_depth dataset](https://drive.google.com/drive/folders/1T6y21Wh1csOoRUhKDPHCkloMST2IkVrt?usp=sharing)<br>
[poster_6dof dataset](https://drive.google.com/drive/folders/1KQXR2KMjjeJZdHq2lMJ3P7TBG6kETHsV?usp=sharing)<br>
[office_zigzag dataset](https://drive.google.com/drive/folders/1Q00eskBZSy--Q-DkHX7xzboBe_KKTxle?usp=sharing)<br>
[dynamic_6dof dataset](https://drive.google.com/drive/folders/1bMHNB8AtAqgeGc8AXCukAiXP8MyvSWT-?usp=sharing)<br>


## Non-polarity evaluation datasets


[bund_1 dataset](https://drive.google.com/drive/folders/1KSGpOunVv47hU6nG9gOsEqxd6nfV7o9Q?usp=sharing)<br>
[bund_2 dataset](https://drive.google.com/drive/folders/1db4drgonbS-T6CSVxj4b8WeIGybGR30F?usp=sharing)<br>
[bund_3 dataset](https://drive.google.com/drive/folders/17OQUgnd2EUwugTMjLf2DSWUOfgSI11ea?usp=sharing)<br>


## Network weigths
[SPADE_E2VID](https://drive.google.com/file/d/1mOdIIJgZm2HiDk-dl40abrHEWeAtDXD0/view?usp=sharing)<br>
[SPADE_E2VID_ABS](https://drive.google.com/file/d/1dK6VEOTEeQ6_g4-cFUA0R80Lr84ktTXe/view?usp=sharing)<br>
[E2VID_](https://drive.google.com/file/d/1xrV8CFt45EBYT3aZihX7SJCOjAbjbd8h/view?usp=sharing)<br>
[E2VID_lightweight](https://drive.google.com/file/d/1MQXdVMHY0fb7c9QrP0eWPBS_uJQrayyZ/view?usp=sharing)<br>
[E2VID](https://drive.google.com/file/d/1q0rnm8OUIHk-II39qpxhp0tqBfIOK-7M/view?usp=sharing)<br>
[FireNet](https://drive.google.com/file/d/1Uqj8z8pDnq78JzoXdw-6radw3RPAyUPb/view?usp=sharing)<br>

## Trainind dataset


The Training dataset can be downkiad fron [this](https://drive.google.com/file/d/1usC0fsnRohMCMJSngMpLPb70w5_nYAeE/view?usp=sharing) link, are just 30 samples from the origianl 1000 samples

* NOTE: All the data is about 17.1 GB

## Citation:

```
@article{cadena2021spade,
  title={SPADE-E2VID: Spatially-Adaptive Denormalization for Event-Based Video Reconstruction},
  author={Cadena, Pablo Rodrigo Gantier and Qian, Yeqiang and Wang, Chunxiang and Yang, Ming},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={2488--2500},
  year={2021},
  publisher={IEEE}
}
```
