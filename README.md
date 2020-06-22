# SPADE_E2VID
=============================

Event to video reconstruction with spade module

- [Overview](##overview)
- [Getting Started](##getting-started)
- [Evaluation datasets](##evaluation-datasets)
- [Non-polarity evaluation datasets](##non-polarity-evaluation-datasets)
- [Network weigths](##network-weigths)

=============================

## Overview
This repository contains the CODE for SPADE_E2VID. SPADE_E2VID uses a ConvLSTM and SPADE layers to reconstruct event-based videos. Our model compared with E2VID, have better reconstruction quality in early frames also has better contrast for all the reconstructios. We provide the code for training and testing.


![SPADE_E2VID vs E2VID](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/compare.gif)

A comparizon for SPADE_E2VID (our model) and E2VID.

![SPADE_E2VID calendar](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/calendar.gif)

A Event-based video recontruction for a chinese calendar.

![SPADE_E2VID Shanghai Jiaotong Gate](https://github.com/RodrigoGantier/SPADE_E2VID/blob/master/res/jiaodamen.gif)

A Event-based video recontruction for the Shanghai Jiaotong Gate.
=============================


## Getting Started
* Clone this repository, download the evaluation datasets and weigths. your directory tree should be as follows:


-[├── SPADE_E2VID]()
-[│   ├── cedric_firenet]()
-│   ├── dvs_datasets
-│   │   ├── bound_1
-│   │   ├── bound_2
-│   │   ├── bound_3
-│   │   ├── boxes_6dof
-│   │   ├── calibration
-│   │   ├── dynamic_6dof
-│   │   ├── office_zigzag
-│   │   ├── poster_6dof
-│   │   └── slider_depth
-│   ├── models
-│   │   ├── E2VID.pth.tar
-│   │   ├── E2VID_*.pth
-│   │   ├── E2VID_lightweight.pth.tar
-│   │   ├── firenet_1000.pth.tar
-│   │   ├── SPADE_E2VID.pth
-│   │   ├── SPADE_E2VID_2.pth
-│   │   └── SPADE_E2VID_ABS.pth
-│   ├── my_org_model
-│   ├── org_e2vid
-│   ├── res
-│   └── spynet

=============================

## Evaluation datasets

[DVS datasets](https://drive.google.com/file/d/1JH4QuJsrb2s67PYzueXRPQiCqJomnNuB/view?usp=sharing)

if you want to download one by one, the individual links are below

[calibration dataset](https://drive.google.com/drive/folders/1ctfatJRZlEMx0xdthKzhpjRU0PYu6QyS?usp=sharing)

[boxes_6dof dataset](https://drive.google.com/drive/folders/1U6_6q1Rwn2S0_7OK_6m2o2XHexmdKsoR?usp=sharing)

[slider_depth dataset](https://drive.google.com/drive/folders/1T6y21Wh1csOoRUhKDPHCkloMST2IkVrt?usp=sharing)

[poster_6dof dataset](https://drive.google.com/drive/folders/1KQXR2KMjjeJZdHq2lMJ3P7TBG6kETHsV?usp=sharing)

[office_zigzag dataset](https://drive.google.com/drive/folders/1Q00eskBZSy--Q-DkHX7xzboBe_KKTxle?usp=sharing)

[dynamic_6dof dataset](https://drive.google.com/drive/folders/1bMHNB8AtAqgeGc8AXCukAiXP8MyvSWT-?usp=sharing)


## Non-polarity evaluation datasets

[bund_1 dataset](https://drive.google.com/drive/folders/1KSGpOunVv47hU6nG9gOsEqxd6nfV7o9Q?usp=sharing)

[bund_2 dataset](https://drive.google.com/drive/folders/1db4drgonbS-T6CSVxj4b8WeIGybGR30F?usp=sharing)

[bund_3 dataset](https://drive.google.com/drive/folders/17OQUgnd2EUwugTMjLf2DSWUOfgSI11ea?usp=sharing)


## Network weigths
[SPADE_E2VID weight](https://drive.google.com/file/d/1mOdIIJgZm2HiDk-dl40abrHEWeAtDXD0/view?usp=sharing)

[SPADE_E2VID non-polarity weight](https://drive.google.com/file/d/1dK6VEOTEeQ6_g4-cFUA0R80Lr84ktTXe/view?usp=sharing)

[E2VID* weight](https://drive.google.com/file/d/1xrV8CFt45EBYT3aZihX7SJCOjAbjbd8h/view?usp=sharing)

[E2VID_lightweight weight](https://drive.google.com/file/d/1MQXdVMHY0fb7c9QrP0eWPBS_uJQrayyZ/view?usp=sharing)

[E2VID weight](https://drive.google.com/file/d/1q0rnm8OUIHk-II39qpxhp0tqBfIOK-7M/view?usp=sharing)

[FireNet weight](https://drive.google.com/file/d/1Uqj8z8pDnq78JzoXdw-6radw3RPAyUPb/view?usp=sharing)
=============================

The Training dataset can be downkiad fron [this](https://drive.google.com/file/d/1usC0fsnRohMCMJSngMpLPb70w5_nYAeE/view?usp=sharing) link, are just 30 samples from the origianl 1000 samples


