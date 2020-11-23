# Ensemble-based Graph Convolutional Networks (EGCN)
This repository holds the codebase, dataset and models for the review purpose of CVPR 2021 submission 2156 "**Effective Skeleton-based Rehabilitation Exercise Assessment with Ensemble-based Graph Convolutional Networks**".

## Introduction
Rehabilitation exercise aims to restore physical functions from injury. With the release of motion sensors like Kinect, skeleton-based rehabilitation assessment attracts increasing research interest in computer vision. Existing attempts on skeleton-based rehabilitation exercise assessment usually rely on geometric features or statistical methods, which is a lack of effective skeleton data representation methods. Usually, skeleton data could be collected with sensors like Kinect or motion captures that provide two groups of features (i.e., position and orientation features). Graph Convolutional Network (GCN) has achieved encouraging performance for skeleton-based action recognition. However, it might not be able to fully make use of different features of the skeleton data. To advance the prior work, we propose an Ensemble-based GCN (EGCN) learning framework for rehabilitation exercise assessment.

<div align="center">
    <img src="resource/info/ensemble_framework.png">
</div>

## Visulization of Position and Angle Features of Skeleton Joints.
EGCN is able to make use of the position and angle features of the skeleton data for exercise evaluation purpose.
Below figures show the visulized views of the skeleton features from **KIMORE** and **UI-PRMD** datasets. The first row of below figures is 3d position features, and the second row is the angle features (a.k.a. orientation features).

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="250px" src="resource/samples/pos_kimore.gif"></td>
    <td><img width="250px" src="resource/samples/pos_uiprmd_kinect.gif"></td>
    <td><img width="250px" src="resource/samples/pos_uiprmd_vicon.gif"></td>
  </tr>
    <td><img width="250px" src="resource/samples/ang_kimore.gif"></td>
    <td><img width="250px" src="resource/samples/ang_uiprmd_kinect.gif"></td>
    <td><img width="250px" src="resource/samples/ang_uiprmd_vicon.gif"></td>
  </tr>
  <tr>
    <td style="text-align:center"><font size="1">Es5 in KIMORE (Kinect v2)<font></td>
    <td style="text-align:center"><font size="1">E1 in UI-PRMD (Kinect v2)<font></td>
    <td style="text-align:center"><font size="1">E1 in UI-PRMD (Vicon)<font></td>
  </tr>
</table>

## Prerequisites
Our codebase is based on **Python3** (>=3.5). There are a few dependencies to run the code. The major libraries we depend are
- [PyTorch](http://pytorch.org/) (Release version 0.4.0)
- Other Python libraries can be installed by `pip install -r requirements.txt`

### Installation
```
cd torchlight; python setup.py install; cd ..
```

## Data Preparation

We experimented on two skeleton-based action evaluation datasts: **UI-PRMD** and **KIMORE**.

### UI-PRMD
[UI-PRMD](https://webpages.uidaho.edu/ui-prmd/) is a data set of movements related to common exercises performed by patients in physical therapy and rehabilitation programs. The data set consists of 10 rehabilitation movements. A sample of 10 healthy individuals repeated each movement 10 times in front of two sensory systems for motion capturing: a Vicon optical tracker, and a Kinect camera. The data is presented as positions and angles of the body joints in the skeletal models provided by the Vicon and Kinect mocap systems. We use the consistent exercise repetitions in the [Reduced Dataset](https://webpages.uidaho.edu/ui-prmd/Reduced%20Data.zip) (174M), which is the same as the prior work. The dataset in our experimental format could be downloaded from [Google Drive](https://drive.google.com/file/d/1bGVFdyi-ZaTX9UGBV9EnuuBSS8i1iSqq/view?usp=sharing) and [Baidu Wangpan](https://pan.baidu.com/s/1E6ETUCxDUw1WQiQORNqtYg) (code:1234).

After uncompressing, put the data into folder ```./data/UI_PRMD```, rebuild the database by this command:
```
sh ./tools/gen/ui_prmd_gendata_all.sh
```

### KIMORE
For the KIMORE dataset, we perform manul segmentation on based on exercise specific features. Below are depth views of the exercises in KIMORE. Es2-4 are segmented as the left and right directions.

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="250px" src="resource/samples/Es1.gif"></td>
    <td><img width="250px" src="resource/samples/Es2_L.gif"></td>
    <td><img width="250px" src="resource/samples/Es3_L.gif"></td>
    <td><img width="250px" src="resource/samples/Es4_L.gif"></td>
  </tr>
  <tr>
    <td><font size="1">KIMORE Es1<font></td>
    <td><font size="1">KIMORE Es2(L)<font></td>
    <td><font size="1">KIMORE Es3(L)<font></td>
    <td><font size="1">KIMORE Es4(L)<font></td>
  </tr>
  <tr>
    <td><img width="250px" src="resource/samples/Es2_R.gif"></td>
    <td><img width="250px" src="resource/samples/Es3_R.gif"></td>
    <td><img width="250px" src="resource/samples/Es4_R.gif"></td>
    <td><img width="250px" src="resource/samples/Es5.gif"></td>
  </tr>
  <tr>
    <td><font size="1">KIMORE Es2(R)<font></td>
    <td><font size="1">KIMORE Es3(R)<font></td>
    <td><font size="1">KIMORE Es4(R)<font></td>
    <td><font size="1">KIMORE Es5<font></td>
  </tr>
</table>

KIMORE can be downloaded from [their website](https://vrai.dii.univpm.it/content/kimore-dataset). The segmented skeleton data could be downloaded from [Google Drive](https://drive.google.com/file/d/15GOEOJFcZDLqC8iEw9t3bi6LGJksc9w8/view?usp=sharing) and [Baidu Wangpan](https://pan.baidu.com/s/1VKRJTvhCxQwIYDdvBT2mYg) (code:1234). After that, this command should be used to build the database for training or evaluation:
After uncompressing, put the data into folder ```./data/KiMoRe```, then rebuild the database by this command:
```
sh ./tools/gen/kimore_gendata_all.sh
```

### Test the Trained Results
We provided the trained results of our **EGCN**. The results are in the folder ``` work_dir ```
To test the trained results of UI-PRMD, run
```
python ./tools/result/ui_prmd_folds_statistics.py
```
To test the trained results of KIMORE, run
```
python ./tools/result/kimore_folds_statistics.py
```

## Training
To train different ensemble strategies in EGCN for KIMORE, run
```
sh train_kimore_cv.sh
```

To train different ensemble strategies in EGCN for UI-PRMD, run
```
sh train_ui_prmd_cv.sh
```

## Contact
For any question, feel free to contact
```
xxx     : xxx@xxx
xxx     : xxx@xxx
```
