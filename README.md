# SI-FCN:Soft-Thresholding Inception Network for Human Activity Recognition
## Approach
<div align="center">
  <img src="figures/SI-Net.png">
</div>
<p align="center">
  Figure 1: The Diagram of a Soft-Thresholding Inception Network.
</p>

<div align="center">
  <img src="figures/DNN.png">
</div>
<p align="center">
  Figure 2: The Diagram of a Fully Connected Network.
</p>

<div align="center">
  <img src="figures/Combination.png">
</div>
<p align="center">
  Figure 3: The Models training and deployment flowchart.
</p>


## implement
In this repository, all the models are implemented by [TensorFlow](https://github.com/tensorflow).

We use the data augmentation strategies with IS-Net and FCN.

In the training phase, we trained two models separately
## requirements
tensorflow-gpu==2.8
cuda==11.3.1
cudnn==8.2.1
## Dataset Download
[UCI-HAR](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
[WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
