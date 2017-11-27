# Semantic Segmentation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

- This repository is for udacity self-driving car nanodegree project - `Semantic Segmentation`.
- Implement this paper: ["Fully Convolutional Networks for Semantic Segmentation (2015)"](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- See [`FCN-VGG16.ipynb`](./FCN-VGG16.ipynb)


## Implementation Details
### Network
`FCN-8s` with `VGG16` as below figure.

![network figure](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/11/figure15.png)

### Dataset
- [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).

### Hyperparameters

![hyperparams](./assets/hyperparams.png)

Learning rate, batch size and keep probability was tunned by random search. if you want to see detail: [`Link`](./fcn8s-vgg-tunning-params.ipynb)



- Optimizer: `Adam`
- Learning rate: `0.0002395`
- Deconvolution `l2 regularization` factor: `1e-3`
- Batch size: `2`
- Training epochs: `30`
- `Keep prob` for dropout (VGG): `0.495`

#### 

## Results
### Loss
After 30 epochs, loss became about 0.05

![loss](./assets/loss.png)

### Nice results
These are pretty nice results. It seems like the network classify road area well.


![good1](./assets/good1.png)
![good2](./assets/good2.png)
![good3](./assets/good3.png)
![good4](./assets/good4.png)
![good5](./assets/good5.png)
![good6](./assets/good6.png)
![good7](./assets/good7.png)
![good8](./assets/good8.png)


### Bad results
These are bad results. I believe that the results will be better using the following methods.
- Use more deeper network (e.g. ResNet)
- Augment given data or train network with another data (e.g. CityScape)
- Use different architecture (e.g. [U-Net](https://arxiv.org/abs/1505.04597))
- Use post processing (e.g. [CRF(Conditional Random Field)](https://arxiv.org/abs/1210.5644))


![bad1](./assets/bad1.png)
![bad2](./assets/bad2.png)
![bad3](./assets/bad3.png)
![bad4](./assets/bad4.png)


## Setup
#### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
