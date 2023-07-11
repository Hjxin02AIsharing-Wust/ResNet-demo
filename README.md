# ResNet-demo
This is our demo to reproduce ResNet on pytorch, the dataset uses more than 4000 images in 5 categories.Here's [my blog](https://www.cnblogs.com/Hjxin02AIsharing-Wust/p/17541936.html) on some interpretations of the Resnet paper.The ResNet paper《[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)》.

The following figure illustrates the network structure of VGG, without residual connections, and with residual connections, from left to right:


![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet3.png)


This is the ResNet architecture with different layers, as you can see it is stacked from a number of residual blocks, unlike AlexNet, VGG and these, an Average pool is used instead of a fully connected layer, as well as a droput operation is not used, after which a fully connected layer of 1000 neurons is connected, and then finally a Softmax is attached:


![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet4.png)


In the figure below, the left graph is the connection structure of 18-layer and 34-layer residue blocks, while the right side is the residue structure of 50, 101 and 152. In order to reduce the computational complexity, the inputs come in to be first dimensionally downgraded and then dimensionally upgraded so that a 1x1 convolution is needed for dimensional conversion due to the different dimensionality of their inputs and outputs:


![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet5.png)


The following table shows the comparison of ResNet with different network depths, we can see that the accuracy improves as the number of network layers increases, indicating that the residual block structure is effective against the degradation phenomenon in deep neural networks.


![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet7.png)


## Dataset

The same dataset as [AlexNet-demo](https://github.com/Hjxin02AIsharing-Wust/AlexNet-demo) and [VGG-demo](https://github.com/Hjxin02AIsharing-Wust/VGG-demo) are used.The dataset uses more than 4000 images in 5 categories, you can download it [here](https://drive.google.com/drive/folders/1z2d7UejBR55QY8dc2GOmSkyfi8C-vUBs).

## Data Preprocess
You need to change the `root_file` parameter in the  `Data_Preprocess.py` file to the address of the dataset you downloaded. We follow the training set: validation set ratio of 9 to 1. You can also change this ratio, just change the `split_rate parameter`. Also we follow the data enhancement operation of the VGG paper, cropping the image to 224x224 at random and flipping it horizontally.

## Usage

### Train
You can use the following commands to train the model. We built the network structure of ResNet18, ResNet34, ResNet50, ResNet101 and ResNet152 in `Net.py`, and you just need to change the corresponding network names in `train.py`：
```shell
python train.py 
```
Here are some of our training settings: batchsize is set to 128, cross-entropy loss function is used, Adam is used for the optimizer, learning rate is set to 0.0002, and 20 epochs are trained.

## Test

We provide the test code：
```shell
python test.py
```
You can use our provided model weights [`ResNet.pth`](https://drive.google.com/drive/folders/1byR1pjxu5oh29bfaDjeq6VbtJCqVe2i7) and test image `roseflower.png`, of course, you can also use other image.









