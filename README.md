# ResNet-demo
This is our demo to reproduce ResNet on pytorch, the dataset uses more than 4000 images in 5 categories.Here's [my blog](https://www.cnblogs.com/Hjxin02AIsharing-Wust/p/17541936.html) on some interpretations of the Resnet paper.The ResNet paper《[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)》.

The following figure illustrates the network structure of VGG, without residual connections, and with residual connections, from left to right:

![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet3.png)

This is the ResNet architecture with different layers, as you can see it is stacked from a number of residual blocks, unlike AlexNet, VGG and these, an Average pool is used instead of a fully connected layer, as well as a droput operation is not used, after which a fully connected layer of 1000 neurons is connected, and then finally a Softmax is attached:

![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet4.png)

In the figure below, the left graph is the connection structure of 18-layer and 34-layer residue blocks, while the right side is the residue structure of 50, 101 and 152. In order to reduce the computational complexity, the inputs come in to be first dimensionally downgraded and then dimensionally upgraded so that a 1x1 convolution is needed for dimensional conversion due to the different dimensionality of their inputs and outputs:

![image](https://github.com/Hjxin02AIsharing-Wust/ResNet-demo/blob/main/image_texture/ResNet5.png)






