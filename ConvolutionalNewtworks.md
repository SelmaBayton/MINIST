Convolutional Neural Networks (CNNs) are a class of deep learning models commonly used for tasks involving images and spatial data. They leverage the concept of convolution, which involves sliding a small filter/kernel over the input data and performing element-wise multiplication and summation operations. Here's a brief mathematical explanation of convolutional networks:

1. Convolution Operation:
The convolution operation in CNNs is performed between an input image (or feature map) and a set of learnable filters. Let's denote the input image as X and the filters as W. The convolution operation can be defined as follows:

Y = X * W

where Y represents the output feature map, * denotes the convolution operation, and W acts as the filter/kernel.

2. Convolution Formula:
To compute the output feature map Y, the filter W is applied to different spatial locations of the input image X. The computation at each spatial location is given by:

Y[i, j] = sum(X[m, n] * W[i-m, j-n])

where Y[i, j] is the value at position (i, j) in the output feature map, X[m, n] represents the input values at position (m, n), and W[i-m, j-n] denotes the filter values.

3. Activation Function:
After the convolution operation, an activation function is typically applied element-wise to introduce non-linearity into the network. Common choices for activation functions in CNNs include ReLU (Rectified Linear Unit), sigmoid, and tanh.

4. Pooling Operation:
Pooling is often employed in CNNs to reduce the spatial dimensions of the feature maps and capture the most important information. The most commonly used pooling operation is max pooling, which selects the maximum value within a pooling window. The pooling operation helps in reducing the computational complexity and providing translational invariance.

5. Fully Connected Layers:
After several convolutional and pooling layers, the output is typically flattened and fed into one or more fully connected layers. These layers are similar to those in a standard neural network, connecting all neurons from the previous layer to the current layer.

6. Loss Function and Optimization:
CNNs are trained using a loss function that measures the discrepancy between the predicted outputs and the ground truth labels. The choice of the loss function depends on the specific task, such as classification (e.g., cross-entropy loss) or regression (e.g., mean squared error).

To train the CNN, an optimization algorithm, such as gradient descent, is employed to iteratively update the network parameters, including the filter weights, biases, and fully connected layer weights, based on the gradients of the loss function.

By leveraging the convolution operation, pooling, and activation functions, CNNs are capable of automatically learning hierarchical representations of the input data, capturing local patterns, and achieving state-of-the-art performance on various computer vision tasks, including image classification, object detection, and image segmentation.
#

```python
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
from imageio import *
import torch
from skimage.transform import resize
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from torchvision.models import *
from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from skimage.util import montage
import skimage.io as skio
from scipy import signal
from torchvision import transforms
import requests
from torchsummary import summary
```

The code imports various libraries and modules necessary for image processing, deep learning, and visualization.

```python
def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()
```

The `plot` function takes an image tensor or numpy array `x` and displays it using matplotlib. If `x` is a tensor, it is first converted to a numpy array. The image is displayed in grayscale and the axis is turned off.

```python
urllib.request.urlretrieve('https://raw.githubusercontent.com/imageio/imageio-binaries/master/images/imageio_banner.png', "image1.png")
im = Image.open("image1.png")
```

The code downloads an image from a given URL and saves it as "image1.png". Then, it opens the image using the `Image` module from PIL (Python Imaging Library).
