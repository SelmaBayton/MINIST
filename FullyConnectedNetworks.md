Fully Connected Networks, also known as Dense Networks or Multilayer Perceptrons (MLPs), are a type of artificial neural network where each neuron is connected to every neuron in the adjacent layers. Here's a brief mathematical explanation of Fully Connected Networks:

1. Neuron Activation:
In a fully connected layer, each neuron receives inputs from all the neurons in the previous layer. Let's denote the input to a neuron j in layer l as z_j^l. The activation of the neuron is computed as:

a_j^l = f(z_j^l)

where a_j^l is the activation value of neuron j in layer l, and f() is the activation function.

2. Weighted Sum:
The input to a neuron j in layer l, denoted as z_j^l, is the weighted sum of the activations from the previous layer, plus a bias term. The weighted sum can be expressed as:

z_j^l = ∑(w_ij^l * a_i^(l-1)) + b_j^l

where w_ij^l represents the weight between neuron i in layer (l-1) and neuron j in layer l, a_i^(l-1) is the activation of neuron i in the previous layer, and b_j^l is the bias term for neuron j in layer l.

3. Activation Function:
After computing the weighted sum, an activation function is applied to introduce non-linearity into the network. Common activation functions used in fully connected networks include sigmoid, tanh, and ReLU.

4. Forward Propagation:
The forward propagation process involves computing the activations of all neurons in each layer, starting from the input layer and moving forward through the network until the output layer is reached. This is done by iteratively applying the activation function to the weighted sum of each neuron.

5. Loss Function and Optimization:
Fully connected networks are typically trained using a loss function that quantifies the discrepancy between the predicted outputs and the ground truth labels. The choice of the loss function depends on the specific task, such as classification or regression. The network parameters, including the weights and biases, are optimized using an optimization algorithm like gradient descent.

6. Backpropagation:
During the training process, the gradients of the loss function with respect to the weights and biases are computed using backpropagation. The gradients are then used to update the network parameters in the opposite direction of the gradient, aiming to minimize the loss and improve the network's performance.

Fully connected networks are widely used in various machine learning tasks, including image classification, natural language processing, and regression problems. They can model complex relationships and capture non-linear patterns in the data. However, as the number of neurons and layers increases, fully connected networks can become computationally expensive and prone to overfitting. Regularization techniques, such as dropout and weight decay, are often employed to mitigate overfitting and improve generalization.

#

```python
!pip install git+https://github.com/williamedwardhahn/mpcr
from mpcr import *
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import pylab
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.utils

def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))

latent_size = 64
hidden_size = 256
image_size = 784
batch_size = 32

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255
X = (X - 0.5) / 0.5
X_test = (X_test - 0.5) / 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select GPU if available

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=device)

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=device)

X = GPU_data(X)
X_test = GPU_data(X_test)
Y = GPU_data(Y)
Y_test = GPU_data(Y_test)

X = (X + 1) / 2
X_test = (X_test + 1) / 2

def get_batch(mode):
    b = batch_size
    if mode == "train":
        r = np.random.randint(X.shape[0] - b)
        x = X[r:r+b, :, :, :]
        y = Y[r:r+b]
    elif mode == "test":
        r = np.random.randint(X_test.shape[0] - b)
        x = X_test[r:r+b, :, :, :]
        y = Y_test[r:r+b]
    return x, y

x, y = get_batch('train')
plt.hist(x.flatten().cpu().numpy())
montage_plot(x[0:25, 0, :, :].detach().cpu().numpy())

X = X.view(-1, 784)
X_test = X_test.view(-1, 784)

def get_batch(mode):
    b = batch_size
    if mode == "train":
        r = np.random.randint(X.shape[0] - b)
        x = X[r:r+b, :]
        y = Y[r:r+b]
    elif mode == "test":
        r = np.random.randint(X_test.shape[0] - b)
        x = X_test[r:r+b, :]
        y = Y_test[r:r+b]
    return x, y

X.shape, X_test.shape

def softmax(x):
    s1 = torch.exp(x - torch.max(x, 1)[0][:, None])
    s =

 s1 / s1.sum(1)[:, None]
    return s

def cross_entropy(outputs, labels):
    return -torch.sum(softmax(outputs).log()[range(outputs.size()[0]), labels.long()]) / outputs.size()[0]

def randn_trunc(s): #Truncated Normal Random Numbers
    mu = 0
    sigma = 0.1
    R = stats.truncnorm((-2 * sigma - mu) / sigma, (2 * sigma - mu) / sigma, loc=mu, scale=sigma)
    return R.rvs(s)

def acc(out, y):
    with torch.no_grad():
        return (torch.sum(torch.max(out, 1)[1] == y).item()) / y.shape[0]

def gradient_step(w):
    for j in range(len(w)):
        w[j].data = w[j].data - c.h * w[j].grad.data
        w[j].grad.data.zero_()

def make_plots():
    acc_train = acc(model(x, w), y)
    xt, yt = get_batch('test')
    acc_test = acc(model(xt, w), yt)
    wb.log({"acc_train": acc_train, "acc_test": acc_test})

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

def relu(x):
    return x * (x > 0)

def model(x, w):
    for j in range(len(w)):
        x = relu(x @ w[j])
    return x

def model(x, w):
    return relu(relu(x @ w[0]) @ w[1])

def model(x, w):
    x = x @ w[0]
    x = relu(x)
    x = x @ w[1]
    x = relu(x)
    return x

wb.init(project="Simple Fully Connected MNIST")
c = wb.config

c.h = 0.001
c.b = 100
c.layers = 2
c.epochs = 100000

c.f_n = [784, 500, 10]

w = [GPU(randn_trunc((c.f_n[i], c.f_n[i+1]))) for i in range(c.layers)]

for i in range(c.epochs):
    x, y = get_batch('train')
    loss = cross_entropy(softmax(model(x, w)), y)
    loss.backward()
    gradient_step(w)
    if (i+1) % 1 == 0:
        make_plots()

wb.init(project="Simple Fully Connected MNIST")
c = wb.config

c.h = 0.001
c.b = 1000
c.layers = 2
c.epochs = 100000

c.f_n = [784, 500, 10]

w = [GPU(randn_trunc((c.f_n[i], c.f_n[i+1]))) for i in range(c.layers)]

optimizer = torch.optim.Adam(w, lr=c.h)

for i in range(c.epochs):
    x, y = get_batch('train')
    loss = cross_entropy(softmax(model(x, w)), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % 1 == 0:
        make_plots()

acc(model(X, w), Y)
acc(model(X_test, w), Y_test)
model(X[0], w)
torch.argmax(model(X[0], w))
Y[0]

for i in range(len(w)):
    plt.imshow(w[i].cpu().detach().numpy())
    plt.show()
```

Explanation:

- The code installs the package `mpcr` using pip.
- The

 necessary libraries and modules are imported.
- Utility functions `plot` and `montage_plot` are defined for visualization purposes.
- Some constants for the model and dataset are set.
- MNIST dataset is downloaded and preprocessed.
- The data is normalized and converted to GPU tensors.
- Functions for getting batches of data and performing softmax, cross-entropy, random number generation, accuracy calculation, and gradient descent step are defined.
- The model architecture using fully connected layers and ReLU activation is defined.
- The weights of the model are initialized and the training loop begins.
- The Adam optimizer is used for training in the second part of the code.
- The model accuracy and predictions are calculated.
- The weights of the model are visualized.
