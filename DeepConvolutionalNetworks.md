Deep Convolutional Networks (DCNs) are an extension of Convolutional Neural Networks (CNNs) that involve stacking multiple convolutional layers to learn complex hierarchical representations of input data. The mathematical explanation of deep convolutional networks builds upon the concepts of convolution, activation functions, pooling, and fully connected layers. Here's a brief overview:

1. Convolutional Operation:
The convolution operation in DCNs is similar to CNNs, where a filter/kernel is convolved with the input data. Let's denote the input feature map at layer l as X_l and the weights of the filters at layer l as W_l. The convolution operation can be defined as:

Y_l = X_l * W_l

where Y_l represents the output feature map at layer l.

2. Activation Function:
An activation function is applied element-wise to the output feature map of each convolutional layer. Common choices include ReLU (Rectified Linear Unit), sigmoid, and tanh. The activation function introduces non-linearity into the network, enabling it to learn complex relationships.

3. Pooling Operation:
Pooling is typically performed after convolutional layers in DCNs. Max pooling is commonly used, where a pooling window slides over the feature map and selects the maximum value within each window. The pooling operation reduces the spatial dimensions and captures the most salient features.

4. Stacking Convolutional Layers:
Deep convolutional networks involve stacking multiple convolutional layers one after another. The output feature map of one layer serves as the input to the next layer. This stacking allows the network to learn increasingly abstract and complex features as information flows deeper into the network.

5. Fully Connected Layers:
After several convolutional and pooling layers, the output feature maps are usually flattened and fed into one or more fully connected layers. These layers connect all neurons from the previous layer to the current layer, similar to a standard neural network. The fully connected layers enable learning of high-level representations and facilitate the final prediction or decision-making process.

6. Loss Function and Optimization:
DCNs are trained using a loss function that measures the discrepancy between the predicted outputs and the ground truth labels. The choice of the loss function depends on the specific task, such as classification or regression. The network parameters, including the filter weights, biases, and fully connected layer weights, are optimized using an optimization algorithm like gradient descent.

The depth of the network allows deep convolutional networks to learn hierarchical representations of the input data, capturing increasingly complex and abstract features. With deeper architectures, DCNs can effectively model intricate patterns, achieve higher accuracy, and extract more meaningful representations from input images or other spatial data. However, it's important to balance model complexity with the available data and computational resources to prevent overfitting and optimize performance.
