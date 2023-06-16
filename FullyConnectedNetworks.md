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
