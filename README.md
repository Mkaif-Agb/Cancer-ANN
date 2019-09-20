# Cancer-ANN
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

# Neural Network
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.[1] Thus a neural network is either a biological neural network, made up of real biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

![Image of a Neural Network](https://miro.medium.com/max/1592/1*yGMk1GSKKbyKr_cMarlWnA.jpeg)


# Artificial neural network
Artificial neural networks (ANN) or connectionist systems are computing systems that are inspired by, but not identical to, biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge of cats, for example, that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the examples that they process.

![How an Artificial Neural Network Works](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/07/Introduction-to-Artificial-Neural-Networks.jpg)

Every Artificial Neural Network should have its value standardized to improve accuracy on the dataset this can be done easily by using a simple function from SKlearn Library
# Standardize
A standard approach is to scale the inputs to have mean 0 and a variance of 1. Also linear decorrelation/whitening/pca helps a lot.

# Dense Layer

A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a. The following is te docstring of class Dense from the keras documentation:
output = activation(dot(input, kernel) + bias)where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

# Dropout Layer 

Dropout is a a technique used to tackle Overfitting . The Dropout method in keras.layers module takes in a float between 0 and 1, which is the fraction of the neurons to drop. Below is the docstring of the Dropout method from the documentation:
Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

# Compile

Every Neural Network should be compiled before Training it on a Dataset. During Compilation wer have to provide our neural network with an optimizer, a loss function as well as the Metrics that we need to observe during Training

## Adam
Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

## Binary Cross Entropy
Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

# Using This ANN we got an Accuracy of around 98% on our test set which is very good we can tweak it a little and can improve Accuracy much more. Some of the Plots taken from this project are given below 

![Correlation](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/Correlation.png)
![Elbow Method](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/Elbow_Method.png)
![Heatmap](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/Heatmap.png)
![JointPlot](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/JointPlot.png)
![Pairplot](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/Pairplot.png)
![Prediction_Confusion_Matrix](https://github.com/Mkaif-Agb/Cancer-ANN/blob/master/Images/Prediction_Confusion_Matrix.png)
