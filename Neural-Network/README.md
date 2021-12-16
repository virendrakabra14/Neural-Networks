## Cat Classifier based on Deep Neural Network

A four layer model has been used. For all layers except the last, ReLU activation is used, while sigmoid activation is used for the last layer.<br>

<pre>
ReLU (Rectified Linear Unit) : ReLU(z) = max(0,z)
Sigmoid : σ(z) = 1 / ( 1 + exp(-z) )
</pre>

This deep neural network gives a higher accuracy than the logistic regression model (an improvement of 10% on the test set).

Please see <a href="./NN-cat.ipynb">NN-cat.ipynb</a> for further details.