import mygrad as mg
from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.activations.relu import relu
from mygrad.nnet.layers import max_pool
from mynn.activations.sigmoid import sigmoid
from mynn.initializers.glorot_normal import glorot_normal
import numpy as np


def accuracy(pred, truth):
    """ Calculates the accuracy of the predicted sentiments.
    
    Parameters
    ----------
    pred: Union[numpy.ndarry, mygrad.Tensor]
        The prediction scores of sentiments of the tweets (as a float from 0 to 1)
    
    truth: numpy.ndarry
        The true tweet sentiment (0 or 1)
    
    Returns
    -------
    float
        The accuracy of the predictions
    """
    if isinstance(pred, mg.Tensor):
        pred = pred.data
    return np.mean(np.round(pred) == truth)
    
def binary_cross_entropy(y_pred, y_truth):
    """ Calculates the binary cross entropy loss for a given set of predictions.
    
    Parameters
    ----------
    y_pred: mg.Tensor, shape=
        The Tensor of class scores output from the model
    
    y_truth: mg.Tensor, shape=
        A constant Tensor or a NumPy array that contains the truth values for each prediction
    
    Returns
    -------
    mg.Tensor, shape=()
        A zero-dimensional tensor that is the loss
    """
    return -mg.mean(y_truth * mg.log(y_pred + 1e-08) + (1 - y_truth) * mg.log(1 - y_pred + 1e-08)) # <COGLINE>

class Model:
    def __init__(self):
        """ Initializes model layers and weights. """
        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(50, 250, 3, stride = 1, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
        self.dense1 = dense(250, 250, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
        self.dense2 = dense(250,1, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
    
    
    def __call__(self, x):
        """ Forward data through the network.
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D, S)
            The data to forward through the network.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, 1)
            The model outputs.
        
        Notes
        -----
        N = batch size
        D = embedding size
        S = sentence length
        """
        # (N, D, S) with D = 200 and S = 77
        x = self.conv1(x) # conv output shape (N, F, S') with F = 250 and S' = 75
        x = relu(x)
        x = max_pool(x, (x.shape[-1],), 1) # global pool output shape (N, F, S') with F = 250, S' = 1
        x = x.reshape(x.shape[0], -1) 
        # (N, F, 1) -> (N, F)
        x = self.dense1(x) # (N, F) @ (F, D1) = (N, D1)
        x = relu(x) 
        x = self.dense2(x)
        # (N, D1) @ (D1, 1) = (N, 1)
        x = sigmoid(x)
        return x # output shape (N, 1)
    
    
    @property
    def parameters(self, load = None):
        """ A convenience function for getting all the parameters of our model. """
        return self.conv1.parameters + self.dense1.parameters + self.dense2.parameters # <COGLINE>

