import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        return nn.DotProduct(x, self.w)


    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1



    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    converged = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.hidden_dim1 = 128
        self.hidden_dim2 = 128
        self.lr = 0.5
        self.batch_size = 50 
        
        self.W1 = nn.Parameter(1, self.hidden_dim1)
        self.b1 = nn.Parameter(1, self.hidden_dim1)
        
        self.W2 = nn.Parameter(self.hidden_dim1, self.hidden_dim2)
        self.b2 = nn.Parameter(1, self.hidden_dim2)
        
        self.W3 = nn.Parameter(self.hidden_dim2, 1)
        self.b3 = nn.Parameter(1, 1)

        self.scale = nn.Constant(np.array([[1/(2*np.pi)]], dtype=np.float64))

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        
        
        normalized_x = nn.Linear(x, self.scale)  
        
        a1 = nn.Linear(normalized_x, self.W1)   
        z1 = nn.AddBias(a1, self.b1)
        h1 = nn.ReLU(z1)
        
        a2 = nn.Linear(h1, self.W2)           
        z2 = nn.AddBias(a2, self.b2)
        h2 = nn.ReLU(z2)
        
        a3 = nn.Linear(h2, self.W3)            
        y_pred = nn.AddBias(a3, self.b3)       
        
        return y_pred

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        y_pred = self.run(x)
        loss = nn.SquareLoss(y_pred, y)
        return loss

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        epoch = 0
        while True:
            epoch_loss = 0.0
            count = 0
            
            for x_batch, y_batch in dataset.iterate_once(self.batch_size):
                loss_node = self.get_loss(x_batch, y_batch)
                loss_value = nn.as_scalar(loss_node)
                
                epoch_loss += loss_value
                count += 1
                
                grads = nn.gradients(loss_node, [self.W1, self.b1,self.W2, self.b2,self.W3, self.b3])
                

                self.W1.update(grads[0], -self.lr)
                self.b1.update(grads[1], -self.lr)
                self.W2.update(grads[2], -self.lr)
                self.b2.update(grads[3], -self.lr)
                self.W3.update(grads[4], -self.lr)
                self.b3.update(grads[5], -self.lr)
            
            avg_loss = epoch_loss / count
            print(f"Epoch {epoch} | Loss: {avg_loss}")
            if avg_loss < 0.0002 :
                break
            epoch += 1


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.hidden_dim1 = 256   
        self.hidden_dim2 = 128   
        self.lr = 0.5            
        self.batch_size = 200     
        
        self.W1 = nn.Parameter(784, self.hidden_dim1)
        self.b1 = nn.Parameter(1, self.hidden_dim1)
        
        self.W2 = nn.Parameter(self.hidden_dim1, self.hidden_dim2)
        self.b2 = nn.Parameter(1, self.hidden_dim2)
        
        self.W3 = nn.Parameter(self.hidden_dim2, 10)
        self.b3 = nn.Parameter(1, 10)


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        a1 = nn.Linear(x, self.W1)       
        z1 = nn.AddBias(a1, self.b1)       
        h1 = nn.ReLU(z1)
        
        a2 = nn.Linear(h1, self.W2)      
        z2 = nn.AddBias(a2, self.b2)       
        h2 = nn.ReLU(z2)
        
        a3 = nn.Linear(h2, self.W3)       
        scores = nn.AddBias(a3, self.b3)   
        return scores
    
    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        scores = self.run(x)
        loss = nn.SoftmaxLoss(scores, y)
        return loss

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        epoch = 0
        while True:
            for x_batch, y_batch in dataset.iterate_once(self.batch_size):
                loss_node = self.get_loss(x_batch, y_batch)
                
                grads = nn.gradients(loss_node, [self.W1, self.b1,self.W2, self.b2,self.W3, self.b3])
                self.W1.update(grads[0], -self.lr)
                self.b1.update(grads[1], -self.lr)
                self.W2.update(grads[2], -self.lr)
                self.b2.update(grads[3], -self.lr)
                self.W3.update(grads[4], -self.lr)
                self.b3.update(grads[5], -self.lr)
            
            val_accuracy = dataset.get_validation_accuracy()
            if val_accuracy >= 0.97:
                break
            epoch += 1
