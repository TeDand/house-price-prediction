import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
import math as m
import random
import time
import os

class NN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, dropout = 0):
        super(NN, self).__init__()

        modules = []

        in_size = input_size
        for size in hidden_sizes:
            modules.append(nn.Linear(in_size, size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p = dropout))
            in_size = size

        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network.forward(x)

    def __call__(self, x):
        return self.forward(x)


class Regressor():

    def __init__(self, x, nb_epoch = 1000, lr = 0.1, hidden_sizes = [3], batches = 5, dropout = 0):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.NN = None
        self.Loss_Layer = nn.MSELoss()

        # Hyper parameters
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.batches = batches
        self.dropout = dropout
        return

        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        if training:
            self._one_hot = x.ocean_proximity.unique()

        x = x.copy()
        ocean = pd.get_dummies(x.ocean_proximity, prefix = 'ocean_proximity', columns = self._one_hot)
        x.drop(['ocean_proximity'], axis = 1, inplace = True)
        x = pd.concat([x, ocean], axis = 1)

        if training:
            self._avg = x.mean()
            self._min = x.min()
            self._max = x.max()
            self._width = self._max - self._min

        x = x.fillna(self._avg)
        x = x.apply(lambda x : (x - self._min) / self._width, axis = 1)

        return torch.tensor(x.to_numpy()), (torch.tensor(y.to_numpy()) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        X, Y = self._preprocessor(x, y, training = True) # Do not forget

        perms = torch.randperm(X.size()[0])
        X = X[perms]
        Y = Y[perms]
        
        # initialising network
        self.NN = NN(self.input_size, self.hidden_sizes, self.output_size, dropout=self.dropout)
        
        # loss layer and optimiser
        optimiser = optim.Adam(self.NN.parameters(), lr = self.lr)

        # train network
        batch_size = m.ceil(X.shape[0] / self.batches)
        X_b = torch.split(X, batch_size)
        Y_b = torch.split(Y, batch_size)
        for e in range(self.nb_epoch):
            for b in range(len(X_b)):
                optimiser.zero_grad()
                result = self.NN(X_b[b].float())
                loss = self.Loss_Layer(result, Y_b[b].float())

                # backward
                loss.backward()

                # gradient descent (weight updates)
                optimiser.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        result = (self.NN(X.float())).detach().numpy()
        return result

 
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y) # Do not forget
        result = self.NN(X.float())
        train_loss = float(self.Loss_Layer(result, Y.float()))
        train_loss = m.sqrt(train_loss)

        return train_loss # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def write_file(data, path): 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as target:
        target.write(data)

def save_regressor(trained_model, file="part2_model.pickle", log=True): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    
    dirpath = os.path.dirname(file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(file, 'wb') as target:
        pickle.dump(trained_model, target)

    if log:
        print("\nSaved model in %s\n" % file)


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y, tests = 10): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    
    n = m.ceil(x.shape[0] * 0.8)
    x_train = x[:n]
    y_train = y[:n]
    x_val = x[n:]
    y_val = y[n:]

    best_err = m.inf
    best_set = None

    nb_epoch = 1000
    lr = 0.0120
    batches = 842
    hidden_sizes = [29, 21, 34, 11]

    for dropout in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        st = time.time()
        regressor = Regressor(x_train, nb_epoch=nb_epoch, lr=lr, hidden_sizes=hidden_sizes, batches=batches, dropout=dropout)
        regressor.fit(x_train, y_train)
        error = regressor.score(x_val, y_val)
        dt = time.time()

        elapsed_time = dt - st
        output =  "Params: nb_epoch=%d, lr=%f, hidden=%s, batches=%d, dropout=%f" % (nb_epoch, lr, hidden_sizes, batches, dropout)
        output += "\nRMSE: %s" % error
        output += "\nTime: %dm %ds" % (elapsed_time // 60, elapsed_time % 60)

        if error < best_err:
            best_err = error
            best_set = nb_epoch, lr, hidden_sizes, batches, dropout
            print("\n\033[92m#################### New best ####################\033[0m", end = "")

            import socket
            save_regressor(regressor, file = "search/models/%s.pickle" % socket.gethostname(), log = False)
            write_file(output, "search/params/%s.txt" % socket.gethostname())
        
        print("\n%s" % output)

    return best_set


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    nb_epoch, lr, hidden_sizes, batches, dropout = RegressorHyperParameterSearch(x_train, y_train, tests = 1000)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=nb_epoch, lr=lr, hidden_sizes=hidden_sizes, batches=batches, dropout=dropout)

    print("Testing fit")
    regressor.fit(x_train, y_train)
    print("Fit tested")

    save_regressor(regressor)

    # Errors
    Error = regressor.score(x_train, y_train)
    print("Validation Error:", Error)
    

if __name__ == "__main__":
    example_main()

