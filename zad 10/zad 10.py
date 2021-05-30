import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD
from torch.nn import BCELoss
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# model przewidujący czy mieszkanie znajduje się w centrum czy nie na podstawie danych z gratkapl-centrenrm.csv
# przy tworzeniu go korzysałem z tutoriala:
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
# oraz z Pańskich ćwiczeń
class Net(nn.Module):
    """W PyTorchu tworzenie sieci neuronowej
    polega na zdefiniowaniu klasy, która dziedziczy z nn.Module.
    """
    def __init__(self):
        super().__init__()

        # Warstwy liniowe
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        """Definiujemy przechodzenie "do przodu" jako kolejne przekształcenia wejścia x"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=0)
        # getting rid of outlier observations
        df = df.loc[(df['Price'] > 1000) & (df['Price'] < 10 ** 7) & (df['SqrMeters'] < 185)]
        # store the inputs and outputs
        self.X = df.values[:, :-3]
        self.y = df.values[:, -2]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


if __name__ == "__main__":
    train, test = prepare_data('gratkapl-centrenrm.csv')
    model = Net()
    train_model(train, model)
    # evaluate the model
    acc = evaluate_model(test, model)
    print('Accuracy: %.3f' % acc)
