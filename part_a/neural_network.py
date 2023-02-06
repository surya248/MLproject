from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        
    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        self.sigmoid = nn.Sigmoid()
        out = self.g(inputs)
        out = self.sigmoid(out)
        out = self.h(out)
        out = self.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_ep = []
    val_acc = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)+lamb*model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        
        val_acc.append(valid_acc)
        train_loss_ep.append(train_loss)

        if (epoch+1)%5==0:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                "Valid Acc: {}".format(epoch+1, train_loss, valid_acc))

    return val_acc,train_loss_ep
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10,50,100,200,500]
    train_loss = []
    val_acc = []
    val_acc_last = []

    print(f'Trying out different values of k and choosing the best among them')

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 40
    lamb = 0
    for ks in k:
        print(ks)
        model = AutoEncoder(1774,ks)
        a,b = train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
        
        val_acc.append(a)
        val_acc_last.append(a[-1])
        train_loss.append(b)

    ind = max(range(len(val_acc_last)), key=val_acc_last.__getitem__)
    print(f'Best Validation accuracy achieved for K={k[ind]}')
    model = AutoEncoder(1774,k[ind])
    train(model, lr, lamb, train_matrix, zero_train_matrix,
      valid_data, num_epoch)
    print('Test Accuracy'+str(evaluate(model, zero_train_matrix, test_data)))

    print(f'Plotting graph of Train Loss and Validation Accuracy for K={k[ind]}')

    plt.subplot(2,1,1)
    plt.plot(range(num_epoch),train_loss[ind])
    plt.xlabel('Epochs')
    plt.ylabel(f'Train Loss for K={k[ind]}')

    plt.subplot(2,1,2)
    plt.plot(range(num_epoch),val_acc[ind])
    plt.xlabel('Epochs')
    plt.ylabel(f'Validation Accuracy for K={k[ind]}')

    plt.title('Plot of train and val objectives for best K')
    plt.show()
    
    lamb = [0.001,0.01,0.1,1]
    lr = 0.025
    ind = 50
    num_epoch = 50
    train_loss = []
    val_acc = []
    for lambs in lamb:
        print(lambs)
        model = AutoEncoder(1774,ind)
        a,b = train(model, lr, lambs, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
        val_acc.append(a)
        val_acc_last.append(a[-1])
        train_loss.append(b)
    ind = max(range(len(val_acc_last)), key=val_acc_last.__getitem__)
    print(f'Best Validation accuracy achieved for Lamb={lamb[ind]}')
    train(model, lr, lamb[ind], train_matrix, zero_train_matrix,
      valid_data, num_epoch)
    print('Test Accuracy'+str(evaluate(model, zero_train_matrix, test_data)))


    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
