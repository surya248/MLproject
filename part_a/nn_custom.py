from utils import *
import utils_partb as utb

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

def load_data(base_path="/home/mohit/ml_project/data"):
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
    def __init__(self,num_question, limits = [],
                 mtype='default', k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        self.mtype = mtype
        self.limits = limits
        self.num = num_question
        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.f = nn.Linear(k,k)
        self.h = nn.Linear(k, num_question)
        
        if self.mtype=='one-hot':
            self.h = nn.Linear(k, 1)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        f_w_norm = torch.norm(self.f.weight, 2) ** 2
        return g_w_norm + h_w_norm + f_w_norm

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

        if self.mtype == 'one-hot':
            tmp_user = torch.nn.functional.one_hot(inputs[:,0].to(torch.int64),num_classes=self.limits[1])
            tmp_ques = torch.nn.functional.one_hot(inputs[:,1].to(torch.int64),num_classes=self.limits[0])
            inputs = torch.cat((tmp_user,tmp_ques),dim=1)
            inputs = inputs.to(torch.float32)
            
        self.sigmoid = nn.Sigmoid()
        out = self.g(inputs)
        out = self.sigmoid(out)
        out = self.f(out)
        out = self.sigmoid(out)
        out = self.h(out)

        if self.mtype == 'defualt':
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


#########################################################
# CUSTOM TRAINING LOOP AND MODEL FOR PART B #############
#########################################################
def train_onehot(base_path="/home/mohit/ml_project/data"):
    train_data = utb.load_3D_from_csv(base_path,dtype='train')
    val_data = utb.load_3D_from_csv(base_path,dtype='val')
    test_data = utb.load_3D_from_csv(base_path,dtype='test')

    max_user = max(np.concatenate(
        (
            train_data[:,0],
            val_data[:,0],
            test_data[:,0]
        )
    )) + 1

    max_ques = max(np.concatenate(
        (
            train_data[:,1],
            val_data[:,1],
            test_data[:,1]
        )
    )) + 1

    model = AutoEncoder(max_ques+max_user,[max_ques,max_user],mtype='one-hot',k=50)

    epoch = 20
    lr = 0.005
    lamb = 0.00005
    batch_size = 128
    n = len(train_data)
    end_ind = n//batch_size

    BCEloss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)

    train_loss_ep = []
    val_acc = []

    for i in range(epoch):
        train_loss = 0

        for ind in range(end_ind):

            inputs = Variable(torch.Tensor(train_data[ind*batch_size:(ind+1)*batch_size,:2]))
            target = Variable(torch.Tensor(train_data[ind*batch_size:(ind+1)*batch_size,2]))

            optimizer.zero_grad()

            output = model(inputs)


            loss = BCEloss(output,target[...,None].to(torch.float64))
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate_onehot(model,val_data)
        train_acc = evaluate_onehot(model,train_data)

        val_acc.append(valid_acc)
        train_loss_ep.append(train_loss)

        if (i+1)%1==0:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                "Training Acc: {} \tValid Acc: {} , ".format(i+1, train_loss,train_acc, valid_acc))
            
        print("Test Accuracy "+str(evaluate_onehot(model,test_data)))

    plt.subplot(2,1,1)
    plt.plot(range(len(val_acc)),val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(range(len(train_loss_ep)),train_loss_ep)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()


def evaluate_onehot(model,data):
    inputs = Variable(torch.Tensor(data[:,:2]))
    
    targets = Variable(torch.Tensor(data[:,2]))
    targets = targets[...,None]
    outputs = model(inputs)

    outputs = torch.sigmoid(outputs)

    outputs = (outputs>0.5).to(torch.int64)
    return (sum(outputs==targets)/float(len(targets))).detach().numpy()[0]
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    train_onehot()
