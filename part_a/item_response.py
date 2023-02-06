from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    #  Using Matrix notation to take advantage of vectorization
    p = 1/(1+np.exp(-(theta[...,None] - beta[None,...])))
    loss = - np.sum(np.nan_to_num(data*np.log(p) + (1-data)*np.log(1-p)))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return loss


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    p = 1/(1+np.exp(-(theta[...,None] - beta[None,...])))
    non_nan = np.invert(np.isnan(data)).astype(int)

    grad_theta = -(np.sum(np.nan_to_num(data),axis=1) - np.sum(non_nan*p,axis=1))
    grad_beta = (np.sum(np.nan_to_num(data),axis=0) - np.sum(non_nan*p,axis=0))

    theta = theta - lr*grad_theta
    beta = beta - lr*grad_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    students = data.shape[0]
    items = data.shape[1]

    theta = np.random.randn(students)
    beta = np.random.randn(items)

    # theta = np.zeros(students)
    # beta = np.zeros(items)

    val_acc_lst = []
    loss = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        loss.append(neg_lld)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst,loss


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta,beta,val_acc,loss = irt(sparse_matrix,val_data,lr=0.008,iterations=40)

    plt.subplot(2,1,1)
    plt.plot(range(len(val_acc)),val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(range(len(val_acc)),loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

    test_acc = evaluate(test_data,theta,beta)
    val_acc.append(evaluate(val_data,theta,beta))

    print(f'\nFinal Validation accuracy is {val_acc[-1]}')
    print(f'Final Test Accuracy is {test_acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    ques = np.random.choice(sparse_matrix.shape[1],3)
    
    # For better visulization
    step = 8

    for i in ques:
        prob = []
        for j in range(0,sparse_matrix.shape[0],step):
            p = sigmoid(theta[j]-beta[i])
            prob.append(p)
        plt.plot(range(0,sparse_matrix.shape[0],step),prob)
    plt.xlabel('Students')
    plt.ylabel('Probablity of correct answer')
    # plt.title('Variation of probablity of correct answers over all students for 3 random questions')
    plt.legend([f'Question {ques[0]}',f'Question {ques[1]}',f'Question {ques[2]}'])
    

    print(f'\nThe figure highlights the fact that all the three graphs overlap a lot ',
        'which means that there are some highly talented students who can answer most of the questions ',
        'irrespective of the questions asked. While for some students all the questions seems hard')

    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
