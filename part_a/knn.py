from sklearn.impute import KNNImputer
from utils import *
import tqdm


def knn_impute_by_user(matrix, valid_data,
                     k, weights='uniform', add_indicator=False):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k,weights=weights,add_indicator=add_indicator)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data,
                        k, weights='uniform',add_indicator=False):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    ##################################################################### 
    nbrs = KNNImputer(n_neighbors=k,weights=weights,add_indicator=add_indicator)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    # print("Validation Accuracy: {}".format(acc))
    return acc

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("Sparse matrix:")
    # print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k = [1, 6, 11, 16, 21, 26]
    weights='uniform'
    add_indicator = False

    acc = []
    for i in k:
        acc.append(knn_impute_by_user(sparse_matrix,val_data,i,weights,add_indicator))

    k_opt = k[max(range(len(acc)), key=acc.__getitem__)]
    test_acc = knn_impute_by_user(sparse_matrix,test_data,k_opt,weights,add_indicator)
    print(f'Best Test Accuracy based on colloborative user filtering : {test_acc*100:.3f} for k = {k_opt}')


    acc_item = []
    for i in k:
        acc_item.append(knn_impute_by_item(sparse_matrix,val_data,i, weights,add_indicator))

    k_opt_item = k[max(range(len(acc)), key=acc.__getitem__)]
    test_acc_item = knn_impute_by_item(sparse_matrix,test_data,k_opt_item, weights,add_indicator)
    print(f'Best Test Accuracy based on colloborative item filtering: {test_acc_item*100:.3f} for k = {k_opt_item}')

    if test_acc>test_acc_item:
        print(f'\nUser based collaborative filtering performed better than test based colloborative filtering')
        print('This is mainly due to fact that it is more easy',
            'to group data by users rather than items, because dufferent students',
            'will have different strength and user based filtering captures on this fact while item based filtering',
            'only take into account the difficulty of the questions')



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
