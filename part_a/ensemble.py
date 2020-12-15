# TODO: complete this file.

from utils import *
from scipy.linalg import sqrtm
import numpy as np
from part_a.item_response import *
from sklearn.impute import KNNImputer
from utils import *
from part_a.matrix_factorization import *



def bootstrap_data(data, size):
    """ Bootstrap data. Bag the data and return a bootstrapped data set

        :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
        :param size: int
        :return: A dictionary {user_id: list, question_id: list,
        is_correct: list}
        """
    data_bootstrap = {'user_id': [], "question_id": [], "is_correct": []}

    for num in range(size):
        i = \
            np.random.choice(len(data["question_id"]), 1)[0]

        c = data["is_correct"][i]
        n = data["user_id"][i]
        q = data["question_id"][i]

        data_bootstrap['user_id'].append(n)
        data_bootstrap['question_id'].append(q)
        data_bootstrap['is_correct'].append(c)

    return data_bootstrap

def run_irt(train_data, val_data):
    """Create matrix prediction using IRT trained on train data

    :param train_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :return: num_users by num_questions matrix of predictions
    """
    best_lr = 0.01
    best_iterations = 13

    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)

    theta, beta, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
        train_data_bootstrap, val_data, best_lr, best_iterations)

    mat = np.zeros((len(theta), len(beta)))

    for i in range(len(theta)):
        for j in range(len(beta)):
            mat[i][j] = np.exp(theta[i] - beta[j]) / (1 + np.exp(theta[i] - beta[j]))

    return mat


def run_knn(train_data, val_data, k):
    """Create matrix prediction using KNN trained on train data. k is k nearest neighbors

    :param train_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :param k: int
    :return: num_users by num_questions matrix of predictions
    """

    sparse_matrix = load_train_sparse("../data").toarray()
    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)

    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(sparse_matrix)

    return mat


def run_fact(train_data, val_data, k):
    """Create matrix prediction using Matrix factorization trained on train data.
        k is dimension of subspace

    :param train_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    :param k: int
    :return: num_users by num_questions matrix of predictions
    """
    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)

    mat, accuracy_train, accuracy_valids, error_train, error_valids = \
        als(train_data_bootstrap, val_data, k=k, lr=0.01, num_iteration=500000)

    return mat


def main():

    #get data
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #produce prediction matrices
    mat_irt = run_irt(train_data, val_data)
    mat_fact1 = run_fact(train_data, val_data, k=82)
    mat_fact2 = run_fact(train_data, val_data, k=388)

    #individual performance
    irt_accuracy_train = sparse_matrix_evaluate(train_data, mat_irt)
    fact1_accuracy_train = sparse_matrix_evaluate(train_data, mat_fact1)
    fact2_accuracy_train = sparse_matrix_evaluate(train_data, mat_fact2)

    irt_accuracy_valid = sparse_matrix_evaluate(val_data, mat_irt)
    fact1_accuracy_valid = sparse_matrix_evaluate(val_data, mat_fact1)
    fact2_accuracy_valid = sparse_matrix_evaluate(val_data, mat_fact2)

    print("Train Accuracy IRT: {}".format(irt_accuracy_train))
    print("Train Accuracy Fact82: {}".format(fact1_accuracy_train))
    print("Train Accuracy Fact388: {}".format(fact2_accuracy_train))
    print('\n')
    print("Valid Accuracy IRT: {}".format(irt_accuracy_valid))
    print("Valid Accuracy Fact82: {}".format(fact1_accuracy_valid))
    print("Valid Accuracy Fact388: {}".format(fact2_accuracy_valid))
    print('\n\n')


    #ensemble
    ensemble = (mat_irt + mat_fact1 + mat_fact2) / 3

    train_accuracy = sparse_matrix_evaluate(train_data, ensemble)
    valid_accuracy = sparse_matrix_evaluate(val_data, ensemble)

    print("train_accuracy {} \n valid accuracy {}".format(train_accuracy, valid_accuracy))
    print("test_accuracy {}".format(sparse_matrix_evaluate(test_data, ensemble)))

    return 0

if __name__ == "__main__":
    main()
