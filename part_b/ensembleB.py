# TODO: complete this file.

from utils import *
from scipy.linalg import sqrtm

import numpy as np
from part_b.item_responseB import *

from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
from part_b.matrix_factorizationB import *

g=0.25


def bootstrap_data(data, size):
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
    best_lr = 0.01
    best_iterations = 17

    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)

    theta, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
        train_data_bootstrap, val_data, best_lr, best_iterations)

    mat = np.zeros((len(theta), len(beta)))

    for i in range(len(theta)):
        for j in range(len(beta)):
            mat[i][j] = (((1-g)*np.exp(a[j]*(theta[i] - beta[j]))) / (1 + np.exp(a[j]*theta[i] - beta[j]))) + g

    return mat


def run_knn(train_data, val_data, k):
    sparse_matrix = load_train_sparse("../data").toarray()
    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)

    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(sparse_matrix)

    return mat


def run_fact(train_data, val_data, k):
    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)


    mat, accuracy_train, accuracy_valids, error_train, error_valids = \
        als(train_data_bootstrap, val_data, k=k, lr=0.05, num_iteration=500000, reg=0.001)

    return mat

def main():

    #get data
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    num_users = len(set(train_data['user_id']))
    num_questions = len(set(train_data['question_id']))




    # mat_knn = run_knn(train_data, val_data, k=11)
    # mat_fact1 = run_fact(train_data, val_data, k=388)
    # mat_fact2 = run_fact(train_data, val_data, k=388)
    # mat_fact3 = run_fact(train_data, val_data, k=388)
    # mat_fact4 = run_fact(train_data, val_data, k=388)
    # mat_fact5 = run_fact(train_data, val_data, k=388)
    # mat_fact6 = run_fact(train_data, val_data, k=388)
    # mat_fact7 = run_fact(train_data, val_data, k=388)
    # mat_fact8 = run_fact(train_data, val_data, k=388)
    # mat_fact9 = run_fact(train_data, val_data, k=388)
    # irt_accuracy_train = sparse_matrix_evaluate(train_data, mat_irt)
    # fact2_accuracy_train = sparse_matrix_evaluate(train_data, mat_fact2)
    # fact_accuracy_train = sparse_matrix_evaluate(train_data, mat_fact)
    #
    # irt_accuracy_valid = sparse_matrix_evaluate(val_data, mat_irt)
    # fact2_accuracy_valid = sparse_matrix_evaluate(val_data, mat_fact2)
    # fact_accuracy_valid = sparse_matrix_evaluate(val_data, mat_fact)

    # print("Irt train accuracy {}".format(irt_accuracy_train))
    # print("Irt valid accuracy {}".format(irt_accuracy_valid))
    # print('\n')
    #
    # print("MatFact2 train accuracy {}".format(fact2_accuracy_train))
    # print("MatFact2 valid accuracy {}".format(fact2_accuracy_valid))
    # print('\n')
    #
    # print("Fact train accuracy {}".format(fact_accuracy_train))
    # print("Fact valid accuracy {}".format(fact_accuracy_valid))
    # print('\n')
    num_mats = 5
    ensemble = np.zeros((num_users, num_questions))
    for i in range(num_mats):
        mat = run_irt(train_data, val_data)
        ensemble += mat

    ensemble = ensemble / num_mats
    train_accuracy = sparse_matrix_evaluate(train_data, ensemble)
    valid_accuracy = sparse_matrix_evaluate(val_data, ensemble)

    print("train_accuracy {} \n valid accuracy {}".format(train_accuracy, valid_accuracy))
    print("test_accuracy {}".format(sparse_matrix_evaluate(test_data, ensemble)))

    #run Matrix factorization



    return 0

if __name__ == "__main__":
    main()
