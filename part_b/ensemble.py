# TODO: complete this file.

from utils import *
from scipy.linalg import sqrtm

import numpy as np
from part_b.item_responseB import *

from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
from part_a.matrix_factorization import *

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
def load_Q(num_questions):
    num_categories = 388
    path = "../data/question_meta.csv"
    Q = np.zeros((num_questions, num_categories))

    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                question_id = int(row[0])
                subjects = row[1]
                subjects = ast.literal_eval(subjects)
                for subject in subjects:
                    Q[question_id][int(subject)] = 1
            except ValueError:
                pass

    return Q

def run_irt(train_data, val_data):
    best_lr = 0.01
    best_iterations = 16

    size = len(train_data['user_id'])
    train_data_bootstrap = bootstrap_data(train_data, size)
    num_questions = len(set(train_data['question_id']))

    Q = load_Q(num_questions)


    U, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
        train_data_bootstrap, val_data, best_lr, best_iterations, Q)

    mat = np.zeros((len(U), len(beta)))

    for i in range(len(U)):
        for j in range(len(beta)):
            mat[i][j] = 0.25 + (1-0.25)*sigmoid(a[j]*(U[i].T@Q[j] - beta[j]))

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
        als(train_data_bootstrap, val_data, k=k, lr=0.01, num_iteration=500000)

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


    # mat_irt = run_irt(train_data, val_data)
    # mat_knn = run_knn(train_data, val_data, k=11)
    # mat_fact = run_fact(train_data, val_data, k=82)
    # mat_fact2 = run_fact(train_data, val_data, k=388)

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

    num_mats = 1
    ensemble = np.zeros((num_users, num_questions))
    for i in range(num_mats):
        mat = run_irt(train_data, val_data)
        ensemble += mat
    ensemble = ensemble / num_mats


    # ensemble = (mat_irt + mat_fact2 + mat_knn) / 3

    train_accuracy = sparse_matrix_evaluate(train_data, ensemble)
    valid_accuracy = sparse_matrix_evaluate(val_data, ensemble)

    print("train_accuracy {} \n valid accuracy {}".format(train_accuracy, valid_accuracy))
    print("test_accuracy {}".format(sparse_matrix_evaluate(test_data, ensemble)))

    #run Matrix factorization



    return 0

if __name__ == "__main__":
    main()
