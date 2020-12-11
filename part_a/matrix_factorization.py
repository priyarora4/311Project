from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

   # d_dun = -(c - u[n].T @ z[q])*z[q]
    u[n] = u[n] + lr*(c - u[n].T @ z[q])*z[q]

    #d_dzq = -(c - u[n].T @ z[q])*u[n]

    z[q] = z[q] + lr*(c - u[n].T @ z[q])*u[n]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    accuracy_valids = []
    error_valids = []
    error_train = []
    accuracy_train = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        # accuracy_valids.append(sparse_matrix_evaluate(val_data, mat))
        # accuracy_train.append(sparse_matrix_evaluate(train_data, mat))
        if i%10000 == 0:
            error_valids.append(squared_error_loss(val_data, u, z))
            error_train.append(squared_error_loss(train_data, u, z))

    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, accuracy_train, accuracy_valids, error_train, error_valids


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # accuracies = []
    # test_k_range = range(1, 500)
    # for k in test_k_range:
    #     reconstruction = svd_reconstruct(train_matrix, k)
    #     acc = sparse_matrix_evaluate(val_data, reconstruction)
    #     print("k={} | acc={}".format(k, acc))
    #     accuracies.append(acc)
    # max_acc = max(accuracies)
    # max_k = accuracies.index(max_acc)
    # print("max k = {} | max val acc = {}".format(max_k, max_acc))
    max_k = 8
    reconstruction = svd_reconstruct(train_matrix, max_k)
    max_k_test_acc = sparse_matrix_evaluate(test_data, reconstruction)
    print("max k = {} | max test acc = {}".format(max_k, max_k_test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    num_iterations = 500000
    lr = 0.01
    for k in range(2, 103, 20):
        mat, accuracy_train, accuracy_valids, error_train, error_valids = \
            als(train_data, val_data, k, lr=lr, num_iteration=num_iterations)

        accuracy_train = sparse_matrix_evaluate(train_data, mat)
        accuracy_valid = sparse_matrix_evaluate(val_data, mat)
        print("K = {}".format(k))
        print("Training accuracy: {}".format(accuracy_train))
        print("Validation accuracy: {}".format(accuracy_valid))
        print('\n\n')

        plt.plot(range(0, num_iterations, 10000), error_train)
        plt.plot(range(0, num_iterations, 10000), error_valids)

        plt.ylabel('Error')
        plt.xlabel('Iteration')

        plt.legend(['Training error', 'Validation error'])

        plt.title('K = {}, lr = {}, num_iterations = {}'.format(k, lr, num_iterations))

        plt.show()




    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
