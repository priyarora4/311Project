from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


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


# def squared_error_loss(data, u, z):
#     """ Return the squared-error-loss given the data.
#     :param data: A dictionary {user_id: list, question_id: list,
#     is_correct: list}
#     :param u: 2D matrix
#     :param z: 2D matrix
#     :return: float
#     """
#     loss = 0
#     for i, q in enumerate(data["question_id"]):
#         loss += (data["is_correct"][i]
#                  - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
#     return 0.5 * loss

def ce_error_loss(data, u, z, reg):
    """ Return the cross entropy loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        # loss += (data["is_correct"][i]
        #          - np.sum(u[data["user_id"][i]] * z[q])) ** 2.

        ui = u[data['user_id'][i]]
        zj = z[q]
        cij = data["is_correct"][i]

        loss += np.log(1 + np.exp(ui.T @ zj)) - cij*(ui.T @ zj) + reg*(np.sum(np.square(ui)) + np.sum(np.square(zj)))

    num_samples = len(data['user_id'])
    return loss / num_samples


def update_u_z(train_data, lr, u, z, reg):
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

   # # d_dun = -(c - u[n].T @ z[q])*z[q]
   #  u[n] = u[n] + lr*(c - u[n].T @ z[q])*z[q]
   #
   #  #d_dzq = -(c - u[n].T @ z[q])*u[n]
   #
   #  z[q] = z[q] + lr*(c - u[n].T @ z[q])*u[n]

    dl_dun = sigmoid(u[n].T @ z[q])*z[q] - c*z[q] + reg*2*u[n]
    u[n] = u[n] - lr*dl_dun

    dl_dzq = sigmoid(u[n].T @ z[q]) * u[n] - c * u[n] + reg*2*z[q]
    z[q] = z[q] - lr * dl_dzq


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration, reg):
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
    accuracy_trains = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, reg)

        if i%10000 == 0:
            # error_valids.append(ce_error_loss(val_data, u, z, reg))
            # error_train.append(ce_error_loss(train_data, u, z, reg))
            mat = sigmoid(u @ z.T)
            accuracy_valids.append(sparse_matrix_evaluate(val_data, mat))
            accuracy_trains.append(sparse_matrix_evaluate(train_data, mat))
            print(i)

    mat = sigmoid(u @ z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, accuracy_trains, accuracy_valids, error_train, error_valids


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
    # max_k = 8
    # reconstruction = svd_reconstruct(train_matrix, max_k)
    # max_k_test_acc = sparse_matrix_evaluate(test_data, reconstruction)
    # max_k_val_acc = sparse_matrix_evaluate(val_data, reconstruction)
    # print("max k = {} | max test acc = {}".format(max_k, max_k_test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    num_iterations = 500000
    best_k = 388
    best_lr = 0.05
    #best_regs = 0.00025, 0.0005, 0.001
    # for reg in [0.0005, 0.001, 0.0025, 0.005]:
    #     for k in [2, 10, 50, 100, 150, 200, 300]:

    for lr in [0.001, 0.005, 0.01, 0.5]:

        mat, accuracy_trains, accuracy_valids, error_train, error_valids = \
            als(train_data, val_data, k=best_k, lr=lr, num_iteration=num_iterations, reg=0)

        # accuracy_train = sparse_matrix_evaluate(train_data, mat)
        # accuracy_valid = sparse_matrix_evaluate(val_data, mat)
        print("K = {}".format(best_k))
        print("lr = {}".format(best_lr))
        print("reg = {}".format(0))
        print("Training accuracy: {}".format(max(accuracy_trains)))
        print("Validation accuracy: {}".format(max(accuracy_valids)))
        best_iteration = (accuracy_trains.index(max(accuracy_trains))) * 10000
        print("highest overfit At iteration {}".format(best_iteration))
        print('\n\n')

        # plt.title("error vs iteration \n lr={}, k={} reg={}".format(lr, best_k, reg))
        # plt.plot(range(0, num_iterations, 10000), error_train)
        # plt.plot(range(0, num_iterations, 10000), error_valids)
        #
        # plt.xlabel('iteration')
        # plt.ylabel('error')
        #
        # plt.legend(['train', 'valid'])
        #
        # plt.show()

    best_k = 388
    best_lr = 0.05
    best_regs = 0.00025, 0.0005, 0.001


    # mat, accuracy_train, accuracy_valids, error_train, error_valids = \
    #              als(train_data, val_data, k=388, lr=lr, num_iteration=num_iterations)

    # num_samples_val = len(val_data['user_id'])
    # num_samples_train = len(train_data['user_id'])
    #
    # avg_error_train = np.asarray(error_train) / num_samples_train
    #
    # avg_error_val = np.asarray(error_valids) / num_samples_val
    #
    # plt.plot(range(0, num_iterations, 10000), avg_error_train)
    # plt.plot(range(0, num_iterations, 10000), avg_error_val)
    #
    # plt.ylabel('Squared Error')
    # plt.xlabel('Iteration')
    #
    # plt.legend(['Training error', 'Validation error'])
    #
    # plt.title('Average Squared Error vs Iteration \n K = {}, lr = {}, num_iterations = {}'.format(best_k, lr, num_iterations))
    #
    # plt.show()

    print("Test Accuracy: {}".format(sparse_matrix_evaluate(test_data, mat)))
    print("Valid Accuracy: {}".format(sparse_matrix_evaluate(val_data, mat)))




    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
