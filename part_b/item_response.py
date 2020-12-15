from utils import *

import numpy as np
import matplotlib.pyplot as plt
import ast

g = 0.25


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, U, beta, a, Q):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    //notice using sparse matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    num_samples = len(data['user_id'])

    for n in range(num_samples):
        c_ij = data['is_correct'][n]
        beta_j = beta[data['question_id'][n]]
        u_i = U[data['user_id'][n]].reshape((388, 1))
        a_j = a[data['question_id'][n]]
        q_j = Q[data['question_id'][n]]

        log_lklihood += c_ij * np.log(np.exp(a_j*(u_i.T@q_j - beta_j)) + g) - \
            np.log(np.exp(a_j*(u_i.T@q_j - beta_j)) + 1) - (c_ij-1)*np.log(1-g)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_U_beta_a(data, lr, U, beta, a, Q):
    """ Update U and beta and a using gradient descent.


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
    num_samples = len(data['user_id'])
    u_sums = np.zeros(U.shape)
    beta_sums = np.zeros(beta.shape)
    a_sums = np.zeros(a.shape)


    for n in range(num_samples):
        c_ij = data['is_correct'][n]
        beta_j = beta[data['question_id'][n]]
        u_i = U[data['user_id'][n]].reshape((388, 1))
        a_j = a[data['question_id'][n]]
        q_j = Q[data['question_id'][n]].reshape((388, 1))
        i = data['user_id'][n]
        j = data['question_id'][n]

        e_aj_qu = np.exp(a_j*u_i.T@q_j)
        e_aj_bj = np.exp(a_j*beta_j)

        sub_gradient = (a_j*q_j*e_aj_qu) * (c_ij/(g*e_aj_bj + e_aj_qu) - 1/(e_aj_bj + e_aj_qu))

        u_sums[i] += sub_gradient.reshape((388,))

    U += lr*u_sums

    for n in range(num_samples):
        c_ij = data['is_correct'][n]
        beta_j = beta[data['question_id'][n]]
        u_i = U[data['user_id'][n]].reshape((388, 1))
        a_j = a[data['question_id'][n]]
        q_j = Q[data['question_id'][n]].reshape((388, 1))
        i = data['user_id'][n]
        j = data['question_id'][n]

        e_aj_qu = np.exp(a_j * u_i.T @ q_j)
        e_aj_bj = np.exp(a_j * beta_j)

        sub_gradient = e_aj_qu*(u_i.T@q_j - beta_j) * (c_ij/(g*e_aj_bj + e_aj_qu) - 1/(e_aj_bj + e_aj_qu))

        a_sums[j] += sub_gradient.reshape((1,))

    a += lr*a_sums

    for n in range(num_samples):
        c_ij = data['is_correct'][n]
        beta_j = beta[data['question_id'][n]]
        u_i = U[data['user_id'][n]].reshape((388, 1))
        a_j = a[data['question_id'][n]]
        q_j = Q[data['question_id'][n]].reshape((388, 1))
        i = data['user_id'][n]
        j = data['question_id'][n]

        e_aj_qu = np.exp(a_j * u_i.T @ q_j)
        e_aj_bj = np.exp(a_j * beta_j)

        sub_gradient = ((a_j*c_ij*g*e_aj_bj) / (g*e_aj_bj + e_aj_qu)) - ((a_j*e_aj_bj)/(e_aj_bj + e_aj_qu)) + a_j*(1-c_ij)

        beta_sums[j] += sub_gradient.reshape((1,))

    beta += lr*beta_sums

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return U, beta, a


def irt(data, val_data, lr, iterations, Q):
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
    num_users = len(set(data['user_id']))
    num_questions = len(set(data['question_id']))
    # theta = np.random.rand(num_users, 1)
    # beta = np.random.rand(num_questions, 1)
    U = np.zeros((num_users, 388))
    beta = np.zeros((num_questions, 1))
    a = np.ones((num_questions, 1))


    val_acc_lst = []
    neg_lld_list_train = []
    neg_lld_list_valid = []
    for i in range(iterations):
        #neg_lld = neg_log_likelihood(data, U=U, beta=beta, a=a, Q=Q)
        # neg_lld_list_train.append(neg_lld)
        #
        # neg_lld_valid = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        # neg_lld_list_valid.append(neg_lld_valid)

        score = evaluate(data=val_data, U=U, beta=beta, a=a, Q=Q)
        val_acc_lst.append(score)
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))
        U, beta, a = update_U_beta_a(data, lr, U=U, beta=beta, a=a, Q=Q)
        print(i)

    # TODO: You may change the return values to achieve what you want.
    return U, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid


def evaluate(data, U, beta, a, Q):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    wrongs = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        u_u = U[u].reshape((388,1))
        q_q = Q[q].reshape((388,1))
        x = a[q]*(u_u.T @ q_q - beta[q])
        p_a = g + (1-g)*sigmoid(x)
        pred.append(p_a[0][0] >= 0.5)
        # if data["is_correct"][i] != (p_a[0][0] >= 0.5):
        #     wrongs.append((data["is_correct"][i], p_a[0][0]))

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


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


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    private_test = load_private_test_csv('../data')


    train_data['question_id'] += val_data['question_id']
    train_data['user_id'] += val_data['user_id']
    train_data['is_correct'] += val_data['is_correct']

    train_data['question_id'] += test_data['question_id']
    train_data['user_id'] += test_data['user_id']
    train_data['is_correct'] += test_data['is_correct']




    num_questions = len(set(train_data['question_id']))
    Q = load_Q(num_questions)



    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # lr = 0.01
    # iterations = 100
    # learning_rates = [0.001, 0.005, 0.01, 0.05]
    # for lr in learning_rates:
    #     if lr == 0.01 or lr == 0.05:
    #         iterations = 50
    #     best_accuracies_per_lr = []
    #
    #     U, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(train_data, val_data, lr, iterations, Q)
    #
    #     max_accuracy = max(val_acc_lst)
    #
    #     print("Current Learning Rate: {}".format(lr))
    #     print("Best Validation Accuracy: {}".format(max_accuracy))
    #     print("At iteration: {}".format(val_acc_lst.index(max_accuracy)))
    #     print("Training Accuracy {}".format(evaluate(train_data, U, beta, a, Q)))
    #     print('\n\n')

        # plt.plot(range(1, iterations + 1), neg_lld_list_train)
        # plt.plot(range(1, iterations + 1), neg_lld_list_valid)
        # plt.xlabel("iteration")
        # plt.ylabel("NLLK")

        # plt.legend(['Train', 'Validation'])

        # plt.show()

        # best_accuracies_per_lr.append(max_accuracy)


    best_lr = 0.01
    best_iterations = 16
    #
    #
    #
    U, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
        train_data, val_data, best_lr, best_iterations, Q)


    mat = np.zeros((len(U), len(beta)))
    for i in range(len(U)):
        for j in range(len(beta)):
            mat[i][j] = g + (1-g)*sigmoid(a[j]*(U[i].T@Q[j] - beta[j]))

    predictions = sparse_matrix_predictions(private_test, mat)
    private_test['is_correct'] = predictions
    save_private_test_csv(private_test)

    score = evaluate(test_data, U, beta, a, Q)
    score_valid = evaluate(val_data, U, beta, a, Q)
    score_train = evaluate(train_data, U, beta, a, Q)
    #
    # wrong1count = 0
    # wrong0count = 0
    # for i in range(len(wrongs_val)):
    #     if wrongs_val[i][0] == 1:
    #         wrong1count += 1
    #
    #     elif wrongs_val[i][0] == 0:
    #         wrong0count += 1
    # print("Prediction wrong Correct is 1: {}".format(wrong1count))
    # print("Prediction wrong Correct is 0: {}".format(wrong0count))
    # # #
    # # #
    #
    print("Test accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score))

    print("Validation accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score_valid))

    print("train accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score_train))

    # num_samples_val = len(val_data['user_id'])
    # num_samples_train = len(train_data['user_id'])
    #
    # avg_log_like_train = np.asarray(neg_lld_list_train) / num_samples_train
    #
    # avg_log_like_val = np.asarray(neg_lld_list_valid) / num_samples_val
    #
    # plt.plot(range(1, best_iterations+1), avg_log_like_train)
    # plt.plot(range(1, best_iterations+1), avg_log_like_val)
    # plt.xlabel("Iteration")
    # plt.ylabel("NLLK")
    # plt.title("IRT Average NLLK vs Iteration, learning rate={}".format(best_lr))
    #
    # plt.legend(['Train', 'Validation'])
    #
    # plt.show()

    # score = evaluate(test_data, theta, beta)
    # print("test accuracy {}".format(score))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    #TODO REPORT TEST ACCURACIES

    # theta, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
    #     train_data, val_data, best_lr, best_iterations)
    #
    # five_questions = []
    # index = 0
    # while len(five_questions) != 5:
    #     if train_data['question_id'][index] not in five_questions:
    #         five_questions.append(train_data['question_id'][index])
    #     index+=1
    #
    #
    # for question_id in five_questions:
    #     probs = []
    #     for i in range(len(theta)):
    #         probs.append(sigmoid(theta[i] - beta[question_id]))
    #
    #     plt.plot(theta, probs, 'r.')
    #     plt.xlabel('Theta')
    #     plt.ylabel('Probability correct')
    #     plt.title("IRT Probability Correct vs Theta, Question Id: {}".format(question_id))
    #     plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
