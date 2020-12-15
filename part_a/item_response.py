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

    for i in range(num_samples):
        beta_j = beta[data['question_id'][i]]
        theta_i = theta[data['user_id'][i]]
        is_correct = data['is_correct'][i]
        log_lklihood += is_correct * (theta_i - beta_j) - np.log(
            1 + np.exp(theta_i - beta_j))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


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
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_samples = len(data['user_id'])
    theta_sums = np.zeros(theta.shape)
    beta_sums = np.zeros(beta.shape)

    for i in range(num_samples):

        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        beta_j = beta[question_id]
        theta_i = theta[user_id]
        is_correct = data['is_correct'][i]

        # updating theta

        sub_gradient_theta = is_correct - (
                    (np.exp(theta_i)) / (np.exp(beta_j) + np.exp(theta_i)))
        theta_sums[user_id] += sub_gradient_theta

    theta = theta + lr * theta_sums

    for i in range(num_samples):

        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        beta_j = beta[question_id]
        theta_i = theta[user_id]
        is_correct = data['is_correct'][i]

        # updating beta

        sub_gradient_beta = ((np.exp(theta_i)) / (
                    np.exp(beta_j) + np.exp(theta_i))) - is_correct
        beta_sums[question_id] += sub_gradient_beta

    beta = beta + lr * beta_sums

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
    num_users = len(set(data['user_id']))
    num_questions = len(set(data['question_id']))
    theta = np.zeros((num_users, 1))
    beta = np.zeros((num_questions, 1))

    val_acc_lst = []
    neg_lld_list_train = []
    neg_lld_list_valid = []
    for i in range(iterations):
        # FOR PLOTTING REASONS
        # neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        # neg_lld_list_train.append(neg_lld)
        # neg_lld_valid = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        # neg_lld_list_valid.append(neg_lld_valid)

        # score = evaluate(data=val_data, theta=theta, beta=beta)
        # val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_list_train, neg_lld_list_valid


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
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    """Tuning learning rate and number of iterations 
    """

    # lr = 0.01
    # iterations = 100
    # learning_rates = [0.001, 0.005, 0.01, 0.05]
    # for lr in learning_rates:
    #     if lr == 0.01 or lr == 0.05:
    #         iterations = 50
    #     best_accuracies_per_lr = []
    #
    #     theta, beta, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(train_data, val_data, lr, iterations)
    #
    #     max_accuracy = max(val_acc_lst)
    #
    #     print("Current Learning Rate: {}".format(lr))
    #     print("Best Validation Accuracy: {}".format(max_accuracy))
    #     print("At iteration: {}".format(val_acc_lst.index(max_accuracy)))
    #     print('\n\n')

    """Plotting negative log likelihood vs iteration
    """

    # # plt.plot(range(1, iterations + 1), neg_lld_list_train)
    # plt.plot(range(1, iterations + 1), neg_lld_list_valid)
    # plt.xlabel("iteration")
    # plt.ylabel("NLLK")
    #
    # # plt.legend(['Train', 'Validation'])
    #
    # plt.show()
    """Printing test and validation accuracy with best hyperparams
    """

    best_lr = 0.01
    best_iterations = 13
    theta, beta, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
        train_data, val_data, best_lr, best_iterations)

    score = evaluate(test_data, theta, beta)
    score_valid = evaluate(val_data, theta, beta)

    print("Test accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score))

    print("Validation accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score_valid))

    """Plotting average log likelihood
    """
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
    # Implement part (c)                                                #
    #####################################################################
    """Plotting theta vs probability
    """
    # theta, beta, val_acc_lst, neg_lld_list_train, neg_lld_list_valid = irt(
    #     train_data, val_data, best_lr, best_iterations)
    #
    # five_questions = []
    # index = 0
    # while len(five_questions) != 5:
    #     if train_data['question_id'][index] not in five_questions:
    #         five_questions.append(train_data['question_id'][index])
    #     index += 1
    #
    # for question_id in five_questions:
    #     probs = []
    #     for i in range(len(theta)):
    #         probs.append(sigmoid(theta[i] - beta[question_id]))
    #
    #     plt.plot(theta, probs, 'r.')
    #     plt.xlabel('Theta')
    #     plt.ylabel('Probability correct')
    #     plt.title("IRT Probability Correct vs Theta, Question Id: {}".format(
    #         question_id))
    #     plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
