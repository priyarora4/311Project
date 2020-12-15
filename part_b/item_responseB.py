from utils import *

import numpy as np
import matplotlib.pyplot as plt
import ast

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, U, beta, a, Q):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param U: Matrix num_users x 388
    :param beta: Vector num_questions x 1
    :param a: Vector num_questions x 1
    :param Q: Matrix num_questions x 388
    :return: float
    """
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

    return -log_lklihood


def update_U_beta_a(data, lr, U, beta, a, Q):
    """ Update U and beta and a using gradient descent.


    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param U: Matrix num_users x 388
    :param beta: Vector num_questions x 1
    :param a: Vector num_questions x 1
    :param Q: Matrix num_questions x 388


    """
    num_samples = len(data['user_id'])
    u_sums = np.zeros(U.shape)
    beta_sums = np.zeros(beta.shape)
    a_sums = np.zeros(a.shape)

    #UPDATE U
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

    #UPDATE a
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

    #UPDATE BETA
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

    return U, beta, a


def irt(data, val_data, lr, iterations, Q):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param Q: matrix num_questions x 388
    :return: (theta, beta, val_acc_lst)
    """
    num_users = len(set(data['user_id']))
    num_questions = len(set(data['question_id']))
    U = np.zeros((num_users, 388))
    beta = np.zeros((num_questions, 1))
    a = np.ones((num_questions, 1))

    val_acc_lst = []
    neg_lld_list_train = []
    neg_lld_list_valid = []
    for i in range(iterations):
        #FOR PLOTTING
        # neg_lld = neg_log_likelihood(data, U=U, beta=beta, a=a, Q=Q)
        # neg_lld_list_train.append(neg_lld)
        # neg_lld_valid = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        # neg_lld_list_valid.append(neg_lld_valid)

        score = evaluate(data=val_data, U=U, beta=beta, a=a, Q=Q)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        val_acc_lst.append(score)
        U, beta, a = update_U_beta_a(data, lr, U=U, beta=beta, a=a, Q=Q)
        #print(i)

    return U, beta, a, val_acc_lst, neg_lld_list_train, neg_lld_list_valid


def evaluate(data, U, beta, a, Q):
    """Evaluate the accuracy of IRT model

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param U: matrix num_users x 388
    :param beta: vector num_questions x 1
    :param a: vector num_question x 1
    :param Q: matrix num_questions x 388
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        u_u = U[u].reshape((388,1))
        q_q = Q[q].reshape((388,1))
        x = a[q]*(u_u.T @ q_q - beta[q])
        p_a = g + (1-g)*sigmoid(x)
        pred.append(p_a[0][0] >= 0.5)

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def load_Q(num_questions):
    """ Construct Q representing subjects relating to each subject

        :param num_questions: int
        :return: matrix num_questions by 388
        """
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

#initalize g as guess factor
g = 0.25

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    private_test = load_private_test_csv('../data')

    #Initialize Q
    num_questions = len(set(train_data['question_id']))
    Q = load_Q(num_questions)

    #####################################################################
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

    ########################################################
    # PLOT NEG LOG LIKELIHOOD
    ########################################################

    # plt.plot(range(1, iterations + 1), neg_lld_list_train)
    # plt.plot(range(1, iterations + 1), neg_lld_list_valid)
    # plt.xlabel("iteration")
    # plt.ylabel("NLLK")

    # plt.legend(['Train', 'Validation'])

    # plt.show()

    # best_accuracies_per_lr.append(max_accuracy)


    #########################################################
    # Submitting predictions to kaggle
    #########################################################

    best_lr = 0.01
    best_iterations = 16
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

    # PRINT ACCURACIES
    print("Test accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score))

    print("Validation accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score_valid))

    print("train accuracy for lambda={} iterations={} :  {}".format(best_lr, best_iterations, score_train))

    #####################################################################
    # Plotting effects of a and g
    #####################################################################

    # question = train_data['question_id'][4]
    #
    # probs1 = []
    # probs2 = []
    # probs3 = []
    # theta = U @ Q[question]
    # for i in range(len(theta)):
    #     probs1.append(g + sigmoid((1-g)*1*(theta[i] - beta[question])))
    #     probs1.append(g + sigmoid((1-g)*0.5*(theta[i] - beta[question])))
    #     probs1.append(g + sigmoid((1-g)*2*(theta[i] - beta[question])))
    #
    # plt.plot(theta, probs1, '.')
    # plt.plot(theta, probs2, '.')
    # plt.plot(theta, probs3, '.')
    # plt.legend(['1', '0.5', '2'])
    # plt.xlabel('U[i]@Q[j]')
    # plt.ylabel('Probability correct')
    # plt.title("IRT Probability Correct vs U[i]@Q[j], Question Id: {}".format(question))
    # plt.show()


if __name__ == "__main__":
    main()
