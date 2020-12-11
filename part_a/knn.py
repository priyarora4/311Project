from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
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
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    all_valid_acc = []
    for k in range(1, 27, 5):
        valid_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        all_valid_acc.append(valid_acc)

    plt.plot(range(1, 27, 5), all_valid_acc)
    plt.xlabel("k")
    plt.ylabel("accuracy")

    plt.savefig('../plots/A1a_by_user.png')

    print('Validation accuracies: ' + str(all_valid_acc))
    best_k_index = all_valid_acc.index(max(all_valid_acc))
    best_k = range(1, 27, 5)[best_k_index]
    print("best k = " + str(best_k))

    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("test acc = " + str(test_acc))


if __name__ == "__main__":
    main()
