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
    print("user | k: {} Validation Accuracy: {}".format(k, acc))
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
    print("item | k: {} Validation Accuracy: {}".format(k, acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    user_acc = []
    question_acc = []
    k_range = range(1, 27, 5)
    for k in k_range:
        user_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
        question_acc.append(knn_impute_by_item(sparse_matrix, val_data, k))

    # Plot With User Based Filtering
    max_k = k_range[int(np.argmax(user_acc))]
    plt.plot(k_range, user_acc)
    plt.xlabel("Values of K")
    plt.ylabel("Validation Accuracy")
    plt.xticks(k_range)
    plt.title("User Based Filtering: kNN Validation Accuracy vs K Values")
    plt.savefig('../plots/CSC311Final1auser.png')
    plt.show()
    test_acc = knn_impute_by_user(sparse_matrix, test_data, max_k)
    print("user: k = {} | test accuracy: {}".format(max_k, test_acc))

    # Plot With Item Based Filtering
    max_k = k_range[int(np.argmax(question_acc))]
    plt.plot(k_range, question_acc)
    plt.xlabel("Values of K")
    plt.ylabel("Validation Accuracy")
    plt.xticks(k_range)
    plt.title("Item Based Filtering: kNN Validation Accuracy vs K Values")
    plt.savefig('../plots/CSC311Final1aitem.png')
    plt.show()
    test_acc = knn_impute_by_item(sparse_matrix, test_data, max_k)
    print("item: k = {} | test accuracy: {}".format(max_k, test_acc))


if __name__ == "__main__":
    main()
