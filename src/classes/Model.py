import sklearn.svm
import sklearn.datasets
import numpy as np


def calc_accuracy(y_acc, y_pred):
    return float("{:.2f}".format(100 * np.mean(y_pred == y_acc)))


digits = sklearn.datasets.load_digits()


class Model:

    def __init__(self, cost, gamma, test_size=0.2) -> None:
        self.cost = cost
        self.gamma = gamma
        self.test_size = test_size

    def get_model_accuracy(self):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            digits.data, digits.target, test_size=self.test_size, shuffle=True, random_state=0
        )
        svm = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.cost, random_state=0)
        svm.fit(x_train, y_train)
        return calc_accuracy(y_test, svm.predict(x_test))
