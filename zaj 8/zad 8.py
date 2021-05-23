import numpy as np
import pandas
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

loss_list = [
    "hinge",
    "log",
    "modified_huber",
    "squared_hinge",
    "perceptron",
    "squared_loss",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive"
]

penalty_list = [
    "l2",
    "l1",
    "elasticnet"
]

learning_rate_list = [
    "optimal",
    "constant",
    "invscaling",
    "adaptive"
]


def batch_iterate(x, y, batch_size):
    """Iterator dzielący dane na mini-batche"""
    assert len(x) == len(y)
    dataset_size = len(x)
    current_index = 0
    while current_index < dataset_size:
        x_batch = x[current_index: current_index + batch_size]
        y_batch = y[current_index: current_index + batch_size]
        yield x_batch, y_batch
        current_index += batch_size


def preprocess_column(column):
    values = pandas.unique(column)
    counter = 1
    for value in values:
        column = column.replace({value: counter})
        counter += 1
    return column


def create_and_evaluate_model(x_train, y_train, x_test, y_test, classes, loss='hinge', penalty="l2", learning_rate="optimal", eta0=1.0):
    model = SGDClassifier(loss=loss, penalty=penalty, learning_rate=learning_rate, eta0=eta0)
    batch_iterator = batch_iterate(x_train, y_train, batch_size=100)
    for x_batch, y_batch in batch_iterator:
        model.partial_fit(x_batch, y_batch, classes=classes)

    y_predicted = model.predict(x_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predicted, average="micro")
    return precision, recall, fscore, model.score(x_test, yTest)


def find_best_fitting_parameters(x_train, y_train, x_test, y_test, classes):
    best_precision, best_recall, best_fscore, best_score = 0, 0, 0, 0
    best_precision_combo = ""
    best_recall_combo = ""
    best_fscore_combo = ""
    best_score_combo = ""

    for _loss in loss_list:
        for _penalty in penalty_list:
            for _learning_rate in learning_rate_list:

                _precision, _recall, _fscore, _score = create_and_evaluate_model(x_train, y_train, x_test,
                                                                                 y_test, classes,
                                                                                 loss=_loss,
                                                                                 penalty=_penalty,
                                                                                 learning_rate=_learning_rate,
                                                                                 eta0=1.0)
                if best_precision < _precision:
                    best_precision = _precision
                    best_precision_combo = "loss=" + _loss + ", penalty=" + _penalty + ", learning_rate=" + _learning_rate
                if best_recall < _recall:
                    best_recall = _recall
                    best_recall_combo = "loss=" + _loss + ", penalty=" + _penalty + ", learning_rate=" + _learning_rate
                if best_fscore < _fscore:
                    best_fscore = _fscore
                    best_fscore_combo = "loss=" + _loss + ", penalty=" + _penalty + ", learning_rate=" + _learning_rate
                if best_score < _score:
                    best_score = _score
                    best_score_combo = "loss=" + _loss + ", penalty=" + _penalty + ", learning_rate=" + _learning_rate

    print(f"Najlepsza precyzja to {best_precision} z parametrami {best_precision_combo}")
    print(f"Najlepszy recall to {best_recall} z parametrami {best_recall_combo}")
    print(f"Najlepszy fscore to {best_fscore} z parametrami {best_fscore_combo}")
    print(f"Najlepszy score to {best_score} z parametrami {best_score_combo}")


if __name__ == "__main__":
    all_data = pandas.read_csv('mushrooms.tsv', header=None, sep='\t')

    # przerobienie danych tekstowych na liczbowe
    for i in range(1, 23):
        all_data[i] = preprocess_column(all_data[i])

    # pierwszy dobór cech
    m, n_plus_1 = all_data.values.shape
    X = np.matrix(all_data.values[:, 1:6]).reshape(m, 5)
    Y = np.array(all_data.values[:, 0])
    xTrain, xTest = X[m // 5:], X[:m // 5]
    yTrain, yTest = Y[m // 5:], Y[:m // 5]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(xTrain)
    x_test_scaled = scaler.fit_transform(xTest)

    # drugi dobór cech
    X2 = np.matrix(all_data.values[:, 1:]).reshape(m, 22)
    x2Train, x2Test = X2[m // 5:], X2[:m // 5]
    x2_train_scaled = scaler.fit_transform(x2Train)
    x2_test_scaled = scaler.fit_transform(x2Test)

    # trzeci dobór cech
    X3 = np.matrix(all_data.values[:, 5:17]).reshape(m, 12)
    x3Train, x3Test = X2[m // 5:], X2[:m // 5]
    x3_train_scaled = scaler.fit_transform(x3Train)
    x3_test_scaled = scaler.fit_transform(x3Test)

    _classes = pandas.unique(all_data[0])
    print("Pierwszych 5 cech:")
    find_best_fitting_parameters(x_train_scaled, yTrain, x_test_scaled, yTest, _classes)

    print("\nWszystkie cechy:")
    find_best_fitting_parameters(x2_train_scaled, yTrain, x2_test_scaled, yTest, _classes)

    print("\nCechy od 5 do 16:")
    find_best_fitting_parameters(x3_train_scaled, yTrain, x3_test_scaled, yTest, _classes)