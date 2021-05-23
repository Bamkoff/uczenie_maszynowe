import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    # ==================================================================================================================
    # Zadanie 3
    # pobranie danych z zadania 3 zrobienie na nich modelu regresji liniowej używając _scikit-learn
    fires_thefts = pandas.read_csv('fires_thefts.csv', header=None)

    split_point = split_point = int(0.8 * len(fires_thefts))

    x_train_fires, y_train_fires = fires_thefts[0][:split_point], fires_thefts[1][:split_point]
    x_test_fires, y_test_fires = fires_thefts[0][split_point:], fires_thefts[1][split_point:]

    x_train_fires = np.array(x_train_fires).reshape(-1, 1)

    liner_model = LinearRegression()
    liner_model.fit(x_train_fires, y_train_fires)

    x_test_fires = np.array(x_test_fires).reshape(-1, 1)

    y_predicted = liner_model.predict(x_test_fires)

    # ewaluacja wygenerowanego modelu na danych testowych
    error = mean_squared_error(y_test_fires, y_predicted)
    print("Zadanie 3 ================================================")
    print(f"Błąd średniokwadratowy wynosi {error}")
    print("Wynik metody score wynosi", liner_model.score(x_test_fires, y_test_fires))

    # ==================================================================================================================
    # Zadanie 4
    # pobranie danych z pliku
    dataset = pandas.read_csv('gratkapl-centrenrm.csv')
    dataset_setosa_multi = pandas.DataFrame()
    dataset_setosa_multi['Price'] = dataset['Price']
    dataset_setosa_multi['Rooms'] = dataset['Rooms']
    dataset_setosa_multi['SqrMeters'] = dataset['SqrMeters']
    dataset_setosa_multi['Floor'] = dataset['Floor']
    dataset_setosa_multi['Centre'] = dataset['Centre']

    # pozbycie się odstających wartości
    dataset_no_outliers = dataset_setosa_multi.loc[(dataset_setosa_multi['Price'] > 1000)
                                                   & (dataset_setosa_multi['Price'] < 10 ** 7)
                                                   & (dataset_setosa_multi['SqrMeters'] < 185)]

    m, n_plus_1 = dataset_no_outliers.values.shape
    n = n_plus_1 - 1
    Xn = dataset_no_outliers.values[:, 0:n].reshape(m, n)

    XMx4 = np.matrix(np.concatenate((np.ones((m, 1)), Xn), axis=1)).reshape(m, n_plus_1)
    yMx4 = np.array(dataset_no_outliers.values[:, n])

    # normalizacja średniej
    XMx4_norm = (XMx4 - np.mean(XMx4, axis=0)) / np.amax(XMx4, axis=0)

    # podzielenie na zbiór do uczenia i zbiór do testowania
    xTrain, xTest = XMx4_norm[m // 5:], XMx4_norm[:m // 5]
    yTrain, yTest = yMx4[m // 5:], yMx4[:m // 5]

    # wygenerowanie modelu funkcją z scikit-learn
    model = LogisticRegression(random_state=0).fit(xTrain, yTrain)
    predict = model.predict(xTest)

    # obliczenie celności modelu regresji liniowej
    acc_Logistic_Model = 0
    for i, rest in enumerate(yTest):
        acc_Logistic_Model += predict[i].item() == yTest[i].item()

    print("\nZadanie 4 ================================================")

    print("Accuracy:", acc_Logistic_Model / len(yTest))

    precision, recall, fscore, support = precision_recall_fscore_support(yTest, predict, average="micro")

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")

    score = model.score(xTest, yTest)

    print(f"Model score: {score}")
