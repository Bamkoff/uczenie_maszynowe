import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
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
    print(yMx4)
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

    print("Accuracy:", acc_Logistic_Model / len(yTest))

    precision, recall, fscore, support = precision_recall_fscore_support(yTest, predict, average="micro")

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-score: {fscore}")

    score = model.score(xTest, yTest)

    print(f"Model score: {score}")
