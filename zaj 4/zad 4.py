import pandas
import numpy as np
import random
# import seaborn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def plot_data(X, y, xlabel, ylabel):
    """Wykres danych (wersja macierzowa)"""
    fig = plt.figure(figsize=(16 * .6, 9 * .6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.scatter(X, y, c='r', s=50, label='Dane')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(.05, .05)
    plt.ylim(y.min() - 1, y.max() + 1)
    plt.xlim(np.min(X) - 1, np.max(X) + 1)
    plt.xticks(range(2))
    return fig


if __name__ == "__main__":
    # pobranie danych z pliku
    dataset = pandas.read_csv('gratkapl-centrenrm.csv')

    # wyodrębnienie danych, które nas interesują
    dataset_setosa_multi = pandas.DataFrame()
    dataset_setosa_multi['Price'] = dataset['Price']
    dataset_setosa_multi['Rooms'] = dataset['Rooms']
    dataset_setosa_multi['SqrMeters'] = dataset['SqrMeters']
    dataset_setosa_multi['Floor'] = dataset['Floor']
    dataset_setosa_multi['Centre'] = dataset['Centre']

    # Sprawdzenie czy są jakieś dane bardzo odstające, ewidentnie błędne, tekstowe lub NaN
    # print(pandas.unique(dataset_setosa_multi['Price']))
    # print(pandas.unique(dataset_setosa_multi['Rooms']))
    # print(pandas.unique(dataset_setosa_multi['SqrMeters']))
    # print(pandas.unique(dataset_setosa_multi['Floor']))
    # print(pandas.unique(dataset_setosa_multi['Centre']))

    for_deviation_check = pandas.DataFrame()
    for_deviation_check['Price'] = dataset['Price']
    for_deviation_check['Centre'] = dataset['Centre']
    plot_data(for_deviation_check['Centre'], for_deviation_check['Price'], 'Centre', 'Price')
    # seaborn.relplot(data=for_deviation_check, x='Centre', y='Price')

    # for_deviation_check['Rooms'] = dataset['Rooms']
    # for_deviation_check['Centre'] = dataset['Centre']
    # seaborn.relplot(data=for_deviation_check, x='Centre', y='Rooms', hue='Centre')

    # for_deviation_check['SqrMeters'] = dataset['SqrMeters']
    # for_deviation_check['Centre'] = dataset['Centre']
    # seaborn.relplot(data=for_deviation_check, x='Centre', y='SqrMeters', hue='Centre')

    # for_deviation_check['Floor'] = dataset['Floor']
    # for_deviation_check['Centre'] = dataset['Centre']
    # seaborn.relplot(data=for_deviation_check, x='Centre', y='Floor', hue='Centre')

    # plt.xticks(range(2))
    plt.show()

    # znalazłem cene 1, która jest ewidentnym błędem oraz pare innych cen i powieżchni odstających od reszty danych

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
    xTrain, xTest = XMx4_norm[m//5:], XMx4_norm[:m//5]
    yTrain, yTest = yMx4[m//5:], yMx4[:m//5]

    # wygenerowanie modelu funkcją z scikit-learn
    model = LogisticRegression(random_state=0).fit(xTrain, yTrain)

    # Przewidzenie wartości dla zbioru testowego
    predict = model.predict(xTest)

    # obliczenie celności modelu regresji liniowej
    acc_Logistic_Model = 0
    for i, rest in enumerate(yTest):
        acc_Logistic_Model += predict[i].item() == yTest[i].item()

    # obliczenie celności modelu losowego
    acc_Random_Model = 0
    for i, rest in enumerate(yTest):
        acc_Random_Model += float(random.randint(0, 1)) == yTest[i].item()

    print("\nAccuracy of logistic model:", acc_Logistic_Model / len(yTest))
    print("\nAccuracy of random model:", acc_Random_Model / len(yTest))
