import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# import seaborn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    alldata = pandas.read_csv(
        'flats.tsv', header=0, sep='\t')

    # sprawdzenie jakie dane zawiera plik
    # print("Liczba pokoi:", pandas.unique(alldata["Liczba pokoi"]))
    # print("Miejsca parkingowe:", pandas.unique(alldata["Miejsce parkingowe"]))
    # print("Liczba pięter w budynku:", pandas.unique(alldata["Liczba pięter w budynku"]))
    # print("Piętro:", pandas.unique(alldata["Piętro"]))
    # print("Typ zabudowy:", pandas.unique(alldata["Typ zabudowy"]))
    # print("Okna:", pandas.unique(alldata["Okna"]))
    # print("Materiał budynku:", pandas.unique(alldata["Materiał budynku"]))
    # print("Rok budowy:", pandas.unique(alldata["Rok budowy"]))
    # print("Forma własności:", pandas.unique(alldata["Forma własności"]))
    # print("Forma kuchni:", pandas.unique(alldata["Forma kuchni"]))
    # print("Stan:", pandas.unique(alldata["Stan"]))
    # print("Stan instalacji:", pandas.unique(alldata["Stan instalacji"]))
    # print("Głośność:", pandas.unique(alldata["Głośność"]))
    # print("Droga dojazdowa:", pandas.unique(alldata["Droga dojazdowa"]))
    # print("Stan łazienki:", pandas.unique(alldata["Stan łazienki"]))
    # print("Powierzchnia w m2:", pandas.unique(alldata["Powierzchnia w m2"]))

    # ==================================================================================================================
    # model pierwszy
    print("Pierwszy model:")

    dataset = pandas.DataFrame()
    dataset['Liczba pokoi'] = alldata['Liczba pokoi']
    dataset['Piętro'] = alldata['Piętro']
    dataset['Miejsce parkingowe'] = alldata['Miejsce parkingowe']
    dataset['opis'] = alldata['opis']
    dataset['cena'] = alldata['cena']
    dataset2 = dataset

    #  zastąpienie w mieszanych danych napisy na dane liczbowe lub NaN
    dataset["Piętro"] = dataset['Piętro'].apply(lambda x: '0' if x in [' parter', ' niski parter'] else x)
    dataset["Piętro"] = dataset['Piętro'].apply(lambda x: '0' if x is np.nan else x)
    dataset["Piętro"] = dataset['Piętro'].apply(lambda x: np.nan if x == ' poddasze' else x)
    dataset["Piętro"] = dataset["Piętro"].str.strip()

    print("liczba rekordów przed usunięciem NaN:", len(dataset))

    # usunięcie recordów z wartościami NaN
    dataset = dataset.dropna()

    print("liczba rekordów po usunięciu NaN:", len(dataset))

    # zrobienie wartości boolowskich z danych kategorycznych o miejscach parkingowych
    dataset = pandas.get_dummies(dataset, columns=["Miejsce parkingowe"])

    # zrobienie danych boolowskich z opisów sprawdzając czy opis zawiera napis nowy,
    # zawiera słowo dom i czy jest na wynajem
    dataset['nowe_w_opisie'] = dataset['opis'].apply(lambda x: 1 if 'nowe' in x.lower() else 0)
    dataset['dom_w_opisie'] = dataset['opis'].apply(lambda x: 1 if 'dom' in x.lower() or 'domek' in
                                                                      x.lower() else 0)
    dataset['wynajem_w_opisie'] = dataset['opis'].apply(lambda x: 1 if 'wynajem' in x.lower() or 'wynajme' in
                                                                          x.lower() else 0)

    # przekonwertowanie stringów na liczby
    dataset["Piętro"] = dataset["Piętro"].astype(float)

    # wyrzucenie kolumny opis
    del dataset['opis']

    # przeniesienie kolumny z ceną na ostatni indeks datasetu dla ułatwienia przetworzenia danych
    key_list = list(dataset.keys().values)
    price_index = 0
    for i, elem in enumerate(key_list):
        if elem == 'cena':
            price_index = i
            break
    key_list.pop(price_index)
    key_list.append('cena')

    dataset = dataset.reindex(columns=key_list)

    # przetworzenie danych do formatu jaki przyjmuje liner_model.predict
    m, n_plus_1 = dataset.values.shape
    n = n_plus_1 - 1

    XMx = np.matrix(dataset.values[:, 0:n]).reshape(m, n)
    yMx = np.array(dataset['cena'].values)

    xTrain, xTest = XMx[m // 5:], XMx[:m // 5]
    yTrain, yTest = yMx[m // 5:], yMx[:m // 5]

    # stworzenie i dopasowanie modelu
    liner_model = LinearRegression()
    liner_model.fit(xTrain, yTrain)

    y_predicted = liner_model.predict(xTest)

    # Ewaluacja
    error = mean_squared_error(yTest, y_predicted)
    print(f"Błąd średniokwadratowy wynosi {error}")
    print("Wynik metody score wynosi", liner_model.score(xTest, yTest))

    # ==================================================================================================================
    # drugi model
    print("\nDrugi model:")

    # dodałem powierzchnie do poprzedniego
    dataset2["Powierzchnia w m2"] = alldata["Powierzchnia w m2"]

    #  zastąpienie w mieszanych danych napisy na dane liczbowe lub NaN
    dataset2["Piętro"] = dataset2['Piętro'].apply(lambda x: '0' if x in [' parter', ' niski parter'] else x)
    dataset2["Piętro"] = dataset2['Piętro'].apply(lambda x: '0' if x is np.nan else x)
    dataset2["Piętro"] = dataset2['Piętro'].apply(lambda x: np.nan if x == ' poddasze' else x)
    dataset2["Piętro"] = dataset2["Piętro"].str.strip()

    print("liczba rekordów przed usunięciem NaN:", len(dataset2))

    # usunięcie rekordów z wartościami NaN
    dataset2 = dataset2.dropna()

    print("liczba rekordów po usunięciu NaN:", len(dataset2))

    # zrobienie wartości boolowskich z danych kategorycznych o miejscach parkingowych
    dataset2 = pandas.get_dummies(dataset2, columns=["Miejsce parkingowe"])

    # zrobienie danych boolowskich z opisów sprawdzając czy opis zawiera napis nowy,
    # zawiera słowo dom i czy jest na wynajem
    dataset2['nowe_w_opisie'] = dataset2['opis'].apply(lambda x: 1 if 'nowe' in x.lower() else 0)
    dataset2['dom_w_opisie'] = dataset2['opis'].apply(lambda x: 1 if 'dom' in x.lower() or 'domek' in
                                                                      x.lower() else 0)
    dataset2['wynajem_w_opisie'] = dataset2['opis'].apply(lambda x: 1 if 'wynajem' in x.lower() or 'wynajme' in
                                                                          x.lower() else 0)

    # przekonwertowanie stringów na liczby
    dataset["Piętro"] = dataset["Piętro"].astype(float)

    # wyrzucenie odstających rekordów
    dataset2 = dataset2.loc[(dataset2["cena"] < 3 * 10 ** 6)
                            & (dataset2["Powierzchnia w m2"] < 550)]

    # wykres cen do znalezienia odstających rekordów
    # plt.scatter(np.arange(0, len(dataset2["cena"])), dataset2["cena"])
    # plt.show()

    # wykres powierzchni do znalenienia odstających rekordów
    # plt.scatter(np.arange(0, len(dataset2["Powierzchnia w m2"])), dataset2["Powierzchnia w m2"])
    # plt.show()

    # wyrzucenie kolumny opis
    del dataset2['opis']

    # przeniesienie kolumny z ceną na ostatni indeks datasetu dla ułatwienia przetworzenia danych
    key_list = list(dataset2.keys().values)
    price_index = 0
    for i, elem in enumerate(key_list):
        if elem == 'cena':
            price_index = i
            break
    key_list.pop(price_index)
    key_list.append('cena')
    dataset2 = dataset2.reindex(columns=key_list)

    # przetworzenie danych do formatu jaki przyjmuje liner_model.predict
    m, n_plus_1 = dataset2.values.shape
    n = n_plus_1 - 1

    XMx = np.matrix(dataset2.values[:, 0:n]).reshape(m, n)
    yMx = np.array(dataset2['cena'].values)

    xTrain, xTest = XMx[m // 5:], XMx[:m // 5]
    yTrain, yTest = yMx[m // 5:], yMx[:m // 5]

    # stworzenie i dopasowanie modelu
    liner_model = LinearRegression()
    liner_model.fit(xTrain, yTrain)

    # Ewaluacja
    y_predicted = liner_model.predict(xTest)

    error = mean_squared_error(yTest, y_predicted)
    print(f"Błąd średniokwadratowy wynosi {error}")
    print("Wynik metody score wynosi", liner_model.score(xTest, yTest))
