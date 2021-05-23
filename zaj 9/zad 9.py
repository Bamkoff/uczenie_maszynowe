from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    all_data = pandas.read_csv('flats_for_clustering.tsv', header=0, sep='\t')

    # sprawdzenie jakie dane zawiera plik
    # print("Piętra:", pandas.unique(all_data["Piętro"]))
    # print("Liczby pięter w budynku:", pandas.unique(all_data["Liczba pięter w budynku"]))
    # print("Liczba pokoi:", pandas.unique(all_data["Liczba pokoi"]))
    # print("Powierzchnia w m2:", pandas.unique(all_data["Powierzchnia w m2"]))

    # wykres cen do znalezienia odstających rekordów
    # plt.scatter(np.arange(0, len(all_data["cena"])), all_data["cena"])
    # plt.show()

    # wykres powierzchni do znalenienia odstających rekordów
    # plt.scatter(np.arange(0, len(all_data["Powierzchnia w m2"])), all_data["Powierzchnia w m2"])
    # plt.show()

    #  zastąpienie w mieszanych danych napisy i NaN na dane liczbowe lub NaN
    all_data["Piętro"] = all_data['Piętro'].apply(lambda x: '0' if x in ['parter', 'niski parter'] else x)
    all_data["Piętro"] = all_data['Piętro'].apply(lambda x: '0' if x is np.nan else x)
    all_data["Piętro"] = all_data['Piętro'].apply(lambda x: np.nan if x == 'poddasze' else x)

    print("liczba rekordów przed usunięciem NaN:", len(all_data))
    # usunięcie recordów z wartościami NaN
    all_data = all_data.dropna()
    print("liczba rekordów po usunięciu NaN:", len(all_data))

    # przekonwertowanie stringów na liczby
    all_data["Piętro"] = all_data["Piętro"].astype(float)

    print("liczba rekordów przed usunięciem rekordów odstających:", len(all_data))
    # wyrzucenie odstających rekordów
    all_data = all_data.loc[(all_data["cena"] < 3 * 10 ** 6)
                            & (all_data["Powierzchnia w m2"] < 550)]
    print("liczba rekordów po usunięciu rekordów odstających:", len(all_data))

    # przekształcenie przechowania danych do postaci, którą przyjmuje KMeans
    m, n_plus_1 = all_data.values.shape
    X = np.matrix(all_data.values).reshape(m, n_plus_1)

    # stworzenie modelu dzielącego dane na 5 klastrów
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    # print(f"przypisane klastry: {kmeans.labels_}")
    #
    # zliczenie ile elemetów znalazło się w danej klasie
    # for i in pandas.unique(kmeans.labels_):
    #     counter = 0
    #     for j in kmeans.labels_:
    #         if j == i:
    #             counter += 1
    #     print(f"klaster {i} ma {counter} elementów")

    # przekonwertowanie 5 cech na 2 algorytmem PCA
    pca = PCA(n_components=2).fit_transform(X)
    # print(pca[:, 1])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    colors = ['blue', 'green', 'red', 'orange', 'black']
    handles = []

    klaster = 0
    for color in colors:
        to_add = []
        x = []
        y = []
        for index in range(len(pca)):
            if kmeans.labels_[index] == klaster:
                to_add.append(index)
        for index in to_add:
            x.append(pca[index, 0])
            y.append(pca[index, 1])
        handles.append(ax.scatter(x, y, c=color, alpha=0.5))
        klaster += 1

    ax.legend(handles, [0, 1, 2, 3, 4])
    # wykres otrzymanych cech
    # plt.scatter(pca[:, 0], pca[:, 1], alpha=0.5, c='green')
    plt.show()
