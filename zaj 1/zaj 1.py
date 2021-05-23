import numpy as np


# zadanie 1
def kwadraty(input_list):
    output_list = [element**2 for element in input_list if element > 0]
    return output_list


test = [1, 2, 3, 4, 5]
print(kwadraty(test))
# zadanie 2
print(np.arange(1, 51).reshape(5, 10))


# zadanie 3
def wlasciwosci_macierzy(A):
    liczba_elementow = A.size
    liczba_kolumn = A.shape[1]
    liczba_wierszy = A.shape[0]
    srednie_wg_wierszy = A.mean(axis=1)
    srednie_wg_kolumn = A.mean(axis=0)
    kolumna_2 = A[:, 2]
    wiersz_3 = A[3, :]
    return (
        liczba_elementow, liczba_kolumn, liczba_wierszy,
        srednie_wg_wierszy, srednie_wg_kolumn,
        kolumna_2, wiersz_3)


A = np.arange(1, 51).reshape(5, 10)
print(wlasciwosci_macierzy(A))


# zadanie 4
def dzialanie1(A, x):
    """ iloczyn macierzy A z wektorem x """
    return np.dot(A, x)


def dzialanie2(A, B):
    """ iloczyn macierzy A · B """
    return np.dot(A, B)


def dzialanie3(A, B):
    """ wyznacznik det(A · B) """
    return np.linalg.det(dzialanie2(A, B))


def dzialanie4(A, B, x):
    """ wynik działania (A · B)^T - B^T · A^T """
    return dzialanie2(A, B).T - dzialanie2(B.T, A.T)
