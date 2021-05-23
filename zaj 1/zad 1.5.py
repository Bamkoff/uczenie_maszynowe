import numpy as np
x_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(3, 3)
x_matrix = np.matrix(x_array)

print(x_array**-1) # zwraca array z każdym elementem podniesionym do potęgi -1
print(x_matrix**-1) # zwraca odwrotność macierzy
