import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
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
    print(f"Błąd średniokwadratowy wynosi {error}")
    print("Wynik metody score wynosi", liner_model.score(x_test_fires, y_test_fires))
