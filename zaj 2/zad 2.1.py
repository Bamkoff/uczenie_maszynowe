import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data2.csv', sep=',')
data_array = data.to_numpy()

x = data_array[:, 1]
y = data_array[:, 2]

plt.plot(x, y, 'ro')  # "gx" - zielone (Green) krzyżyki (x)
plt.show()
