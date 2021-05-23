import csv
import numpy as np
import matplotlib.pyplot as plt


def h_(theta, x):
    return theta[0] + theta[1] * x


def J_(h, theta, x, y):
    """Funkcja kosztu"""
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i])**2 for i in range(m))


def gradient_descent(h, cost_fun, theta, x, y, alpha, eps):
    current_cost = cost_fun(h, theta, x, y)
    log = [[current_cost, theta]]  # log przechowuje wartości kosztu i parametrów
    m = len(y)
    while True:
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta  # jednoczesna aktualizacja - używamy zmiennej tymaczasowej
        try:
            current_cost, prev_cost = cost_fun(h, theta, x, y), current_cost
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log

# Część podstawowa

reader = csv.reader(open('fires_thefts.csv'), delimiter=',')
x = list()
y = list()
for xi, yi in reader:
    x.append(float(xi))
    y.append(float(yi))

theta, log = gradient_descent(h_, J_, [2.0, 1.0], x, y, 0.001, 0.00001)
print("Theta", theta)
print("J(Theta) =", log[-1][0])
print("Iterations:", len(log))
print("Predykcja dla 50 pożarów:", h_(theta, 50))
print("Predykcja dla 100 pożarów:", h_(theta, 100))
print("Predykcja dla 200 pożarów:", h_(theta, 200))
# Część zaawansowana

theta, log_1 = gradient_descent(h_, J_, [2.0, 0.2], x, y, 0.1, 0.00001)
theta, log_01 = gradient_descent(h_, J_, [2.0, 0.2], x, y, 0.01, 0.00001)
theta, log_001 = gradient_descent(h_, J_, [2.0, 0.2], x, y, 0.001, 0.00001)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
x_ = np.arange(0, 201, 1)
y_1 = []

if len(x_) > len(log_1):
    y_1 = [log_1[i][0] for i in np.arange(0, len(log_1))] + [None for i in np.arange(0, 201 - len(log_1))]
else:
    y_1 = [log_1[i][0] for i in x_]
y_01 = [log_01[i][0] for i in x_]
y_001 = [log_001[i][0] for i in x_]

ax.plot(x_, y_1, color='red', lw=2, label='0.1')
ax.plot(x_, y_01, color='green', lw=2, label='0.01')
ax.plot(x_, y_001, color='blue', lw=2, label='0.001')
ax.legend()
plt.axis([0, 200, 0, 10**15])
plt.show()
