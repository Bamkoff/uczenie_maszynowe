import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Część A
    reader = csv.reader(open('data6.tsv'), delimiter='\t')
    x = list()
    y = list()
    for xi, yi in reader:
        x.append(float(xi))
        y.append(float(yi))

    one_dim = np.poly1d(np.polyfit(x, y, 1))
    two_dim = np.poly1d(np.polyfit(x, y, 2))
    five_dim = np.poly1d(np.polyfit(x, y, 5))

    myline = np.linspace(1, 200, 100)

    fig1 = plt.figure(figsize=(6, 10))
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.set_title('Pierwszego stopnia')
    ax1.scatter(x, y)
    ax1.plot(myline, one_dim(myline))

    ax2 = fig1.add_subplot(3, 1, 2)
    ax2.set_title('Drugiego stopnia')
    ax2.scatter(x, y)
    ax2.plot(myline, two_dim(myline))

    ax3 = fig1.add_subplot(3, 1, 3)
    ax3.set_title('Piątego stopnia')
    ax3.scatter(x, y)
    ax3.plot(myline, five_dim(myline))
    #
    # plt.show()

    # w modelu wygenerowanym regresją wielomianową 5 stopnia akurat tym sposobem nie widać bardzo dużego overfitu, ale to może przez metode

    # Część B
    def cost(theta, X, y, lamb=0):
        """Wersja macierzowa funkcji kosztu zmieniona do regularyzacji"""
        m = len(y)
        J = 1.0 / (2.0 * m) * ((X * theta - y).T * (X * theta - y) + lamb * np.sum(np.power(theta[1:], 2)))
        return J.item()


    def gradient(theta, X, y, lamb=0):
        """Wersja macierzowa gradientu funkcji kosztu zmieniona do regularyzacji"""
        return 1.0 / len(y) * (X.T * (X * theta - y) + lamb/len(y)*theta)


    def gradient_descent(fJ, fdJ, theta, X, y, alpha=0.1, eps=10 ** -5, lamb=0):
        """Algorytm gradientu prostego (wersja macierzowa)"""
        current_cost = fJ(theta, X, y, lamb)
        logs = [[current_cost, theta]]
        while True:
            theta = theta - alpha * fdJ(theta, X, y, lamb)
            current_cost, prev_cost = fJ(theta, X, y, lamb), current_cost
            if abs(prev_cost - current_cost) > 10 ** 15:
                print('Algorithm does not converge!')
                break
            if abs(prev_cost - current_cost) <= eps:
                break
            logs.append([current_cost, theta])
        return theta, logs


    def plot_data(X, y, xlabel, ylabel):
        """Wykres danych (wersja macierzowa)"""
        fig = plt.figure(figsize=(16 * .6, 9 * .6))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        ax.scatter([X[:, 1]], [y], c='r', s=50, label='Dane')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.margins(.05, .05)
        ax.set_title('model 4 stopnia po regularyzacji')
        plt.ylim(y.min() - 1, y.max() + 1)
        plt.xlim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)
        return fig


    def plot_fun(fig, fun, X):
        """Wykres funkcji `fun`"""
        ax = fig.axes[0]
        x0 = np.min(X[:, 1]) - 1.0
        x1 = np.max(X[:, 1]) + 1.0
        Arg = np.arange(x0, x1, 0.1)
        Val = fun(Arg)
        return ax.plot(Arg, Val, linewidth='2')


    def h_poly(Theta, x):
        """Funkcja wielomianowa"""
        return sum(theta * np.power(x, i) for i, theta in enumerate(Theta.tolist()))


    def polynomial_regression(theta):
        """Funkcja regresji wielomianowej"""
        return lambda x: h_poly(theta, x)


    Xn = np.matrix(x).reshape(-1, 1)
    Xn /= np.amax(Xn, axis=0)
    Xn2 = np.power(Xn, 2)
    Xn2 /= np.amax(Xn2, axis=0)
    Xn3 = np.power(Xn, 3)
    Xn3 /= np.amax(Xn3, axis=0)
    Xn4 = np.power(Xn, 4)
    Xn4 /= np.amax(Xn4, axis=0)
    Xn5 = np.power(Xn, 5)
    Xn5 /= np.amax(Xn5, axis=0)

    Y_matrix = np.matrix(y).reshape(-1, 1)
    X = np.matrix(np.concatenate((np.ones((len(x), 1)), Xn), axis=1)).reshape(len(x), 2)
    X3 = np.matrix(np.concatenate((Xn, Xn2, Xn3), axis=1)).reshape(len(x), 3)
    X4 = np.matrix(np.concatenate((Xn, Xn2, Xn3, Xn4), axis=1)).reshape(len(x), 4)
    X5 = np.matrix(np.concatenate((Xn, Xn2, Xn3, Xn4, Xn5), axis=1)).reshape(len(x), 5)
    theta_start = np.matrix([0, 0, 0, 0]).reshape(4, 1)
    theta, _ = gradient_descent(cost, gradient, theta_start, X4, Y_matrix, 0.3)

    fig = plot_data(X4, Y_matrix, xlabel='x', ylabel='y')
    plot_fun(fig, polynomial_regression(theta), X)
    plt.show()

    # ciężko to porównać z modelami z podpunktu A ze względu na inną metodę wprowadzenia i skalowanie danych,
    # ale kod z częsci B powninien wykonać regularyzacje ze względu na zmienioną funkcje kosztu i zmieniony gradient
    # funkcji kosztu dla regresji liniowej