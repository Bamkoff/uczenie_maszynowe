import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))


x = np.arange(-1.0, 5.0, 0.025)


# podpunkt A

y1 = (6 - 4) * x**2 + (8 - 5) * x + (1 - 6)
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_ylabel('y')
ax1.set_xlabel('x')
ax1.set_title('Podpunkt A')
ax1.plot(x, y1, color='red', lw=2)

# podpunkt B
y2 = (np.e**x) / ((np.e**x) + 1.0)
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_ylabel('y')
ax2.set_xlabel('x')
ax2.set_title('Podpunkt B')
p1 = ax2.plot(x, y1, color='red', lw=2)
p2 = ax2.plot(x, y2, color='green', lw=2)

plt.subplots_adjust(wspace=0.2, hspace=.8)

plt.show()
