import numpy as np
x_array = np.array([1.0, 2.0, 3.0, 1.0, 3.0, 6.0]).reshape(2, 3)
x_matrix = np.matrix(x_array)

y_array = np.array([5.0, 6.0]).reshape(2, 1)
y_matrix = np.matrix(y_array)
print('on arrays:\n', np.matmul(np.matmul(np.linalg.inv(np.matmul(x_array.T, x_array)), x_array.T), y_array))
print('\non matrix:\n', (x_matrix.T * x_matrix)**-1 * x_matrix.T * y_matrix)

# results:

# on arrays:
# [[13.9375  ]
#  [ 1.3125  ]
#  [-0.015625]]
#
# on matrix:
# [[13.9375  ]
#  [ 1.3125  ]
#  [-0.015625]]
