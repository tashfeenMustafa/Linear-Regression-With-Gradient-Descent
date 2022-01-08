## Submitted By:
## Name: Tashfeen Mustafa Choudhury
## ID: 1821069
## Course: CSE425
## Semester: Autumn 2021
## Assignment: 3

## Linear Regression Using Gradient Descent
## hypothesis function, h(x) = C0 + C1x, (theta => C)
import numpy as np

training_data = [
    [166, 54],
    [195, 82],
    [200, 72],
    [260, 72],
    [265, 90],
    [335, 124],
    [370, 94],
    [450, 118],
    [517, 152],
    [552, 132]
]

training_data = np.asarray(training_data)

x_train = training_data[:, :1]
y_train = training_data[:, 1:2]

print('x_train: {}'.format(x_train))
print('y_train: {}'.format(y_train))

# get length of the training dataset
n = len(x_train)

# initialize the value for gamma/learning rate
learning_rate = 0.0001

# initialize C0 and C1
C0 = np.zeros((n, 1))
C1 = np.zeros((n, 1))

print('C0: {}'.format(C0))
print('C1: {}'.format(C1))

mean_square_error = 0

epochs = 0
while epochs < 1000:
    print('Turn: {}\n'.format(epochs))

    h_x = C0 + C1 * x_train  # defining the hypothesis function h(x)
    print('h_x: {}'.format(h_x))

    error = h_x - y_train  # calculating the cost
    print('cost: {}'.format(error))

    mean_square_error = np.sum(error ** 2)  # calculating the mean square error
    mean_square_error = mean_square_error / n
    print('Mean_Square_Error: {}'.format(mean_square_error))

    if mean_square_error == float('inf'):
        break

    # calculating C0 and C1
    C0 = C0 - learning_rate * 2 * np.sum(error) / n
    C1 = C1 - learning_rate * 2 * np.sum(error * x_train) / n

    print('C0: {}'.format(C0))
    print('C1: {}'.format(C1))

    epochs += 1
