import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mean(values):
    return sum(values) / float(len(values))


def stddev(values):
    m = mean(values)
    std = sum(abs(x - m) for x in values) / len(values)
    return std


def zscore_normalize(data):
    mean_val = mean(data)
    stddev_val = stddev(data)
    normalized_data = [(x - mean_val) / stddev_val for x in data]
    return normalized_data, mean_val, stddev_val


def gradient_descent(X1, X2, X3, Y, learning_rate, epochs):
    m = len(X1)
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    theta3 = 0.0
    cost_history = []

    for epoch in range(epochs):
        error_sum = 0.0
        theta0_temp = 0.0
        theta1_temp = 0.0
        theta2_temp = 0.0
        theta3_temp = 0.0

        for i in range(m):
            xi1 = X1[i]
            xi2 = X2[i]
            xi3 = X3[i]
            yi = Y[i]
            h = theta0 + theta1 * xi1 + theta2 * xi2 + theta3 * xi3
            error = h - yi
            error_sum += error

            theta0_temp += error
            theta1_temp += error * xi1
            theta2_temp += error * xi2
            theta3_temp += error * xi3

        theta0 -= (learning_rate / m) * theta0_temp
        theta1 -= (learning_rate / m) * theta1_temp
        theta2 -= (learning_rate / m) * theta2_temp
        theta3 -= (learning_rate / m) * theta3_temp

        cost = (1 / (2 * m)) * error_sum ** 2
        cost_history.append(cost)

    return theta0, theta1, theta2, theta3, cost_history


def closed_form_solution(floor, bed, area, Y):
    m = len(floor)
    X = np.array([np.ones(m), floor, bed, area]).T
    transpose_X = np.transpose(X)
    XtX = np.dot(transpose_X, X)
    inverse_XtX = np.linalg.inv(XtX)
    inverse_trans = np.dot(inverse_XtX, transpose_X)
    theta = np.dot(inverse_trans, Y)
    print(len(theta))
    return theta


X = []
Y = []


floors_data = []
bedrooms_data = []
area_data = []

data_file = 'DataX.dat'

with open(data_file, 'r') as file:
    for line in file:
        values = line.strip().split()

        # Assuming the order in the file is area, bedrooms, area
        floors = float(values[0])
        bedrooms = float(values[1])
        area = float(values[2])

        # Append the values to their respective lists
        floors_data.append(floors)
        bedrooms_data.append(bedrooms)
        area_data.append(area)


normalized_floors, mean_floors, stddev_floors = zscore_normalize(floors_data)
normalized_bedrooms, mean_bedrooms, stddev_bedrooms = zscore_normalize(bedrooms_data)
normalized_area, mean_area, stddev_area = zscore_normalize(area_data)
#print(normalized_floors)
print("-------------------------")
#print(normalized_bedrooms)
print("-------------------------")
#print(normalized_area)

data_file2 = 'DataY.dat'

with open(data_file2, 'r') as file:
    for line in file:
        values = line.strip().split()
        prices = float(values[0])

        Y.append(prices)

learning_rate = 0.02
epochs = 1000

#######################################################################################################
#######################################################################################################

theta0, theta1, theta2, theta3, cost_history = gradient_descent(normalized_floors, normalized_bedrooms, normalized_area, Y, learning_rate, epochs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for normalized_floors and normalized_bedrooms
ax.scatter(normalized_floors, normalized_bedrooms, Y, c='r', marker='o', label='Data Points')

normalized_floors_range = np.linspace(min(normalized_floors), max(normalized_floors), 100)
normalized_bedrooms_range = np.linspace(min(normalized_bedrooms), max(normalized_bedrooms), 100)
normalized_floors_mesh, normalized_bedrooms_mesh = np.meshgrid(normalized_floors_range, normalized_bedrooms_range)

predicted_Y_mesh = [theta0 + theta1 * x1 + theta2 * x2 for x1, x2 in zip(normalized_floors_mesh.flatten(), normalized_bedrooms_mesh.flatten())]
predicted_Y_mesh = np.array(predicted_Y_mesh).reshape(normalized_floors_mesh.shape)

ax.plot_surface(normalized_floors_mesh, normalized_bedrooms_mesh, predicted_Y_mesh, cmap='viridis', alpha=0.8)

# Set labels and title
ax.set_xlabel('Normalized floors')
ax.set_ylabel('Normalized Bedrooms')
ax.set_zlabel('Y')
plt.title('3D Mesh Plot (Gradient Descent))')
plt.show()

#######################################################################################################
#######################################################################################################

theta_cf = closed_form_solution(normalized_floors,normalized_bedrooms,normalized_area,Y)
theta0, theta1,theta2,theta3=theta_cf

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for normalized_floors and normalized_bedrooms
ax.scatter(normalized_floors, normalized_bedrooms, Y, c='r', marker='o', label='Data Points')

normalized_floors_range = np.linspace(min(normalized_floors), max(normalized_floors), 100)
normalized_bedrooms_range = np.linspace(min(normalized_bedrooms), max(normalized_bedrooms), 100)
normalized_floors_mesh, normalized_bedrooms_mesh = np.meshgrid(normalized_floors_range, normalized_bedrooms_range)

predicted_Y_mesh = [theta0 + theta1 * x1 + theta2 * x2 for x1, x2 in zip(normalized_floors_mesh.flatten(), normalized_bedrooms_mesh.flatten())]
predicted_Y_mesh = np.array(predicted_Y_mesh).reshape(normalized_floors_mesh.shape)

ax.plot_surface(normalized_floors_mesh, normalized_bedrooms_mesh, predicted_Y_mesh, cmap='plasma', alpha=0.8)

# Set labels and title
ax.set_xlabel('Normalized floors')
ax.set_ylabel('Normalized Bedrooms')
ax.set_zlabel('Y')
plt.title('3D Mesh Plot (Closed-Form Solution))')
plt.show()

#######################################################################################################
#######################################################################################################

theta0, theta1, theta2, theta3, cost_history = gradient_descent(normalized_floors, normalized_bedrooms, normalized_area, Y, learning_rate, epochs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for normalized_floors and normalized_bedrooms
ax.scatter(normalized_floors, normalized_bedrooms, Y, c='r', marker='o', label='Data Points')

normalized_floors_range = np.linspace(min(normalized_floors), max(normalized_floors), 100)
normalized_bedrooms_range = np.linspace(min(normalized_bedrooms), max(normalized_bedrooms), 100)
normalized_floors_mesh, normalized_bedrooms_mesh = np.meshgrid(normalized_floors_range, normalized_bedrooms_range)

predicted_Y_mesh = [theta0 + theta1 * x1 + theta2 * x2 for x1, x2 in zip(normalized_floors_mesh.flatten(), normalized_bedrooms_mesh.flatten())]
predicted_Y_mesh = np.array(predicted_Y_mesh).reshape(normalized_floors_mesh.shape)

ax.plot_surface(normalized_floors_mesh, normalized_bedrooms_mesh, predicted_Y_mesh, cmap='viridis', alpha=0.8)

theta_cf = closed_form_solution(normalized_floors,normalized_bedrooms,normalized_area,Y)
theta0, theta1,theta2,theta3=theta_cf

normalized_floors_range = np.linspace(min(normalized_floors), max(normalized_floors), 100)
normalized_bedrooms_range = np.linspace(min(normalized_bedrooms), max(normalized_bedrooms), 100)
normalized_floors_mesh, normalized_bedrooms_mesh = np.meshgrid(normalized_floors_range, normalized_bedrooms_range)

predicted_Y_mesh = [theta0 + theta1 * x1 + theta2 * x2 for x1, x2 in zip(normalized_floors_mesh.flatten(), normalized_bedrooms_mesh.flatten())]
predicted_Y_mesh = np.array(predicted_Y_mesh).reshape(normalized_floors_mesh.shape)

ax.plot_surface(normalized_floors_mesh, normalized_bedrooms_mesh, predicted_Y_mesh, cmap='plasma', alpha=0.8)
# Set labels and title
ax.set_xlabel('Normalized Floors')
ax.set_ylabel('Normalized Bedrooms')
ax.set_zlabel('Y')
plt.title('Comparison Between GD and CFS')
plt.show()