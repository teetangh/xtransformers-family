#!/usr/bin/env python3
import numpy as np
import math
import random


def compute_z(theta, x):
    z = 0
    for j in range(len(x)):
        z += theta[j] * x[j]
    z += theta[len(x)]
    return z


def compute_g(z):
    return (1)/(1 + math.exp(-z))


def compute_h(z):
    return compute_g(z)


def binary_cross_entropy_loss(Y_train, Y_predict):

    total = 0
    for i in range(len(Y_train)):
        total -= (Y_train[i] * math.log(Y_predict[i])) + \
            ((1 - Y_train[i]) * math.log(1-Y_predict[i]))

    average = total / len(Y_train)
    return average


def compute_loss_gradients(theta, X_train, Y_train, Y_predict):
    delta_theta = []

    for j in range(len(X_train[0])):
        grad = 0
        for i in range(len(Y_train)):
            grad += ((Y_predict[i] - Y_train[i]) * X_train[i][j])/len(Y_train)

        delta_theta.append(grad)

    return delta_theta


def main():
    # f = int(input("no of features: "))
    n = int(input("no of rows: "))

    X_train = []
    Y_train = []
    for i in range(n):
        row = [int(r) for r in input().split()]

        X_train.append(row[0:-1])
        Y_train.append(row[-1])

    theta = [np.random.randn() for i in range(len(X_train))]
    print("theta", theta)
    for i in range(n):
        print(X_train[i], Y_train[i])

    epochs = 5
    epsilon = 0.00000000000000001
    alpha = 0.001
    for e in range(epochs):
        Y_predict = []
        for i in range(n):
            print(X_train[i])
            Y_predict.append(compute_h(compute_z(theta, X_train[i])))

        current_loss = binary_cross_entropy_loss(Y_train, Y_predict)
        print("=========> Epoch number:", e, "Current Loss: ", current_loss)
        print("Y_predict", Y_predict)
        if current_loss <= epsilon:
            break

        delta_theta = compute_loss_gradients(
            theta, X_train, Y_train, Y_predict)
        print("delta_theta", delta_theta)
        for j in range(len(theta) - 1):
            theta[j] = theta[j] - alpha * delta_theta[j]


if __name__ == "__main__":
    main()
