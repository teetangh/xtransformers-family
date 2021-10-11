#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def main():
    data = []
    centroids = []

    num_data = int(input("Please enter the number of points for the input"))
    for i in range(num_data):
        print(f"Enter the point {i} ", sep="")
        data.append([float(x) for x in input().split()])
    print("The points in the dataset are:", data)

    num_centroids = int(
        input("Please enter the number of centroids"))
    for i in range(num_centroids):
        centroids.append([np.random.randint(1, 10), np.random.randint(1, 10)])
    print("inital clusters are: ", centroids)

    cluster_assigned = [np.random.randint(1, 10) for _ in range(num_data)]
    # while(True):
    for _ in range(50):
        for i in range(num_data):
            distances = []
            distances = map(lambda y: calculate_distance(
                data[i], y),  centroids)
            distances = list(distances)
            print("Point", i, " has distances", distances)
            cluster_assigned[i] = np.argmin(list(distances))

        print("Clusters assigned: ", cluster_assigned, "\n")

        for j in range(num_centroids):

            cluster_points = []
            # cluster_points = filter(lambda x: x , cluster_assigned)

            for i in range(num_data):
                # print(" debug1  ", cluster_assigned[i], j)
                if cluster_assigned[i] == j:
                    cluster_points.append(data[i])

            # print(" debug2  ", cluster_points)
            if len(cluster_points) != 0:
                centroids[j] = np.sum(
                    cluster_points, axis=0)/len(cluster_points)
        print("New Centroids are ", centroids, "\n")
        # break

    # temp = [int(x) for x in range(len(centroids))]
    # print(cluster_assigned)
    # print(temp)

    plt.scatter(*zip(*data), marker="o", c=cluster_assigned)
    plt.scatter(*zip(*centroids), marker="x",
                c=[int(x) for x in range(len(centroids))])
    plt.show()
    plt.close(3)
    # fig = plt.figure()
    # plt.xlim(0, 20)
    # plt.ylim(0, 20)


if __name__ == '__main__':
    print(calculate_distance([1, 1], [2, 2]))
    main()
