#!/usr/bin/env python3
import numpy as np
import functools
import matplotlib.pyplot as plt


def distance(X, Y):
    return np.sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)


def sort_distances(item1, item2):
    if item1[1] < item2[1]:
        return -1
    elif item1[1] == item2[1]:
        return 0
    else:
        return 1


def main():
    Rn = int(input("Enter number of reference points: "))
    Qn = int(input("Enter number of Query points: "))
    k = int(input("Enter the value of k: "))

    R = []
    reference_txt = "Enter the {value}th example in the Reference points. "
    for i in range(Rn):
        Xi, Yi = input(reference_txt.format(value=i)).split()

        point = (int(Xi), int(Yi))
        R.append(point)

    print(R)
    Q = []
    query_txt = "Enter the {value}th example in the query points. "
    for i in range(Qn):
        Xi, Yi = input(query_txt.format(value=i)).split()

        point = (int(Xi), int(Yi))
        Q.append(point)

    print(Q)

    for q in Q:
        distances = []
        for i in range(len(R)):
            distances.append([i, distance(q, R[i])])

        print("unsorted Distances: ", distances)
        # distances.sort(key=sort_distances)
        distances = sorted(distances, key=functools.cmp_to_key(sort_distances))
        print("Sorted distances:", distances)

        k_nearest_neighbours = [R[distances[i][0]] for i in range(k)]
        print("k_nearest_neighbours:",  k_nearest_neighbours)

    # plt.scatter(R[0], R[1])
    # plt.scatter(Q[0], Q[1])
    plt.scatter(*zip(*R))
    plt.scatter(*zip(*Q))
    plt.show()


if __name__ == "__main__":
    main()
