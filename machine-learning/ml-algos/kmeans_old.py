#!/usr/bin/env python3
# Input Process(architecture/design/workflow) Output => helps in coding
import numpy as np
import random
import sys


def compute_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    En = int(input("enter the number of entities"))
    E = []
    for i in range(En):
        text = "Enter the {value}th point"
        E.append(list([int(x) for x in input(text.format(value=i)).split()]))

    k, MaxIters = [int(x) for x in input("Enter k and MaxIters: ").split()]

    print("\nE", E)
    print("k", k)
    print("MaxIters", MaxIters)

    cluster_centroids = []
    for _ in range(k):
        cluster_centroids.append(E[random.randrange(0, En, 1)])

    print("cluster_centroids ", cluster_centroids)

    labels = []
    for i in range(len(E)):
        nearest_distance = 99999999999
        nearest_cluster = [-9999, -9999]
        for j in range(len(cluster_centroids)):
            current_distance = compute_distance(E[i], cluster_centroids[j])
            if current_distance < nearest_distance:
                nearest_cluster = cluster_centroids[j]
                nearest_distance = current_distance
                print(i, j)
        labels.append(nearest_cluster)
    print("labels", labels)

    changed = False
    iteration = 0

    while True:
        for j in range(len(cluster_centroids)):
            centroid_new = [0, 0]
            cluster_vertices = 0
            for i in range(len(labels)):
                if labels[i] == cluster_centroids[j]:
                    centroid_new[0] += E[i][0]
                    centroid_new[1] += E[i][1]
                    cluster_vertices += 1

            if cluster_vertices != 0:
                centroid_new[0] /= cluster_vertices
                centroid_new[1] /= cluster_vertices
                # round(centroid_new[0],3 )
                # round(centroid_new[1],3 )
                cluster_centroids[j] = [centroid_new[0], centroid_new[1]]
            print("iteration: ", iteration,
                  "cluster_centroids", cluster_centroids)
            iteration += 1

            changed = True
        if changed == False or iteration > MaxIters:
            break


if __name__ == "__main__":
    sys.stdin = open("kmeans_input.txt", "r")
    # sys.stdout = open("kmeans_output.txt", "w+")
    main()
