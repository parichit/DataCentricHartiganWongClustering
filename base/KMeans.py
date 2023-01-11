from utils.kmeans_utils import *
import sys


def Kmeans(data, num_clusters, threshold, num_iterations, centroids, seed):

    loop_counter = 0

    if centroids is []:
        centroids = init_centroids(data, num_clusters, seed)
        print("Initialized centroids manually")

    # Calculate the cluster assignments for data points
    # assigned_clusters, _, stat = calculate_distances_less_modalities(data, centroids)
    old_assigned_clusters, _ = calculate_distances(data, centroids)

    if len(np.unique(old_assigned_clusters)) < num_clusters:
        print("KMeans: Found less modalities, safe exiting with current centroids.")
        return centroids, loop_counter, sys.float_info.max, data.shape[0]*num_clusters

    while loop_counter < num_iterations:

        loop_counter += 1
        # print("Counter: ", loop_counter)

        # Re-calculate the centroids
        new_centroids = calculate_centroids(data, old_assigned_clusters)

        if check_convergence(new_centroids, centroids, threshold):
            print("Kmeans: Convergence at iteration: ", loop_counter)
            break

        # Calculate the cluster assignments for data points
        new_assigned_clusters, _ = calculate_distances(data, new_centroids)

        # if (new_assigned_clusters == old_assigned_clusters).all():
        #     print("Kmeans: Convergence at iteration: ", loop_counter)
        #     break

        # if len(np.unique(new_assigned_clusters)) < num_clusters:
        #     print("KMeans: Found less modalities, safe exiting with current centroids.")
        #     return centroids, loop_counter, sys.float_info.max, data.shape[0]*loop_counter*num_clusters

        # Calculate the cluster assignments for data points
        centroids[:] = new_centroids[:]
        old_assigned_clusters[:] = new_assigned_clusters[:]

    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, data.shape[0]*loop_counter*num_clusters


