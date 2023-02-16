from utils.kmeans_utils import *
from utils.vis_utils import *
import sys


def DCHWKmeans_vec(dataset, num_clusters, num_iterations, seed):

    loop_counter = 0

    centroids = init_centroids(dataset, num_clusters, seed)

    # Centroid status checker
    centroid_status = False
    new_assigned_clusters = np.zeros(shape=(len(dataset)))

    # Calculate the cluster assignments for data points
    assigned_clusters, distances = calculate_distances(dataset, centroids)
    new_assigned_clusters[:]= assigned_clusters[:]

    # Re-calculate the centroids
    new_centroids = calculate_centroids(dataset, assigned_clusters)

    for i in range(num_clusters):
        if len(np.where(assigned_clusters == i)[0]) == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            return centroids, loop_counter, sys.float_info.max, assigned_clusters


    while loop_counter<num_iterations:

        loop_counter += 1
        # print("Counter: ", loop_counter)
        # all_indices = []

        # Get current clustering state
        assign_dict, radius, cluster_info = get_membership(assigned_clusters, distances, num_clusters)
        neighbors, dist_mat = find_neighbors(new_centroids, radius)
        # neighbors, he_indices = find_all_he_indices_neighbor(dataset, new_centroids, radius, 
        #                                                                  assign_dict, cluster_info)
        # for k in he_indices.keys():
        #     all_indices += he_indices[k]

        # print("Counter", loop_counter)
        # print("Num of HE:", len(all_indices), "Cluster Size: ", cluster_info)

        for center in neighbors.keys():

            all_indices = []

            if check_centroid_status(center, new_centroids, centroids):

                centroid_status = True

                if cluster_info[center] > 1:

                    he_points = find_cluster_specific_he_points(dataset, neighbors, dist_mat, new_centroids, 
                             assign_dict, cluster_info, center)
                    
                    # he_points = list(he_indices[center])
                    if len(he_points) > 0:
                        all_indices += he_points

                        new_clus, sse123 = calculate_sse_specific(dataset[he_points],
                                                    new_centroids[neighbors[center]], 
                                                    neighbors[center], cluster_info, center)

                    # for index in range(len(he_indices_dict[center])):
                    #     point = he_indices_dict[center][index]
                    #     print("Point: ", point, "\t", assigned_clusters[point], "-->", temp[index],
                    #           " old dist: ", distances[point], " new distance: ", dist123[index])

                        distances[he_points] = sse123
                        new_assigned_clusters[he_points] = new_clus

            else:
                centroid_status = False

        # t  = np.where(assigned_clusters != new_assigned_clusters)[0]
        # print("Changed: ", len(t), "Size HE: ", len(all_indices))
        # print("Old: ", assigned_clusters[t], "New: ", new_assigned_clusters[t])
        if len(np.unique(new_assigned_clusters)) < num_clusters:
            print("DCHWKMeans: Found less modalities, safe exiting with current centroids.", loop_counter)
            return new_centroids, loop_counter, sys.float_info.max, assigned_clusters

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break

        # if len(all_indices) > 0:
        #     temp = np.where(assigned_clusters != new_assigned_clusters)[0]
        #     print("Size of changed: ", len(temp), " Predicted: ", len(all_indices))
        #     print("Data that actually changed membership: ", temp)
        #     print("Predicted: ", all_indices )
        #     for i in temp:
        #         print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])

        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        new_centroids = calculate_centroids(dataset, new_assigned_clusters)
        assigned_clusters[:] = new_assigned_clusters[:]


    sse = get_quality(dataset, assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, sse, assigned_clusters
