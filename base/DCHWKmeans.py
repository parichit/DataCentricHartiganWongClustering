from utils.kmeans_utils import *
from utils.vis_utils import *
import sys


def DCHWKmeans(data, num_clusters, num_iterations, seed):

    loop_counter = 0

    centroids = init_centroids(data, num_clusters, seed)

    # Centroid status checker
    centroid_status = False
    new_assigned_clusters = np.zeros(shape=(len(data)))

    # Calculate the cluster assignments for data points
    assigned_clusters, distances = calculate_distances(data, centroids)
    new_assigned_clusters[:]= assigned_clusters[:]

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, assigned_clusters)

    for i in range(num_clusters):
        if len(np.where(assigned_clusters == i)[0]) == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            return centroids, loop_counter, sys.float_info.max, assigned_clusters


    while loop_counter<num_iterations:

        loop_counter += 1
        # print("Counter: ", loop_counter)
        all_indices = []

        # Find inter-centroid ddist matrix
        dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
        assign_dict, radius, cluster_info = get_membership(assigned_clusters, distances, num_clusters)

        # if loop_counter >= 4:
        #     print(dist_mat)
        #     print(new_centroids, loop_counter)
        #     for i in range(len(cluster_info)):
        #         print(cluster_info[i], len(assign_dict[i]))

        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):
                
                centroid_status = True

                if cluster_info[curr_cluster] > 1: 

                    indices = assign_dict[curr_cluster]
                    
                    he_indices = find_he_new(data, new_centroids[curr_cluster], dist_mat, indices, 
                        curr_cluster, num_clusters, cluster_info)
                    
                    if he_indices != []:
                        
                        all_indices += list(he_indices)
                        new_clus = calculate_sse_specific(data, new_centroids, he_indices, cluster_info,
                        assigned_clusters, curr_cluster)

                        assigned_clusters[he_indices] = new_clus

            else:
                centroid_status = False


        if len(np.unique(assigned_clusters)) < num_clusters:
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
        new_centroids = calculate_centroids(data, assigned_clusters)
        # assigned_clusters[:] = new_assigned_clusters[:]


    sse = get_quality(data, assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, sse, assigned_clusters
