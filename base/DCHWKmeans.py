from utils.kmeans_utils import *
from utils.vis_utils import *
from scipy.spatial import distance


def DCHWKmeans(data, num_clusters, num_iterations, seed):

    loop_counter = 0

    centroids = init_centroids(data, num_clusters, seed)
    print("Initialized centroids manually")

    # Centroid status checker
    centroid_status = False
    dist_mat = np.zeros((num_clusters, num_clusters))

    # Calculate the cluster assignments for data points
    assigned_clusters, distances = calculate_distances(data, centroids)
    cluster_size = get_size(assigned_clusters, num_clusters)

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, assigned_clusters)
    assign_dict = {}

    for i in cluster_size:
        if i == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            exit("Exiting")

    while loop_counter<num_iterations:

        loop_counter += 1
        # print("\n", loop_counter)
        # print(new_centroids)

        # Find the neighbors of current cluster and iterate over the neighbors
        assign_dict, radius = get_membership(assigned_clusters, distances, num_clusters, assign_dict)
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True
                sse, he_indices = find_all_points_test(data, curr_cluster, new_centroids[curr_cluster], radius, assign_dict[curr_cluster])
                
                if len(he_indices) > 0:
                    
                    assigned_clusters, distances = calculate_sse_specific(data, new_centroids, cluster_size, he_indices, assigned_clusters, curr_cluster, sse, distances)

            else:
                centroid_status = False


        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        new_centroids = calculate_centroids(data, assigned_clusters)
        cluster_size = get_size(assigned_clusters, num_clusters)

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break

    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, assigned_clusters
