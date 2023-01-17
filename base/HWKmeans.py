from utils.kmeans_utils import *
from utils.vis_utils import *
from scipy.spatial import distance


def HWKmeans_test(data, num_clusters, num_iterations, seed):

    loop_counter = 0

    centroids = init_centroids(data, num_clusters, seed)
    print("Initialized centroids manually")

    # Centroid status checker
    centroid_status = False
    new_assigned_clusters = np.zeros(shape=(len(data)))

    # Calculate the cluster assignments for data points
    assigned_clusters, _ = calculate_distances(data, centroids)
    new_assigned_clusters[:] = assigned_clusters[:]

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, assigned_clusters)

    while loop_counter<num_iterations:

        he_data_indices = []
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True
                # print("centroids updated: ", centroid_status)
                data_indices = np.where(assigned_clusters == curr_cluster)[0]
               
                # Compare the SSE with other clusters
                for index in data_indices:
                    lowest_sse = -999
                    sse1 = (len(data_indices) * np.sum(np.square(data[index, :] - new_centroids[curr_cluster, :])))/(len(data_indices)-1)
                
                    for ot_cluster in range(num_clusters):

                        if curr_cluster != ot_cluster:
                            
                            size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])

                            if sse_after_move(data, new_centroids, sse1, lowest_sse, index, curr_cluster, ot_cluster, size_ot_cluster):
                                                                                                        
                                # Update the cluster membership for the data point
                                new_assigned_clusters[index] = ot_cluster
                                he_data_indices.append(index)

            else:
                centroid_status = False
        
        # _, distances = calculate_distances(data, new_centroids)
        for i in he_data_indices:
            print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])
        # vis_data_with_he(data, new_centroids, assigned_clusters, distances,
        #                  loop_counter, he_data_indices, he_data_indices)

        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        assigned_clusters[:] = new_assigned_clusters[:]
        new_centroids = calculate_centroids(data, assigned_clusters)

        # print("Size of centroids: ", len(centroids), "\t", len(new_centroids))

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break

        loop_counter += 1
        print("\nCounter: ", loop_counter)

    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter


def HWKmeans(data, num_clusters, num_iterations, seed):

    loop_counter = 0

    centroids = init_centroids(data, num_clusters, seed)

    # Centroid status checker
    centroid_status = False
    dist_mat = np.zeros((num_clusters, num_clusters))
    new_assigned_clusters = np.zeros(shape=(len(data)))
    assign_dict = {}

    # Calculate the cluster assignments for data points
    assigned_clusters, distances = calculate_distances(data, centroids)
    new_assigned_clusters[:] = assigned_clusters[:]
    cluster_size = get_size(assigned_clusters, num_clusters)

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, assigned_clusters)
    
    for i in cluster_size:
        if i == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            exit("Exiting")
    

    while loop_counter<num_iterations:

        loop_counter += 1
        # print("\n", loop_counter)
        # print(new_centroids)

        # Find the neighbors of current cluster and iterate over the neighbors
        # assign_dict, radius = get_membership(assigned_clusters, distances, num_clusters, assign_dict)

        # print("Counter: ", loop_counter, " He Data:")
        # for i in he_indices_dict.keys():
        
        # for i in neighbors.keys():
        #     print(i, " --> ", neighbors[i])
        # print("Points assigned to the current cluster")
        # print(new_centroids)
        # print(data[38])
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True

                indices = np.where(assigned_clusters == curr_cluster)[0]
                curr_cluster_size = len(indices)

                # Compare the SSE with other clusters
                sse = distance.cdist(data[indices, :], new_centroids, 'sqeuclidean')
                
                if curr_cluster_size > 1:
                    curr_sse = (curr_cluster_size * sse[:, curr_cluster])/(curr_cluster_size-1)
                else:
                    curr_sse = sse[:, curr_cluster]
                
                for ot_cluster in range(num_clusters):
                        
                        if curr_cluster != ot_cluster:
                            
                            size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])
                            ot_sse = (size_ot_cluster * sse[:, ot_cluster])/(size_ot_cluster+1)

                            temp_indices = np.where(ot_sse < curr_sse)[0]
                            
                            # Update the cluster membership for the data point
                            if len(temp_indices) > 0:
                                he_data_indices = []
                                temp_indices2 = indices[temp_indices].tolist()
                                new_assigned_clusters[temp_indices2] = ot_cluster
                                he_data_indices += temp_indices2
                            
            else:
                centroid_status = False
            
        # if len(he_data_indices) > 0 and loop_counter == 1:
        #     print("Data that actually changed it's membership: ", he_data_indices)
        #     for i in he_data_indices:
        #         print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])
        #     vis_data_with_he(data, new_centroids, assigned_clusters, distances,
        #                  loop_counter, he_data_indices, he_data_indices)

        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        assigned_clusters[:] = new_assigned_clusters[:]
        new_centroids = calculate_centroids(data, assigned_clusters)
        _, distances = calculate_distances(data, new_centroids)

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break
            
    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, assigned_clusters
