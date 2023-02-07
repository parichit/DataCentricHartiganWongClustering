from utils.kmeans_utils import *
from utils.vis_utils import *
from scipy.spatial import distance

def HWKmeans_test123(data, num_clusters, num_iterations, seed):

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
    cluster_info = get_size(assigned_clusters, num_clusters)

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, assigned_clusters)
    
    for i in cluster_info:
        if i == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            exit("Exiting")

    while loop_counter<num_iterations:

        loop_counter += 1
        print("\n Counter: ", loop_counter)
        # print(new_centroids)

        # Find the neighbors of current cluster and iterate over the neighbors
        assign_dict, radius = get_membership(assigned_clusters, distances, num_clusters, assign_dict)

        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True

                he_sse, he_indices = find_all_points_test(data, curr_cluster, new_centroids[curr_cluster, :], radius, assign_dict[curr_cluster])
                
                # if loop_counter == 1:
                #     print("Center: ", curr_cluster, " HE: ", he_indices)

                indices = np.where(assigned_clusters == curr_cluster)[0]
                curr_cluster_info = len(indices)

                # Compare the SSE with other clusters
                sse = distance.cdist(data[indices, :], new_centroids, 'sqeuclidean')

                # print(he_sse.shape, "\nHello\n", sse[:, curr_cluster].shape, len(indices), type(sse), type(he_sse))
                
                # if (np.round(he_sse, 2).any() != np.round(sse[:, curr_cluster], 2).any()):
                #     print("SSE mismatch for clusters: ", curr_cluster)
                
                if curr_cluster_info > 1:
                    curr_sse = (curr_cluster_info * sse[:, curr_cluster])/(curr_cluster_info-1)
                else:
                    curr_sse = sse[:, curr_cluster]

                temp = []

                for ot_cluster in range(num_clusters):
                        
                        if curr_cluster != ot_cluster:
                            
                            size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])
                            ot_sse = (size_ot_cluster * sse[:, ot_cluster])/(size_ot_cluster+1)

                            temp_indices = np.where(ot_sse < curr_sse)[0]

                            # Update the cluster membership for the data point
                            if len(temp_indices) > 0:
                                he_data_indices = []
                                temp_indices2 = indices[temp_indices].tolist()
                                # print(len(temp_indices), len(temp_indices2))
                                new_assigned_clusters[temp_indices2] = ot_cluster
                                he_data_indices += temp_indices2
                            
                            # elif len(temp_indices) == 1:


                                for i in he_data_indices:
                                    if i not in he_indices:
                                        # print("Predicted HE nor found in actual data that changed it's memberhsip")
                                        # print("Current: ", curr_cluster, " Other: ", ot_cluster)
                                        temp += [i]
                                
                                if len(temp) >0:
                                    print(curr_cluster_info, size_ot_cluster, curr_cluster, ot_cluster)
                                    print(len(he_indices), len(he_data_indices), len(temp))

                                    vis_data_with_he_test(data, centroids, assigned_clusters, distances, [curr_cluster, ot_cluster],
                                    loop_counter, he_data_indices, he_indices)
                                    loop_counter += num_iterations
                                    break

            else:
                centroid_status = False

            # if len(temp)>0:
            #     break
            
        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        assigned_clusters[:] = new_assigned_clusters[:]
        new_centroids = calculate_centroids(data, assigned_clusters)
        _, distances = calculate_distances(data, new_centroids)

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break

        if loop_counter >= num_clusters:
            break
            
    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, assigned_clusters