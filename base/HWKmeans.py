from utils.kmeans_utils import *
from utils.vis_utils import *
from scipy.spatial import distance
import sys


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
    new_assigned_clusters = np.zeros(shape=(len(data)))

    # Calculate the cluster assignments for data points
    assigned_clusters, distances = calculate_distances(data, centroids)
    new_assigned_clusters[:] = assigned_clusters[:]

    # Re-calculate the centroids
    new_centroids = calculate_centroids(data, centroids, assigned_clusters, num_clusters)
    
    for i in range(num_clusters):
        if len(np.where(assigned_clusters == i)[0]) == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            return centroids, loop_counter, sys.float_info.max, assigned_clusters


    while loop_counter<num_iterations:

        loop_counter += 1
        all_indices = []

        # for i in range(num_clusters):
        #     # centroid_motion.append(np.sqrt(np.sum(np.square(new_centroids[i] - centroids[i]))))
        #     centroid_motion.append(np.sum(np.square(new_centroids[i] - centroids[i])))
        # distances += centroid_motion

        # print(loop_counter)
        # dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
        assign_dict, radius, cluster_info = get_membership(assigned_clusters, distances, num_clusters)
        # print(radius[3], dist_mat[3, 2]/2, radius[2])

        # if loop_counter >= 4:
        #     print(dist_mat)
        #     print(new_centroids, loop_counter)
        #     for i in range(len(cluster_info)):
        #         print(cluster_info[i], len(assign_dict[i]))

        # neighbors, he_indices = find_all_he_indices_neighbor(data, new_centroids, radius, 
        #                                                                  assign_dict, cluster_info)
        # for k in he_indices.keys():
        #     all_indices += he_indices[k]

        # print("Counter", loop_counter)
        # print("Num of HE:", len(all_indices), "Cluster Size: ", cluster_info)
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True  

                # Compare the SSE with other clusters
                if cluster_info[curr_cluster] > 1:  
                    
                    indices = assign_dict[curr_cluster]

                    # all_indices += list(find_he_new(data, new_centroids, dist_mat, indices, 
                    #                             curr_cluster, num_clusters, cluster_info))
                    

                    sse = calculate_sse(data[indices, :], new_centroids)

                    my_size = cluster_info[curr_cluster]/(cluster_info[curr_cluster]-1)
                    sse[:, curr_cluster] = sse[:, curr_cluster] * my_size

                    for ot_cluster in range(num_clusters):
                            
                        if ot_cluster != curr_cluster:
                            
                            ot_size = cluster_info[ot_cluster]/(cluster_info[ot_cluster]+1)
                            
                            #if cluster_info[ot_cluster] > 1:
                            sse[:, ot_cluster] = sse[:, ot_cluster] * ot_size

                    new_assigned_clusters[indices] = np.argmin(sse, axis=1)
                    distances[indices] = np.min(sse, axis=1)

            else:
                centroid_status = False

        # t  = np.where(assigned_clusters != new_assigned_clusters)[0]
        # print("Counter:", loop_counter)
        # print("Num change: ", len(t), "Data Changed: ", t)
        # print("Old: ", assigned_clusters[t], "New: ", new_assigned_clusters[t])
        if len(all_indices) > 0:
            temp = np.where(assigned_clusters != new_assigned_clusters)[0] 
            all_indices = list(np.unique(all_indices))
            # print("Loop Counter: ", loop_counter)
            # print("\nSize of changed: ", len(temp), " Predicted: ", len(all_indices))
            # print("Data that actually changed it's membership: ", temp)
            # print("Predicted:", all_indices)
            
            for i in temp:
                if i not in all_indices:
                    print("#############################")
                    print("Loop Counter: ", loop_counter)
                    print(i, " not found in Prediction")
                    print("#############################")
            
            #         new_clus =  int(new_assigned_clusters[i])
            #         old_clus =  assigned_clusters[i]
            #         old_fac = cluster_info[old_clus]/(cluster_info[old_clus]-1)
            #         new_fac = cluster_info[new_clus]/(cluster_info[new_clus]+1)
            #         temp2 = np.sum(np.square(data[i, :] - new_centroids[old_clus, :]))
            #         ntemp2 = temp2 * old_fac
            #         temp1 = np.sum(np.square(data[i, :] - new_centroids[new_clus, :]))
            #         ntemp1 = temp1 * new_fac
            #         temp3 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[new_clus, :])))
            #         temp4 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[old_clus, :])))
            #         print("Data: ", i, " Old cluster: ", old_clus, "/", old_fac , " New Cluster: ", new_clus, "/", new_fac)
            #         print("Old SSE:", temp2, "/", ntemp2, " New SSE: ", temp1, "/", ntemp1, 
            #           "Old Dist: ", temp4, " New Dist: ", temp3)
            #         print("distance between the centroids: ", dist_mat[old_clus, new_clus])
            #         print("Old Size: ", cluster_info[old_clus], " New Size: ", cluster_info[new_clus])
            #         centroid_status = False
            #         # vis_data_with_he(data, new_centroids, neighbors, assigned_clusters, radius, loop_counter, [i], [])
                    break

            # for i in temp:
            #     print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])
            # 
            # vis_data_with_he_test(data, new_centroids, neighbors, assigned_clusters, radius, loop_counter, temp, all_indices)
        
        if len(np.unique(new_assigned_clusters)) < num_clusters:
            print("HWKMeans: Found less modalities, safe exiting with current centroids.")
            return new_centroids, loop_counter, sys.float_info.max, new_assigned_clusters
        
        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter, "\n")
            break
        
        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        new_centroids = calculate_centroids(data, new_centroids, new_assigned_clusters, num_clusters)
        assigned_clusters[:] = new_assigned_clusters[:]       

    sse = get_quality(data, assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, sse, assigned_clusters
