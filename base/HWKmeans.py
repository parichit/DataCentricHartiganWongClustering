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
    new_centroids = calculate_centroids(data, assigned_clusters)
    
    for i in range(num_clusters):
        if len(np.where(assigned_clusters == i)[0]) == 0:
            print("For ", num_clusters, " clusters. Intial centroids created empty partitions.")
            return centroids, loop_counter, sys.float_info.max, assigned_clusters


    while loop_counter<num_iterations:

        loop_counter += 1

        all_indices = []
        all_data_changed = []
        centroid_motion = []

        # for i in range(num_clusters):
        #     # centroid_motion.append(np.sqrt(np.sum(np.square(new_centroids[i] - centroids[i]))))
        #     centroid_motion.append(np.sum(np.square(new_centroids[i] - centroids[i])))
        # distances += centroid_motion
        
        # dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
        # cluster_info = get_size(assigned_clusters, num_clusters)

        dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
        assign_dict, radius, cluster_info = get_membership(assigned_clusters, distances, num_clusters)

        # assign_dict, radius = get_membership(assigned_clusters, distances, num_clusters)
        
        # if loop_counter == 1:
        #     print("Old Distances: ", distances[39, 1], distances[39, 3])
        #     # temp1 = np.sqrt(np.sum(np.square(new_centroids[1] - data[39])))
        #     # temp2 = np.sqrt(np.sum(np.square(new_centroids[3] - data[39])))
        #     temp1 = np.sum(np.square(new_centroids[1] - data[39]))
        #     temp2 = np.sum(np.square(new_centroids[3] - data[39]))
        #     print("Distance from new centroids: ", temp1, temp2)
        #     print("Cluster radius: ", radius[1], radius[3])
        #     print("half the radius: ", radius[1]/2, radius[3]/2)
        #     # temp3 = np.sqrt(np.sum(np.square(data[39] - ((new_centroids[1] + new_centroids[3])/2))))
        #     temp3 = np.sum(np.square(data[39] - ((new_centroids[1] + new_centroids[3])/2)))
        #     print("Dist from modpoint: ", temp3)

        #     print("Distance between new centroids: ", dist_mat[1, 3]/2)

        # if loop_counter == 1:
        #     print("Old Distances: ", distances[14, 1], distances[14, 3])
        #     print("Centroid motion: ", centroid_motion[1], centroid_motion[3])
        #     # temp1 = np.sqrt(np.sum(np.square(new_centroids[1] - data[14])))
        #     # temp2 = np.sqrt(np.sum(np.square(new_centroids[3] - data[14])))
        #     temp1 = np.sum(np.square(new_centroids[1] - data[14]))
        #     temp2 = np.sum(np.square(new_centroids[3] - data[14]))
        #     print("Distance from new centroids: ", temp1, temp2)
        #     print("Distance between new centroids: ", dist_mat[1, 3])
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True
                indices = np.where(assigned_clusters == curr_cluster)[0]

                if cluster_info[curr_cluster] > 1:

                    # if loop_counter == 4 and curr_cluster == 0:
                    #     print(len(indices), assigned_clusters[0])

                    # Compare the SSE with other clusters
                    sse = distance.cdist(data[indices, :], new_centroids, 'sqeuclidean')
                    sse_copy = np.zeros(shape=sse.shape)
                    sse_copy[:] = sse[:]
                    
                    sse[:, curr_cluster] = (len(indices) * sse[:, curr_cluster])/(len(indices)-1)
                    curr_sse = sse[:, curr_cluster]

                    # if curr_cluster == 1:
                    #     i = np.where(indices == 14)[0]
                    #     print("Loop: ", sse[i, curr_cluster], "\t", sse[i, 3])
                    
                    # all_indices = []
                    # he_indices = find_he(data, new_centroids[curr_cluster], dist_mat, indices, 
                    # curr_cluster, num_clusters, cluster_info)
                    
                    # he_indices = find_he_by_radius(data, new_centroids[curr_cluster], radius, indices, num_clusters)

                    # _, he_indices_dict = find_all_he_indices_neighbor(data, new_centroids, radius,
                    #                             assign_dict)
                    # for i in he_indices_dict.keys():
                    #     if len(he_indices_dict[i]) > 0:
                    #         all_indices += list(he_indices_dict[i])

                    he_indices, _, _ = find_he_new(data, new_centroids[curr_cluster], dist_mat, indices, 
                        curr_cluster, num_clusters, cluster_info, distances)
                    
                    all_indices += list(he_indices)
                    data_changed = []
                
                    for ot_cluster in range(num_clusters):
                            
                        if ot_cluster != curr_cluster:
                                
                            size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])
                            sse[:, ot_cluster] = (size_ot_cluster * sse[:, ot_cluster])/(size_ot_cluster+1)

                            ot_sse = sse[:, ot_cluster]

                            # if ot_cluster == 3:
                            #     i = np.where(indices == 14)[0]
                            #     print("other: ", size_ot_cluster/(size_ot_cluster+1), "\t", sse[i, 3])

                            temp_indices = np.where(ot_sse < curr_sse)[0]
                            
                            if len(temp_indices) > 0:
                                data_changed += list(temp_indices)

                    # Update cluster membership for data point
                    if len(data_changed) > 0 :
                        data_changed = np.unique(data_changed)
                        new_clus = np.argmin(sse[data_changed, :], axis=1)
                        temp = indices[data_changed].tolist()
                        
                        # assigned_clusters[temp] = new_clus
                        new_assigned_clusters[temp] = new_clus
                        # print(new_assigned_clusters[temp])

                        # min_dist = np.sqrt(np.min(sse_copy[data_changed, :], axis=1))
                        min_dist = np.min(sse_copy[data_changed, :], axis=1)
                        distances[temp] = min_dist

                        # print("Loop counter: ", loop_counter, " Current: ", curr_cluster, " Other: ", ot_cluster)
                        # print("Predicted: ", he_indices, " Actual Change: ", temp)
                        
                        all_data_changed += temp

                # break
            else:
                centroid_status = False

        
        if len(np.unique(assigned_clusters)) < num_clusters:
            print("HWKMeans: Found less modalities, safe exiting with current centroids.")
            return new_centroids, loop_counter, sys.float_info.max, assigned_clusters
            
        if len(all_data_changed) > 0:
            # print("Loop Counter: ", loop_counter)
            print("\nSize of changed: ", len(all_data_changed), " Predicted: ", len(all_indices))
            # print("Data that actually changed it's membership: ", all_data_changed)
            # print("Predicted:", all_indices)
            for i in all_data_changed:
                if i not in all_indices:
                    print("#############################")
                    print("Loop Counter: ", loop_counter)
                    print(i, " not found in Prediction")
                    print("#############################")

            #         new_clus =  int(new_assigned_clusters[i])
            #         old_clus =  assigned_clusters[i]
            #         old_fac = cluster_info[old_clus]/(cluster_info[old_clus]-1)
            #         new_fac = cluster_info[new_clus]/(cluster_info[new_clus]+1)
            #         temp1 = np.sum(np.square(data[i, :] - new_centroids[new_clus, :]))
            #         temp2 = np.sum(np.square(data[i, :] - new_centroids[assigned_clusters[i], :]))
            #         ntemp1 = (temp1 * cluster_info[new_clus])/(cluster_info[new_clus]+1)
            #         ntemp2 = (temp2 * cluster_info[old_clus])/(cluster_info[old_clus]-1)
            #         temp3 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[new_clus, :])))
            #         temp4 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[assigned_clusters[i], :])))
            #         print("Data: ", i, " Old cluster: ", old_clus, "/", old_fac , " New Cluster: ", new_clus, "/", new_fac)
            #         print("Old SSE:", temp2, "/", ntemp2, " New SSE: ", temp1, "/", ntemp1, 
            #           "Old Dist: ", temp4, " New Dist: ", temp3)
                    break

            # for i in all_data_changed:
            #     print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])

            # for i in all_data_changed:
            #     new_clus =  int(new_assigned_clusters[i])
            #     old_clus =  assigned_clusters[i]
            #     old_fac = cluster_info[old_clus]/(cluster_info[old_clus]-1)
            #     new_fac = cluster_info[new_clus]/(cluster_info[new_clus]+1)
            #     temp1 = np.sum(np.square(data[i, :] - new_centroids[new_clus, :]))
            #     temp2 = np.sum(np.square(data[i, :] - new_centroids[assigned_clusters[i], :]))
            #     ntemp1 = (temp1 * cluster_info[new_clus])/(cluster_info[new_clus]+1)
            #     ntemp2 = (temp2 * cluster_info[old_clus])/(cluster_info[old_clus]-1)
            #     temp3 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[new_clus, :])))
            #     temp4 = np.sqrt(np.sum(np.square(data[i, :] - new_centroids[assigned_clusters[i], :])))
            #     print("Data: ", i, " Old cluster: ", old_clus, "/", old_fac , " New Cluster: ", new_clus, "/", new_fac)
            #     print("Old SSE:", temp2, "/", ntemp2, " New SSE: ", temp1, "/", ntemp1, 
            #           "Old Dist: ", temp4, " New Dist: ", temp3)


            # vis_data_with_he(data, new_centroids, assigned_clusters, distances, loop_counter, all_data_changed, [])

            # vis_data_with_he_test(data, new_centroids, assigned_clusters, radius, loop_counter, all_data_changed, [])
        
        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break

        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        new_centroids = calculate_centroids(data, new_assigned_clusters)
        assigned_clusters[:] = new_assigned_clusters[:]
        # _, distances = calculate_distances(data, new_centroids)

    sse = get_quality(data, assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter, sse, assigned_clusters
