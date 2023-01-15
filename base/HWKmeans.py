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
        loop_counter += 1
        
        for curr_cluster in range(num_clusters):

            if check_centroid_status(curr_cluster, new_centroids, centroids):

                centroid_status = True

                indices = np.where(assigned_clusters == curr_cluster)[0]
                curr_cluster_size = len(indices)

                # Compare the SSE with other clusters
                sse = distance.cdist(data[indices, :], new_centroids, 'sqeuclidean')
                curr_sse = (curr_cluster_size * sse[:, curr_cluster])/(curr_cluster_size-1)
                
                for ot_cluster in range(num_clusters):
                        
                        if curr_cluster != ot_cluster:
                            
                            size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])
                            ot_sse = (size_ot_cluster * sse[:, ot_cluster])/(size_ot_cluster+1)

                            temp_indices = np.where(ot_sse <= curr_sse)[0]

                            # if curr_cluster == 2:
                            #     print(curr_sse)
                            #     print(ot_sse)
                            
                            # Update the cluster membership for the data point
                            if len(temp_indices) > 0:
                                # print(temp_indices)
                                temp_indices2 = indices[temp_indices].tolist()
                                new_assigned_clusters[temp_indices2] = ot_cluster
                                he_data_indices += temp_indices2

            else:
                centroid_status = False
        
        # if len(he_data_indices) > 0:
        #     # print("hey: ", he_data_indices)
        #     _, distances = calculate_distances(data, new_centroids)
        #     for i in he_data_indices:
        #         print("Point: ", i, " Old center: ", assigned_clusters[i], "\t", "new center: ", new_assigned_clusters[i])
            # vis_data_with_he(data, new_centroids, assigned_clusters, distances,
            #              loop_counter, he_data_indices, he_data_indices)

        # Re-calculate the centroids
        centroids[:] = new_centroids[:]
        assigned_clusters[:] = new_assigned_clusters[:]
        new_centroids = calculate_centroids(data, assigned_clusters)

        # print("Size of centroids: ", len(centroids), "\t", len(new_centroids))

        if centroid_status == False:
            print("Convergence at iteration: ", loop_counter)
            break


    # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
    return new_centroids, loop_counter


# def op_trans(data, new_centroids, assigned_clusters, new_assigned_clusters, live_set, loop_counter):

#     new_live_set = []

#     for i in range(len(data)):
        
#         curr_cluster = assigned_clusters[i, 0]

#         # if cluster is in LIVE set do 4(a)
#         if curr_cluster in live_set:

#             sse = np.sum(np.square(data[i, :] - new_centroids), 1)
            
#             my_size = len(np.where(assigned_clusters[: 0] == curr_cluster))
#             my_sse = (my_size*sse[curr_cluster])/(my_size-1)

#             wch_min_sse = np.argmin(sse)
#             sec_sse = sse[assigned_clusters[i, 1]]

#             ot_size = len(np.where(assigned_clusters[: 0] == wch_min_sse))
#             ot_sse = (ot_size*sse[wch_min_sse])/(ot_size+1)
            
#             if ot_sse < my_sse:
#                 new_assigned_clusters[i, 0] = int(wch_min_sse)
#                 # new_assigned_clusters[i, 1] = curr_cluster
#                 new_live_set += [curr_cluster, int(wch_min_sse)]

#                 if my_sse < sec_sse:
#                     new_assigned_clusters[i, 1] = curr_cluster
#                     # new_live_set += [assigned_clusters[i, 1]]
            
#             if ot_sse < sec_sse:
#                 new_assigned_clusters[i, 1] = int(wch_min_sse)
#                 # new_live_set += [assigned_clusters[i, 1], int(wch_min_sse)]


#         # If cluster is not in LIVE set do 4(b)                
#         else:
#             # Loop only over LIVE set
#             sse = np.sum(np.square(data[i, :] - new_centroids[live_set, :]), 1)
#             wch_min_sse = np.argmin(sse)
#             ot_clus = live_set[wch_min_sse]

#             my_size = len(np.where(assigned_clusters[: 0] == curr_cluster))
#             my_sse = np.sum(np.square(data[i, :] - new_centroids[curr_cluster, :]))
#             my_sse = (my_size*my_sse)/(my_size-1)

#             sec_sse = sse[assigned_clusters[i, 1]]

#             ot_size = len(np.where(assigned_clusters[: 0] == ot_clus))
#             ot_sse = (ot_size*sse[wch_min_sse])/(ot_size+1)


#             if ot_sse < my_sse:
#                 new_assigned_clusters[i, 0] = int(wch_min_sse)
#                 # new_assigned_clusters[i, 1] = curr_cluster
#                 new_live_set += [curr_cluster, int(wch_min_sse)]

#                 if my_sse < sec_sse:
#                     new_assigned_clusters[i, 1] = curr_cluster
#                     # new_live_set += [assigned_clusters[i, 1]]
            
#             if ot_sse < sec_sse:
#                 new_assigned_clusters[i, 1] = int(wch_min_sse)
#                 # new_live_set += [assigned_clusters[i, 1], int(wch_min_sse)]

#     if len(new_live_set) == 0:
#         if loop_counter == 0:
#             print("LIVE set empty, at iteration: ", loop_counter)
#         exit()

#     # Update the centroids
#     # new_centroids = calculate_centroids_hw(data, new_assigned_clusters)

#     new_live_set = np.sort(np.unique(new_live_set)).tolist()

#     return new_live_set, new_assigned_clusters#, new_centroids


# def quick_trans(data, new_centroids, new_assigned_clusters, live_set):

#     new_live_set = []

#     for point in range(len(data)):

#         # print(assigned_clusters[point, :], "\t", new_assigned_clusters[point, :])
#         curr_cluster = new_assigned_clusters[point, 0]
#         ot_cluster = new_assigned_clusters[point, 1]

#         # if (assigned_clusters[point, :] != new_assigned_clusters[point, :]).any():
#         # if curr_cluster in live_set and ot_cluster in live_set:

#             # print(point, ": yes", curr_cluster, ot_cluster)

#         my_size = len(np.where(new_assigned_clusters[: 0] == curr_cluster))
#         ot_size = len(np.where(new_assigned_clusters[: 0] == ot_cluster))

#         my_sse = np.sum(np.square(data[point, :] - new_centroids[curr_cluster, :]))
#         ot_sse = np.sum(np.square(data[point, :] - new_centroids[ot_cluster, :]))

#         my_sse = (my_size* my_sse)/(my_size-1)
#         ot_sse = (ot_size*ot_sse)/(ot_size+1)

#         if ot_sse < my_sse:
#             new_assigned_clusters[point, 0] = ot_cluster
#             new_assigned_clusters[point, 1] = curr_cluster
#             new_live_set += [curr_cluster, ot_cluster]

#                 # new_centroids = calculate_centroids_hw(data, new_assigned_clusters, new_centroids, [curr_cluster, ot_cluster])

#     new_live_set = np.sort(np.unique(new_live_set)).tolist()

#     return new_live_set, new_assigned_clusters 
    

# def HWKmeans_1(data, num_clusters, num_iterations, seed):

#     loop_counter = 0

#     # All clusters are in LIVE set
#     live_set = [i for i in range(num_clusters)]

#     centroids = init_centroids(data, num_clusters, seed)
#     assigned_clusters, _ = calculate_distances_test(data, centroids)

#     print("initial: ", centroids)

#     new_assigned_clusters = np.zeros(shape=(assigned_clusters.shape), dtype='int')
    
#     # Re-calculate the centroids
#     new_centroids = calculate_centroids(data, assigned_clusters)
#     new_assigned_clusters[:] = assigned_clusters[:]

#     print("Updated: ",new_centroids)

#     ###########################
#     ######Optimal Transfer Stage
#     # DO one pass through the Data
#     ###########################
#     new_live_set, new_assigned_clusters = op_trans(data, new_centroids, 
#         assigned_clusters, new_assigned_clusters, live_set, loop_counter)

#     new_centroids = calculate_centroids(data, new_assigned_clusters)
    
#     # Copy Live set
#     live_set = new_live_set
    
#     # print("before Loop")
#     # print(assigned_clusters[0:10, :], "\t", new_assigned_clusters[0:10, :])

#     while loop_counter < num_iterations:

#         ###########################
#         ###### Quick Transfer Stage
#         # DO one pass through the Data
#         ###########################
         
#         loop_counter +=1
#         # print("Counter: ", loop_counter)

#         if loop_counter < 2:
#             print(new_centroids)

#         new_live_set, new_assigned_clusters = quick_trans(data, new_centroids, 
#             new_assigned_clusters, live_set)

#         live_set = new_live_set

#         print("Counter: ", loop_counter, " ", live_set)

#         assigned_clusters[:] = new_assigned_clusters[:]
#         new_centroids =  calculate_centroids(data, new_assigned_clusters)
 
#         if len(new_live_set) == 0:
#             print("yes")
#             new_live_set, new_assigned_clusters, new_centroids = op_trans(data, new_centroids, 
#                             assigned_clusters, new_assigned_clusters, live_set, loop_counter)
#             live_set = new_live_set
#             new_centroids =  calculate_centroids(data, new_assigned_clusters)
#         else:
#             # Update the centroids
#             continue


#     # sse = get_quality(data, new_assigned_clusters, new_centroids, num_clusters)
#     return new_centroids, loop_counter

