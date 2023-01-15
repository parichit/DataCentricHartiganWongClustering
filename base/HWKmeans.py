from utils.kmeans_utils import *
from utils.vis_utils import *
from scipy.spatial import distance


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

