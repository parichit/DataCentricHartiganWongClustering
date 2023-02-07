import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


def vis_data_with_he(data, centroids, assigned_clusters, distances,
                         loop_counter, data_changed, he_data_indices):

    dataset = pd.DataFrame(data, columns=['x', 'y'])
    temp = pd.DataFrame(centroids, columns=['x', 'y'])
    dataset = dataset.append(temp, ignore_index=True)

    temp1 = [str(x) for x in assigned_clusters]
    temp1 += ["Centroid" for i in range(centroids.shape[0])]

    temp2 = [10 for i in range(len(assigned_clusters))]
    temp2 += [20 for i in range(centroids.shape[0])]

    dataset['labels'] = temp1
    dataset['size'] = temp2

    x_vals = []
    y_vals = []

    dist = {}
    mid_points = []
    centroid_neighbor = {}

    for i in range(len(centroids)):

        s = np.where(assigned_clusters == i)[0]
        cen1_rad = np.max(distances[s])

        for j in range(len(centroids)):
            if i != j:
                cen1 = centroids[i]
                cen2 = centroids[j]

                mid_points.append([(cen1[0] + cen2[0]) / 2, (cen1[1] + cen2[1]) / 2])

                if i not in dist.keys():
                    dist[i] = {j: np.sqrt(np.sum(np.square(cen1 - cen2))) / 2}
                else:
                    dist[i][j] = np.sqrt(np.sum(np.square(cen1 - cen2))) / 2

                # If half the distance between the centroids is less than the radius of the current cluster
                if dist[i][j] < cen1_rad:
                    if i not in centroid_neighbor.keys():
                        centroid_neighbor[i] = [j]
                    else:
                        centroid_neighbor[i].append(j)

    # mark HE data separately
    if len(he_data_indices) != 0:
        dataset.iloc[he_data_indices, dataset.shape[1] - 2] = 'HE-Data'
    # exit()

    if len(data_changed) != 0:
        dataset.iloc[data_changed, dataset.shape[1] - 2] = 'Actual-Change'

    plt.axis("equal")
    ax = sns.scatterplot(data=dataset, x='x', y='y', hue="labels", size='size')
    for i in range(len(centroids)):
        x_vals += [centroids[i][0]]
        y_vals += [centroids[i][1]]
        plt.text(x=centroids[i][0], y=centroids[i][1], s=str(i))

    plt.plot(x_vals, y_vals)

    for i in data_changed:
        temp = dataset.iloc[i, 0:2].values
        plt.text(x=temp[0], y=temp[1], s=str(i))

    for i in centroid_neighbor.keys():
        for j in centroid_neighbor[i]:
            temp = dataset.loc[dataset['labels'] == 'Centroid'][['x', 'y']].values[i]
            d = plt.Circle(xy=(temp[0], temp[1]), radius=dist[i][j], color='green', fill=False)
            ax.add_patch(d)
            plt.plot(x_vals, y_vals)
            # plt.axvline(x=mid_points[i][0])
            # plt.axhline(y=mid_points[i][1])

    plt.title(str(loop_counter) + " 2D visualization for data")
    plt.savefig(str(loop_counter) + ".png")
    plt.show()


def vis_data_with_he_test(data, centroids, assigned_clusters, cluster_radius,
                         loop_counter, data_changed, he_data_indices):

    dataset = pd.DataFrame(data, columns=['x', 'y'])
    temp = pd.DataFrame(centroids, columns=['x', 'y'])
    dataset = dataset.append(temp, ignore_index=True)

    temp1 = [str(x) for x in assigned_clusters]
    temp1 += ["Centroid" for i in range(centroids.shape[0])]

    temp2 = [10 for i in range(len(assigned_clusters))]
    temp2 += [20 for i in range(centroids.shape[0])]

    dataset['labels'] = temp1
    dataset['size'] = temp2

    x_vals = []
    y_vals = []

    dist = {}
    mid_points = []
    centroid_neighbor = {}

    for i in range(len(centroids)):

        cen1_rad = cluster_radius[i]

        for j in range(len(centroids)):
            if i != j:
                cen1 = centroids[i]
                cen2 = centroids[j]

                mid_points.append([(cen1[0] + cen2[0]) / 2, (cen1[1] + cen2[1]) / 2])

                if i not in dist.keys():
                    dist[i] = {j: np.sqrt(np.sum(np.square(cen1 - cen2))) / 2}
                else:
                    dist[i][j] = np.sqrt(np.sum(np.square(cen1 - cen2))) / 2

                # If half the distance between the centroids is less than the radius of the current cluster
                if dist[i][j] < cen1_rad:
                    if i not in centroid_neighbor.keys():
                        centroid_neighbor[i] = [j]
                    else:
                        centroid_neighbor[i].append(j)

    # mark HE data separately
    if len(he_data_indices) != 0:
        dataset.iloc[he_data_indices, dataset.shape[1] - 2] = 'HE-Data'

    if len(data_changed) != 0:
        dataset.iloc[data_changed, dataset.shape[1] - 2] = 'Actual-Change'

    plt.axis("equal")
    ax = sns.scatterplot(data=dataset, x='x', y='y', hue="labels", size='size')
    for i in range(len(centroids)):
        x_vals += [centroids[i][0]]
        y_vals += [centroids[i][1]]
        plt.text(x=centroids[i][0], y=centroids[i][1], s=str(i))

    plt.plot(x_vals, y_vals)

    for i in data_changed:
        temp = dataset.iloc[i, 0:2].values
        plt.text(x=temp[0], y=temp[1], s=str(i))

    colors = ["green", "brown", "blue", "pink", "black"]

    print(centroid_neighbor)
    for i in range(len(centroids)):
        # for j in centroid_neighbor[i]:
        temp = dataset.loc[dataset['labels'] == 'Centroid'][['x', 'y']].values[i]
        d1 = plt.Circle(xy=(temp[0], temp[1]), radius=cluster_radius[i], color=colors[i], fill=False)
        d2 = plt.Circle(xy=(temp[0], temp[1]), radius=cluster_radius[i]/2, color=colors[i], fill=False)
        ax.add_patch(d1)
        ax.add_patch(d2)
            # plt.plot(x_vals, y_vals)
            # plt.axvline(x=mid_points[i][0])
            # plt.axhline(y=mid_points[i][1])

    plt.title(str(loop_counter) + " 2D visualization for data")
    plt.savefig(str(loop_counter) + ".png")
    plt.show()
    plt.close()


def find_all_he_indices_neighbor(dataset, new_centroids, radius, assign_dict):
    centroids_neighbor = get_midpoints_np(new_centroids, radius)
    he_indices_dict = find_all_points_neighbor(dataset, centroids_neighbor, new_centroids, assign_dict)
    return centroids_neighbor, he_indices_dict


def get_midpoints_np(new_centroids, radius):

    centroid_neighbor = {}

    # for k in range(len(new_centroids)):
    #     dist_mat[:, k] = np.sum(np.square(np.subtract(new_centroids, new_centroids[k])), 1)
    # dist_mat = np.sqrt(dist_mat)
    # dist_mat = np.divide(dist_mat, 2)
    
    dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
    dist_mat = np.divide(dist_mat, 2)

    for i in range(len(new_centroids)):

        cen1_rad = radius[i]
        neighbors = np.where(dist_mat[i] < cen1_rad)[0]
        centroid_neighbor[i] = np.array(neighbors)

    return centroid_neighbor


def find_all_points_test(data, curr_cluster, radius, new_centroid, assigned_clusters, 
assign_dict, distances):

    he_data = []

    all_indices = np.where(assigned_clusters == curr_cluster)[0]
    assign_dict[curr_cluster] = all_indices

    # print("test-1: ", new_centroid.shape)
    sse = np.sqrt(np.sum(np.square(data[all_indices, :] - new_centroid), 1))
    distances[all_indices.tolist()] = sse
    weight_factor = 1


    if len(all_indices) > 1:
        weight_factor = len(all_indices)/(len(all_indices)-1)
        # max_dist =  max_dist * (len(all_indices)/(len(all_indices)-1))
        sse *= len(all_indices)/(len(all_indices)-1)

    max_dist = np.max(sse)
    radius[curr_cluster] = max_dist

    my_radius = (radius[curr_cluster]/2)
    indices = np.where((sse+weight_factor) >= (my_radius-weight_factor))[0]

    if curr_cluster == 0:
        i = np.where(all_indices == 415)[0]
        print(sse[i], "\t", radius[curr_cluster], my_radius, weight_factor)

    if len(indices)>0:
        he_data = all_indices[indices].tolist()
        sse = sse[indices].tolist()

    return assign_dict, radius, he_data


def find_all_points_neighbor(dataset, centroids_neighbor, new_centroids, assign_dict):

    he_data = {}

    for curr_cluster in range(len(new_centroids)):

        center1 = new_centroids[curr_cluster]
        temp_list_1 = np.array(assign_dict[curr_cluster])
        he_data_indices = []

        # Determine the sign of other centroid
        for ot_cen in centroids_neighbor[curr_cluster]:

            if curr_cluster != ot_cen:

                # print("Centers: ", curr_cluster, "-", ot_cen)
                center2 = new_centroids[ot_cen]
                mid_point = np.divide(np.add(center1, center2), 2)
                
                test_data = dataset[temp_list_1]

                point_sign = find_sign_by_product(mid_point, center2, test_data)
                same_sign = np.where(point_sign > 0)[0]

                # print("Center-1: ", curr_cluster, " other center: ", ot_cen,
                #       " No. HE points: ", len(same_sign))

                if len(same_sign) > 0:
                    he_data_indices += temp_list_1[same_sign].tolist()

        he_data[curr_cluster] = np.unique(he_data_indices)

    return he_data



def find_sign_by_product(mid_point, center2, points):

    temp_vec = np.subtract(center2, mid_point)
    points_vec = np.subtract(points, mid_point)
    return points_vec.dot(temp_vec)



def get_membership(assigned_clusters, distances, num_clusters):

    radius = {}
    assign_dict = {}
    cluster_info = []

    for i in range(num_clusters):
        indices = np.where(assigned_clusters == i)[0].tolist()
        assign_dict[i] = indices
        # radius[i] = (len(indices)/(len(indices)-1))*np.max(distances[indices])
        radius[i] = np.max(distances[indices])
        # print(len(indices), radius[i], np.max(distances[indices]))
        cluster_info += [len(indices)]
    
    return assign_dict, radius, cluster_info


def get_size(assigned_clusters, num_clusters):
    sizes = []
    for i in range(num_clusters):
        sizes += [len(np.where(assigned_clusters == i)[0])]
    return sizes



def find_he(data, new_centroids, dist_mat, indices, 
curr_cluster, num_clusters, cluster_info):

    my_dists = np.sqrt(np.sum(np.square(data[indices, :] - new_centroids), 1))
    temp = np.sqrt(np.sum(np.square(data[indices, :] - new_centroids), 1))
    # ot_dists = np.zeros(shape=my_dists.shape)
    ot_dists = np.zeros(shape=(cluster_info[curr_cluster], num_clusters))

    my_dists = (my_dists * cluster_info[curr_cluster])/(cluster_info[curr_cluster]-1)

    acc_index = []
    he_points = []

    # print(ot_dists.shape, len(indices), cluster_info[curr_cluster], len(my_dists))
    
    ot_dists[:, curr_cluster] = my_dists[:]

    for ot_cluster in range(num_clusters):

        if ot_cluster != curr_cluster:

            # print(ot_cluster, my_dists)

            ot_dists[:, ot_cluster] = temp[:]

            # ot_dists[:, ot_cluster] = np.abs(ot_dists[:, ot_cluster] - dist_mat[curr_cluster, ot_cluster])
            # # ot_dists = (ot_dists * cluster_info[ot_cluster])/(cluster_info[ot_cluster]-1)
            # temp = np.where(ot_dists[:, ot_cluster] < my_dists)[0]
            # if len(temp) > 0:
            #     acc_index += list(temp)

            # ot_dists[:, ot_cluster] = (ot_dists[:, curr_cluster] + dist_mat[curr_cluster, ot_cluster]) 
            ot_dists[:, ot_cluster] = (ot_dists[:, ot_cluster] * cluster_info[ot_cluster])/(cluster_info[ot_cluster]+1)
            temp1 = (dist_mat[curr_cluster, ot_cluster] * cluster_info[ot_cluster])/(cluster_info[ot_cluster]+1)
            ub = ot_dists[:, ot_cluster] + temp1
            # lb = np.abs(ot_dists[:, curr_cluster] - dist_mat[curr_cluster, ot_cluster]) 
            ot_dists[:, ot_cluster] = ub

            # temp = np.where(ot_dists < my_dists)[0]
            # if len(temp) > 0:
            #     acc_index += list(temp)

    # print(ot_dists)
    acc_dist = np.min(ot_dists, axis=1)
    acc_index = np.where(acc_dist < my_dists)[0]
    # new_assigned_clusters[indices] = acc_index

    if len(acc_index) > 0:
        he_points = indices[acc_index]
        # distances[points] = acc_dist[acc_index]


    # if len(acc_index) > 0:
    #     acc_index = np.unique(acc_index)
    #     # print(acc_index)
    #     he_point = indices[acc_index]

    # print(he_points)
    return he_points


def find_he_new(data, new_centroids, dist_mat, indices, 
curr_cluster, num_clusters, cluster_info, distances):

    my_size = cluster_info[curr_cluster]

    my_dists = np.sqrt(np.sum(np.square(data[indices, :] - new_centroids), 1))
    # my_dists = np.sum(np.square(data[indices, :] - new_centroids), 1)
    # my_dists = distances[indices]


    ot_dists = np.zeros(shape=(my_size, num_clusters))

    # my_dists = (my_dists * my_size)/(my_size-1)

    acc_index = []
    points = []

    # print("Hello:", assigned_clusters)
    # print(ot_dists.shape, len(indices), cluster_info[curr_cluster])
    
    ot_dists[:, curr_cluster] = my_dists[:]

    for ot_cluster in range(num_clusters):

        if ot_cluster != curr_cluster:
            
            ot_size = cluster_info[ot_cluster]
            
            ratio = (my_size/(my_size-1))/(ot_size/(ot_size+1))

            # ot_dists[:, ot_cluster] = my_dists[:]
            # ot_dists[:, ot_cluster] *= ratio
            min_l = np.abs(ot_dists[:, curr_cluster] - dist_mat[curr_cluster, ot_cluster])
            # max_l = ot_dists[:, curr_cluster] + dist_mat[curr_cluster, ot_cluster]

            # ot_dists[:, ot_cluster] = (ot_dists[:, ot_cluster] * cluster_info[ot_cluster])/(cluster_info[ot_cluster]+1)

            temp = np.where((min_l < my_dists*ratio))[0]
            if len(temp) > 0:
                acc_index += temp.tolist()
        

    # acc_index = np.argmin(ot_dists, axis=1)
    # print(ot_dists)
    # acc_dist = np.min(ot_dists, axis=1)
    # acc_index = np.where(acc_dist < my_dists)[0]
    # new_assigned_clusters[indices] = acc_index

    if len(acc_index) > 0:
        acc_index = np.unique(acc_index)
        points = indices[acc_index]
        # distances[points] = acc_dist[acc_index]
        # new_clus = np.argmin(ot_dists[acc_index], axis=1)
        # new_assigned_clusters[points] = new_clus
        # print("\n")
        # print("Actual Change: ", points)
        # print("Predicted: ", he_data_indices)

    # return acc_index.tolist(), acc_dist.tolist()
    return points, ot_dists[acc_index].tolist(), distances


def find_he_by_radius(data, new_centroids, cluster_radius, indices, num_clusters):

    my_dist = np.sqrt(np.sum(np.square(data[indices, :] - new_centroids), 1))

    temp_list = []

    for ot_clus in range(num_clusters):
        temp_list += np.where(my_dist >= cluster_radius[ot_clus]/2)[0].tolist()
    
    # if len(temp_list) > 0:
    #     temp_list = list(np.unique(temp_list))
    #     temp_list = indices[temp_list].tolist()
    
    if len(temp_list) > 0:
        temp_list = np.unique(temp_list)
        temp_list = indices[temp_list].tolist()
        # print(temp_list)
    
    return temp_list







