import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.manifold import TSNE


def vis_data_with_he(dataset, centroids, centroid_neighbor, assigned_clusters, cluster_radius,
                         loop_counter, data_changed, he_data_indices):

    cols = ["col"+str(i) for i in range(dataset.shape[1])]

    dataset = pd.DataFrame(dataset)
    dataset.columns = [cols]
    temp = pd.DataFrame(centroids)
    temp.columns = [cols]

    # pca = PCA(n_components=2)
    # ss = StandardScaler()

    dataset = dataset.append(temp, ignore_index=True)

    # dataset = ss.fit_transform(dataset)
    # principalComponents = pca.fit_transform(dataset)
    # dataset = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    tsnecomp = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=40, 
                    random_state=10).fit_transform(dataset)
    dataset = pd.DataFrame(data=tsnecomp, columns=['PC1', 'PC2'])

    temp1 = [str(x) for x in assigned_clusters]
    temp1 += ["Centroid" for i in range(centroids.shape[0])]

    temp2 = [10 for i in range(len(assigned_clusters))]
    temp2 += [20 for i in range(centroids.shape[0])]

    dataset['labels'] = temp1
    dataset['size'] = temp2

    x_vals = []
    y_vals = []


    # mark HE data separately
    if len(he_data_indices) != 0:
        dataset.iloc[he_data_indices, dataset.shape[1] - 2] = 'HE-Data'

    if len(data_changed) != 0:
        dataset.iloc[data_changed, dataset.shape[1] - 2] = 'Actual-Change'

    plt.axis("equal")
    ax = sns.scatterplot(data=dataset, x='PC1', y='PC2', hue="labels", size='size')
    for i in range(len(centroids)):
        temp = dataset.loc[dataset['labels'] == 'Centroid'][['PC1', 'PC2']].values[i]
        # x_vals += [centroids[i][0]]
        # y_vals += [centroids[i][1]]
        plt.text(x=temp[0], y=temp[1], s=str(i))

    plt.plot(x_vals, y_vals)

    for i in data_changed:
        temp = dataset.iloc[i, 0:2].values
        plt.text(x=temp[0], y=temp[1], s=str(i))

    colors = ["green", "brown", "blue", "pink", 
              "black", "yellow", "orange", "aqua", "peru", 
              "darkred", "navy", "mintcream", "sienna", "lightcoral", 
              "blueviolet", "fuchsia", "lightpink", "steelblue", 
              "palegreen", "peru"]

    # for i in dataset['labels']:
    #     if i == "Centroid":
    #         print("Found")

    # for i in range(len(centroids)):
    #     # for j in centroid_neighbor[i]:
    #     temp = dataset.loc[dataset['labels'] == 'Centroid'][['PC1', 'PC2']].values[i]
    #     d1 = plt.Circle(xy=(temp[0], temp[1]), radius=cluster_radius[i], color=colors[i], fill=False)
    #     d2 = plt.Circle(xy=(temp[0], temp[1]), radius=cluster_radius[i]/2, color=colors[i], fill=False)
    #     ax.add_patch(d1)
    #     ax.add_patch(d2)
            # plt.plot(x_vals, y_vals)
            # plt.axvline(x=mid_points[i][0])
            # plt.axhline(y=mid_points[i][1])

    plt.title(str(loop_counter) + " 2D visualization for data")
    # plt.savefig(str(loop_counter) + ".png")
    plt.show()
    plt.close()


def vis_data_with_he_test(data, centroids, centroid_neighbor, assigned_clusters, cluster_radius,
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
    # mid_points = []
    # centroid_neighbor = {}


    # for i in range(len(centroids)):

    #     cen1_rad = cluster_radius[i]

    #     for j in range(len(centroids)):
    #         if i != j:
    #             cen1 = centroids[i]
    #             cen2 = centroids[j]

    #             mid_points.append([(cen1[0] + cen2[0]) / 2, (cen1[1] + cen2[1]) / 2])

    #             if i not in dist.keys():
    #                 dist[i] = {j: np.sqrt(np.sum(np.square(cen1 - cen2))) / 2}
    #             else:
    #                 dist[i][j] = np.sqrt(np.sum(np.square(cen1 - cen2))) / 2

    #             # If half the distance between the centroids is less than the radius of the current cluster
    #             if dist[i][j] < cen1_rad:
    #                 if i not in centroid_neighbor.keys():
    #                     centroid_neighbor[i] = [j]
    #                 else:
    #                     centroid_neighbor[i].append(j)

    # mark HE data separately
    if len(he_data_indices) != 0:
        dataset.iloc[he_data_indices, dataset.shape[1] - 2] = 'HE-Data'

    if len(data_changed) != 0:
        dataset.iloc[data_changed, dataset.shape[1] - 2] = 'Actual-Change'

    #plt.axis("equal")
    ax = sns.scatterplot(data=dataset, x='x', y='y', hue="labels", size='size')
    for i in range(len(centroids)):
        x_vals += [centroids[i][0]]
        y_vals += [centroids[i][1]]
        plt.text(x=centroids[i][0], y=centroids[i][1], s=str(i))

    plt.plot(x_vals, y_vals)

    for i in data_changed:
        temp = dataset.iloc[i, 0:2].values
        plt.text(x=temp[0], y=temp[1], s=str(i))

    colors = ["green", "brown", "blue", "pink", 
              "black", "yellow", "orange", "aqua", "peru", 
              "darkred", "navy", "mintcream", "sienna"]

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
    # plt.savefig(str(loop_counter) + ".png")
    plt.show()
    plt.close()


def find_all_he_indices_neighbor(dataset, new_centroids, radius, assign_dict, cluster_info):
    centroids_neighbor, dist_mat = find_neighbors(new_centroids, radius)
    he_indices_dict = find_all_he_points(dataset, centroids_neighbor, dist_mat, new_centroids, 
                                               assign_dict, cluster_info)
    return centroids_neighbor, he_indices_dict


def find_neighbors(new_centroids, radius):

    centroid_neighbor = {}
    dist_mat = distance.cdist(new_centroids, new_centroids, "euclidean")
    dist_mat = np.divide(dist_mat, 2)

    for i in range(len(new_centroids)):

        cen1_rad = radius[i]
        neighbors = np.where(dist_mat[i] < cen1_rad)[0]
        centroid_neighbor[i] = np.array(neighbors)

    return centroid_neighbor, dist_mat



def find_all_he_points(dataset, centroids_neighbor, dist_mat, new_centroids, 
                             assign_dict, cluster_info):

    he_data = {}
    
    for curr_cluster in range(len(new_centroids)):

        center1 = new_centroids[curr_cluster]
        temp_list_1 = assign_dict[curr_cluster]
        he_data_indices = []

        if cluster_info[curr_cluster] > 1:

            my_scale = cluster_info[curr_cluster]/(cluster_info[curr_cluster]-1)
            test_data = dataset[temp_list_1]


            # Determine the sign of other centroid
            for ot_cen in centroids_neighbor[curr_cluster]:

                if curr_cluster != ot_cen:

                    # print("Centers: ", curr_cluster, "-", ot_cen)
                    center2 = new_centroids[ot_cen]
                    ot_scale = cluster_info[ot_cen]/(cluster_info[ot_cen]+1)

                    # Find coordinates of points to draw affine vectors
                    centroid_vec = np.subtract(center2, center1)
                    centroid_vec = centroid_vec / np.linalg.norm(centroid_vec)

                    if cluster_info[ot_cen] < cluster_info[curr_cluster]:
                        scale_fac = ((cluster_info[ot_cen]/cluster_info[curr_cluster]))*100
                        # if curr_cluster == 4 and ot_cen == 14:
                        #     temp_fac = (cluster_info[ot_cen]/cluster_info[curr_cluster]) + (my_scale/ot_scale)
                        #     temp_fac1 = dist_mat[curr_cluster, ot_cen] - temp_fac
                        #     print(temp_fac, temp_fac1)
                    else:
                        scale_fac = ((cluster_info[curr_cluster]/cluster_info[ot_cen]))*100

                    if scale_fac >= dist_mat[curr_cluster, ot_cen] or int(scale_fac) == int(dist_mat[curr_cluster, ot_cen]):
                        if cluster_info[ot_cen] > cluster_info[curr_cluster]:
                            scale_fac = (cluster_info[ot_cen]/cluster_info[curr_cluster]) + (my_scale/ot_scale)
                        else:
                            scale_fac = (cluster_info[curr_cluster]/cluster_info[ot_cen]) + (my_scale/ot_scale)
                        
                        scale_fac = dist_mat[curr_cluster, ot_cen] - scale_fac
                        # mid_point = np.divide(np.add(center1, center2), 2)
                        mid_point = center1 + (scale_fac*centroid_vec)

                    else:
                        mid_point = center1 + (scale_fac*centroid_vec)
                    
                    point_sign = find_sign_by_product(mid_point, center2, test_data)

                    # if curr_cluster == 19 and ot_cen == 0:
                    #     p = 673
                    #     i = np.where(temp_list_1 == p)[0]
                    #     perp = np.sqrt(np.sum(np.square(dataset[p, ] - mid_point)))
                    #     dist = np.sqrt(np.sum(np.square(center1 - mid_point)))
                    #     dist1 = np.sqrt(np.sum(np.square(dataset[p,] - new_centroids[ot_cen])))
                    #     # print("Dist: ", dist_mat[curr_cluster, ot_cen], scal_factor)
                    #     # print("Mid point: ", mid_point1)
                    #     print("Perp Dist: ", perp, dist, dist1, dist_mat[curr_cluster, ot_cen], scale_fac)

                    same_sign = np.where(point_sign > 0)[0]

                    if len(same_sign) > 0:
                        he_data_indices += temp_list_1[same_sign].tolist()

            he_data[curr_cluster] = np.unique(he_data_indices).tolist()

    return he_data


def calculate_sse_specific(data, centroids, neighbors, cluster_info, curr_cluster):

    # Find pairwise distances
    n, _ = data.shape
    dist_mat = np.zeros((n, len(centroids)), dtype=float)
    my_size = cluster_info[curr_cluster]/(cluster_info[curr_cluster]-1)

    for i in range(len(centroids)):
        
        center = neighbors[i]

        for i in range(n):
            dist_mat[i, :] = np.sum(np.square(data[i, :] - centroids), 1)
        
        for i in range(len(centroids)):
            center = neighbors[i]
            if center == curr_cluster:
                dist_mat[:, i] *= my_size
            else:
                ot_size = cluster_info[center]/(cluster_info[center]+1)
                dist_mat[:, i] *= ot_size
        
        # if center == curr_cluster:
        #     dist_mat[:, i] = np.sum(np.square(data - centroids[i]), 1) * my_size
        # else:
        #     ot_size = cluster_info[center]/(cluster_info[center]+1)
        #     dist_mat[:, i] = np.sum(np.square(data - centroids[i]), 1) * ot_size

    # Find the closest centroid
    assigned_clusters = np.argmin(dist_mat, axis=1)
    distances = np.sqrt(np.min(dist_mat, axis=1))

    temp = np.array(neighbors)
    assigned_clusters = temp[assigned_clusters]

    return assigned_clusters, distances



def find_cluster_specific_he_points(dataset, neighbors, dist_mat, new_centroids, 
                             assign_dict, cluster_info, curr_cluster):

    he_points = []
    
    center1 = new_centroids[curr_cluster]
    temp_list_1 = assign_dict[curr_cluster]
    he_data_indices = []

    my_scale = cluster_info[curr_cluster]/(cluster_info[curr_cluster]-1)
    test_data = dataset[temp_list_1, :]

        
    # Determine the HE data in relation to neighbor 
    # centroids
    for ot_cen in neighbors[curr_cluster]:

        if curr_cluster != ot_cen:

            center2 = new_centroids[ot_cen]
            ot_scale = cluster_info[ot_cen]/(cluster_info[ot_cen]+1)

            # Find coordinates of points to draw affine vectors
            centroid_vec = np.subtract(center2, center1)
            centroid_vec = centroid_vec / np.linalg.norm(centroid_vec)

            if cluster_info[ot_cen] < cluster_info[curr_cluster]:
                scale_fac = (cluster_info[ot_cen]/cluster_info[curr_cluster])*100
            else:
                scale_fac = (cluster_info[curr_cluster]/cluster_info[ot_cen])*100

            if scale_fac >= dist_mat[curr_cluster, ot_cen]:
                if cluster_info[ot_cen] > cluster_info[curr_cluster]:
                    scale_fac = (cluster_info[ot_cen]/cluster_info[curr_cluster]) + (my_scale/ot_scale)
                else:
                    scale_fac = (cluster_info[curr_cluster]/cluster_info[ot_cen]) + (my_scale/ot_scale)
                
                scale_fac = dist_mat[curr_cluster, ot_cen] - scale_fac
                mid_point = center1 + (scale_fac*centroid_vec)
                # mid_point = np.divide(np.add(center1, center2), 2)

            else:
                mid_point = np.add(center1, np.multiply(scale_fac, centroid_vec))
            
            point_sign = find_sign_by_product(mid_point, center2, test_data)

            same_sign = np.where(point_sign > 0)[0]

            if len(same_sign) > 0:
                he_data_indices += temp_list_1[same_sign].tolist()

    he_points = np.unique(he_data_indices).tolist()

    return he_points



def find_sign_by_product(mid_point, center2, points):
    temp_vec = np.subtract(center2, mid_point)
    points_vec = np.subtract(points, mid_point)
    return points_vec.dot(temp_vec)


def get_membership(assigned_clusters, distances, num_clusters):

    radius = {}
    assign_dict = {}
    cluster_info = []

    for i in range(num_clusters):
        indices = np.where(assigned_clusters == i)[0]
        assign_dict[i] = indices
        radius[i] = np.max(distances[indices])
        cluster_info += [len(indices)]
    
    return assign_dict, radius, cluster_info


def get_size(assigned_clusters, num_clusters):
    sizes = []
    for i in range(num_clusters):
        sizes += [len(np.where(assigned_clusters == i)[0])]
    return sizes



def find_he_new(data, new_centroids, dist_mat, indices, 
curr_cluster, num_clusters, cluster_info):

    my_size = cluster_info[curr_cluster]
    my_dists = np.sqrt(np.sum(np.square(data[indices, :] - new_centroids), 1))

    ot_dists = np.zeros(shape=(my_size, num_clusters))

    points = []
    he_index = []
    
    ot_dists[:, curr_cluster] = my_dists[:]

    for ot_cluster in range(num_clusters):

        if ot_cluster != curr_cluster:
            
            ot_size = cluster_info[ot_cluster]
            ratio = (my_size/(my_size-1))/(ot_size/(ot_size+1))
            min_l = np.abs(ot_dists[:, curr_cluster] - dist_mat[curr_cluster, ot_cluster])

            temp = np.where(min_l+0.1 < my_dists*ratio)[0]
            if len(temp) > 0:
                he_index += temp.tolist()

            # if curr_cluster == 1 and ot_cluster == 3:
            #     print(indices)
            #     i = np.where(indices == 14)[0]
            #     print("Old sse: ", my_dists[i], my_dists[i]*ratio, " New: ", min_l[i])
        

    if len(he_index) > 0:
        he_index = np.unique(he_index)
        points = indices[he_index]


    return points 










