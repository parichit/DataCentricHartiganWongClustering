import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.cluster import adjusted_mutual_info_score as amis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
from sklearn.cluster import kmeans_plusplus


def init_centroids(data, num_clusters, seed):

    # Randomly select points from the data as centroids
    # np.random.seed(seed)

    # indices = np.random.choice(data.shape[0], num_clusters, replace=False, )
    return np.array(data[0:num_clusters, :])
    # return np.array(data[indices, :])


    # return np.array(data[0:num_clusters, :])
    # return centers
    # return np.array(data[indices, :])


def calculate_distances(data, centroids):

    # Find pairwise distances
    n, d = data.shape
    dist_mat = np.zeros((n, len(centroids)), dtype=float)

    for i in range(n):
        dist_mat[i, :] = np.sqrt(np.sum(np.square(data[i] - centroids), 1))

    return np.argmin(dist_mat, axis=1), np.round(np.min(dist_mat, axis=1), 5)


def calculate_distances_less_modalities(data, centroids):

    num_clusters = len(centroids)
    stat = False

    # Find pairwise distances
    n, d = data.shape
    dist_mat = np.zeros((n, num_clusters), dtype=float)

    for i in range(n):
        dist_mat[i, :] = np.sqrt(np.sum(np.square(data[i] - centroids), 1))

    assigned_clusters = np.argmin(dist_mat, axis=1)
    u_clusters = np.sort(np.unique(assigned_clusters))

    remove_indices = [i for i in range(num_clusters) if i not in u_clusters]

    if len(remove_indices) > 0:
        stat = True

    return assigned_clusters, np.round(np.min(dist_mat, axis=1), 5), stat


def calculate_distances_specific(data, centroids, neighbors):

    # Find pairwise distances
    n, d = data.shape
    dist_mat = np.zeros((n, len(centroids)), dtype=float)

    for i in range(len(centroids)):
        dist_mat[:, i] = np.sum(np.square(data - centroids[i]), 1)
    dist_mat = np.sqrt(dist_mat)

    # Find the closest centroid
    assigned_clusters = np.argmin(dist_mat, axis=1)
    distances = np.min(dist_mat, axis=1)

    temp = np.array(neighbors)
    assigned_clusters = temp[assigned_clusters]

    return assigned_clusters, np.round(distances, 5)


def calculate_centroids(data, assigned_clusters):

    temp = [np.mean(data[np.where(assigned_clusters == i),], axis=1)[0] for i in np.sort(np.unique(assigned_clusters))]
    new_centroids = np.array(temp)

    return np.round(new_centroids, 5)


def create_sorted_structure(assigned_clusters, distances, num_clusters):

        BST_List = {i: [] for i in range(num_clusters)}

        # Add data to each sublist
        for i in range(len(assigned_clusters)):
            BST_List[assigned_clusters[i]].append([distances[i], i])

        # Create the balanced BST out of each list in the dictionary
        for i in range(num_clusters):
            BST_List[i] = SortedDict(BST_List[i])
        return BST_List


def move_data_around(bst_list, he_bst_indices, he_points, assigned_clusters, new_assigned_clusters,
                     he_new_assigned_centers, he_new_min_dist):

    data_indices = np.where(assigned_clusters[he_points] != he_new_assigned_centers)[0]

    if data_indices.size > 0:
        old_assign = assigned_clusters[he_points]

        for index in data_indices:

            old_center = old_assign[index]
            new_center = he_new_assigned_centers[index]

            actual_point = he_points[index]

            if new_assigned_clusters[actual_point] == old_center:
                bst_list[old_center].pop(he_bst_indices[actual_point])
                bst_list[new_center].update({he_new_min_dist[index]: actual_point})
                he_bst_indices[actual_point] = he_new_min_dist[index]
                new_assigned_clusters[actual_point] = new_center
            else:
                if he_new_min_dist[index] < he_bst_indices[actual_point]:
                    old_center = new_assigned_clusters[actual_point]
                    bst_list[old_center].pop(he_bst_indices[actual_point])
                    bst_list[new_center].update({he_new_min_dist[index]: actual_point})
                    he_bst_indices[actual_point] = he_new_min_dist[index]
                    new_assigned_clusters[actual_point] = new_center

    return bst_list, new_assigned_clusters


def move_all_data_around(bst_list, he_bst_indices, he_points, assigned_clusters,
                     he_new_assigned_centers, he_new_min_dist):

    moved_indices = np.where(assigned_clusters[he_points] != he_new_assigned_centers)[0]
    non_moved_indices = np.where(assigned_clusters[he_points] == he_new_assigned_centers)[0]

    # Handling the non moved data
    if non_moved_indices.size > 0:
        # print("Debug-1.0", assigned_clusters[he_points][non_moved_indices], he_points[non_moved_indices])
        for index in non_moved_indices:
            actual_point = he_points[index]
            if he_new_min_dist[index] < he_bst_indices[actual_point]:

                # print("Debug-1.1", "Point:", actual_point, "old: ", assigned_clusters[actual_point],
                #       "new: ", he_new_assigned_centers[index],
                #       "old dist: ", he_bst_indices[actual_point],
                #       "new dist: ", he_new_min_dist[index])
                #
                # for i in range(len(bst_list)):
                #     print(i, bst_list[i])

                old_center = assigned_clusters[actual_point]
                # print(bst_list[old_center])
                bst_list[old_center].pop(he_bst_indices[actual_point])
                bst_list[old_center].update({he_new_min_dist[index]: actual_point})
                he_bst_indices[actual_point] = he_new_min_dist[index]

    # Handling the moved data
    if moved_indices.size > 0:
        old_assign = assigned_clusters[he_points]
        # print("Debug-0:", he_points, len(he_points), len(moved_indices), np.array(he_points)[moved_indices])

        for index in moved_indices:

            old_center = old_assign[index]
            new_center = he_new_assigned_centers[index]

            actual_point = he_points[index]

            # print("Debug-1:", old_center, new_center, actual_point, he_bst_indices[actual_point], he_new_min_dist[index])

            # print(bst_list[old_center])
            # print(bst_list[new_center])

            bst_list[old_center].pop(he_bst_indices[actual_point])
            bst_list[new_center].update({he_new_min_dist[index]: actual_point})
            he_bst_indices[actual_point] = he_new_min_dist[index]

    return bst_list


def calculate_my_distances(point, centroids):

    temp = []
    for i in centroids:
        temp.append(np.sqrt(np.mean(np.square(point-i))))

    return np.argmin(temp), np.min(temp)


def check_convergence(current_centroids, centroids, threshold):

    rms = round(np.sqrt(np.mean(np.square(current_centroids-centroids))), 5)
    if rms <= threshold:
        return True
    return False


def pred_membership(data, centroids):
    dist_mat = cdist(data, centroids, 'euclidean')
    # Find the closest centroid
    assigned_cluster = np.argmin(dist_mat, axis=1).tolist()
    return assigned_cluster


def check_ARI(label1, label2):
    return round(ari(label1, label2), 2)


def check_amis(label1, label2):
    return round(amis(label1, label2), 2)


def check_empty_clusters(assignment, num_clusters):

    counter = 0

    while counter < 4:
        if len(np.unique(assignment)) < num_clusters:
            assignment = np.random.randint(0, len(assignment), num_clusters)
        else:
            break
        counter +=1

    return assignment


def vis_PCA(dataset, labels):

    pca = PCA(n_components=2)
    ss = StandardScaler()

    dataset = pd.DataFrame(dataset)

    dataset = ss.fit_transform(dataset)
    principalComponents = pca.fit_transform(dataset)
    pc_esc = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    pc_esc['labels'] = labels

    sns.color_palette("colorblind")
    sns.scatterplot(data=pc_esc, x='PC1', y='PC2', hue="labels")

    plt.show()


def do_PCA(dataset, centroids1, centroids2, labels, title, file_name):

    pca = PCA(n_components=2)
    ss = StandardScaler()

    dataset = pd.DataFrame(dataset)
    centroids1 = pd.DataFrame(centroids1)
    # centroids2 = pd.DataFrame(centroids2)

    dataset = dataset.append(centroids1, ignore_index=True)
    # dataset = dataset.append(centroids2, ignore_index=True)

    dataset = ss.fit_transform(dataset)
    principalComponents = pca.fit_transform(dataset)
    pc_esc = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    temp1 = [str(x) for x in labels]
    temp1 += ["Centroid" for i in range(5)]

    temp2 = [10 for i in range(len(labels))]
    temp2 += [20 for i in range(len(centroids1))]
    # temp2 += [15 for i in range(len(centroids2))]

    pc_esc['labels'] = temp1
    pc_esc['size'] = temp2

    sns.color_palette("colorblind")
    sns.scatterplot(data=pc_esc, x='PC1', y='PC2', hue="labels", size='size')
    plt.title(title+" points: 2D visualization for data")

    plt.savefig(file_name + ".png")
    plt.show()
    #plt.close()


def get_quality(data, final_assign, final_centroids, num_clusters):

    final_sse = 0

    for i in range(len(final_centroids)):
        indices = np.where(final_assign == i)[0]
        final_sse += np.sqrt(np.sum(np.square(data[indices, :] - final_centroids[i, :])))

    final_sse /= num_clusters

    return final_sse



