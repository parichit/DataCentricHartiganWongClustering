import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.cluster import adjusted_mutual_info_score as amis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys


def init_centroids(data, num_clusters, seed):

    # Randomly select points from the data as centroids
    # np.random.seed(seed)
    # indices = random.sample(range(num_clusters), num_clusters)
    # indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    return np.array(data[0:num_clusters, :])
    return np.array(data[indices, :])


def calculate_distances(data, centroids):

    # Find pairwise distances
    n, d = data.shape
    dist_mat = np.zeros((n, len(centroids)), dtype=float)

    for i in range(n):
        dist_mat[i, :] = np.sqrt(np.sum(np.square(data[i] - centroids), 1))
        # dist_mat[i, :] = np.sum(np.square(data[i] - centroids), 1)

    return np.argmin(dist_mat, axis=1), np.round(np.min(dist_mat, axis=1), 5)



# def calculate_sse_specific(data, new_centroids, cluster_info, he_data_indices, assigned_clusters, 
# curr_cluster, my_sse, distances):

#     curr_cluster_info = cluster_info[curr_cluster]

#     sse = distance.cdist(data[he_data_indices, :], new_centroids, 'sqeuclidean')
#     sse_copy = np.zeros(shape=sse.shape)
#     sse_copy[:] = sse[:]

#     if curr_cluster_info > 1:
#         sse[:, curr_cluster] = (curr_cluster_info * sse[:, curr_cluster])/(curr_cluster_info-1)

#     # if curr_cluster_info > 1:
#     #     my_sse = (curr_cluster_info * my_sse)/(curr_cluster_info-1)
#     my_sse = sse[:, curr_cluster]
#     all_indices = []
    
#     for ot_cluster in range(len(new_centroids)):
                        
#         if ot_cluster != curr_cluster:
            
#             # print(curr_cluster, ot_cluster)
#             size_ot_cluster = cluster_info[ot_cluster]
#             sse[:, ot_cluster] = (size_ot_cluster * sse[:, ot_cluster])/(size_ot_cluster+1)
#             ot_sse = sse[:, ot_cluster]

#             temp_indices = np.where(ot_sse < my_sse)[0]
            
#             # Update the cluster membership for the data point
#             if len(temp_indices) > 0:
#                 # temp_indices2 = he_data_indices[temp_indices].tolist()
#                 # assigned_clusters[temp_indices2] = ot_cluster

#                 # distances[temp_indices2] = np.min(ot_sse[temp_indices,])
#                 # distances[temp_indices2] = ot_sse[temp_indices]
#                 all_indices += list(temp_indices)
    
#     if len(all_indices) > 0 :
#         all_indices = np.unique(all_indices)
#         min_dist = np.min(sse_copy[all_indices, :], axis=1)
#         new_clus = np.argmin(sse[all_indices, :], axis=1)
#         temp = he_data_indices[all_indices].tolist()
#         # print(temp)
#         assigned_clusters[temp] = new_clus
#         distances[temp] = min_dist

#     return assigned_clusters, distances


def calculate_sse_specific(data, new_centroids, he_data_indices, mysize, assigned_clusters, 
curr_cluster, cluster_info):

    sse = distance.cdist(data[he_data_indices, :], new_centroids, 'sqeuclidean')
    sse_copy = np.zeros(shape=sse.shape)
    sse_copy[:] = sse[:]

    if mysize > 1:
        sse[:, curr_cluster] = (mysize * sse[:, curr_cluster])/(mysize-1)

    all_indices = []
    
    for ot_cluster in range(len(new_centroids)):
                        
        if ot_cluster != curr_cluster:
            
            # size_ot_cluster = len(np.where(assigned_clusters == ot_cluster)[0])
            sse[:, ot_cluster] = (cluster_info[ot_cluster] * sse[:, ot_cluster])/(cluster_info[ot_cluster]+1)

            temp_indices = np.where(sse[:, ot_cluster] < sse[:, curr_cluster])[0]
            
            # Update the cluster membership for the data point
            if len(temp_indices) > 0:
                all_indices += list(temp_indices)
    

    if len(all_indices) > 0 :
        all_indices = np.unique(all_indices)
        new_clus = np.argmin(sse[all_indices, :], axis=1)
        temp = he_data_indices[all_indices].tolist()
        assigned_clusters[temp] = new_clus

        # print("\n")
        # print("Actual Change: ", temp)
        # print("Predicted: ", he_data_indices)

    return assigned_clusters


def calculate_centroids(data, assigned_clusters):

    temp = [np.mean(data[np.where(assigned_clusters == i),], axis=1)[0] for i in np.sort(np.unique(assigned_clusters))]
    new_centroids = np.array(temp)

    return np.round(new_centroids, 5)


def check_centroid_status(curr_cluster, new_centroids, centroids):
    
    for i in range(centroids.shape[0]):
        for j in range(len(centroids[0])):
            
            if new_centroids[i][j] != centroids[i][j]:
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


def sse_after_move(data, new_centroids, sse1, lowest_sse, index, curr_cluster, ot_cluster, size2):
    
    sse2 = (size2 * np.sum(np.square(data[index, :] - new_centroids[ot_cluster, :])))/(size2+1)
    status = False

    if sse2 < sse1:
        status = True
        print(index, curr_cluster, ot_cluster, sse1, sse2)
    return status


def get_quality(data, final_assign, final_centroids, num_clusters):

    final_sse = 0

    for i in range(num_clusters):
        indices = np.where(final_assign == i)[0]
        final_sse += np.sum(np.square(data[indices, :] - final_centroids[i, :]))
        # print(i, final_sse)

    final_sse /= num_clusters

    return final_sse



