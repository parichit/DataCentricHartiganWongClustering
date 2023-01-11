import sys
sys.path.append("../")

from utils.dataIO import *
from base.KMeans import *
from base.DCKmeans import *
from pathlib import Path
import time
from utils.assign_clusters import *
<<<<<<< HEAD
import pandas as pd
from sklearn.cluster import kmeans_plusplus
=======
import glob

>>>>>>> origin

'''
Algo1: KMeans
Algo2: DCKMeans
'''

# Set parameters
<<<<<<< HEAD
threshold = 0.0
=======
threshold = 0.001
>>>>>>> origin
num_iterations = 100

# file_list = ['test_data_case1.csv']
file_list = ['test_100_2_3.csv']
<<<<<<< HEAD
file_list = ['ijcnn.csv']
# file_list = ['magic.csv']
# file_list = ['user_knowledge_train.csv']
# file_list = ['hapt_train.csv']
# file_list = ['covertype.csv']
# file_list = ['spambase.csv']
# file_list = ['10_clusters.csv']

DATA_PATH = "/Users/schmuck/Documents/Box Sync/Ph.D./DATASETS"
=======
file_list = ['crop.csv']
# file_list = ['ms.csv']
file_list = ['user_knowledge_train.csv']
file_list = ['hapt_train.csv']
# file_list = ['covtype.csv']
# file_list = ['spambase.csv']
# file_list = ['magic.csv']
# file_list = ['5000000_points.csv']
>>>>>>> origin

data_path = "/Users/schmuck/Documents/OneDrive - Indiana University/Box Sync/PhD/DATASETS"

# Make changes for adjusting the current directory here
<<<<<<< HEAD
file_path = os.path.join(DATA_PATH, "benchmark", "clustering_data")
# file_path = os.path.join(Path(__file__).parents[1], "benchmark", "scal_data")
file_path = os.path.join(DATA_PATH, "real_data")
# file_path = os.path.join(DATA_PATH, "sample_data")

num_clusters = 100
seed = 12

for data_file in file_list:

    # data, labels = read_simulated_data(os.path.join(file_path, data_file))
    data = read_simulated_data123(os.path.join(file_path, data_file))
    # data = np.array(pd.read_csv(os.path.join(file_path, data_file), sep=" "))
=======
file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(Path(__file__).parents[1], "benchmark", "scal_data")
file_path = os.path.join(data_path, "real_data")
# file_path = os.path.join(data_path, "sample_data")
# file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(data_path, "data")

num_clusters = 5
seed = 1245

seeds = np.random.randint(1, 1200, 1000)
seeds = [1]
counter = 1

# file_list = glob.glob(os.path.join(data_path, file_path, "*"))

for data_file in file_list:

    data, labels = read_simulated_data(os.path.join(file_path, data_file))
    # data = np.load(os.path.join(file_path, "264792_4_0.001_1000000000_.npy"))
    # data = np.load(data_file)

>>>>>>> origin
    # data, labels = read_real_data(os.path.join(file_path, data_file), ["user_knowledge_train.csv"])
    # vis_PCA(data, labels)
    # exit(0)
    # np.random.seed(seed)
    # centers, _ = kmeans_plusplus(data, n_clusters=num_clusters, random_state=seed)
    # np.savetxt("ms_cen_ea.csv", centers, delimiter=" ")

<<<<<<< HEAD
    print("Data Shape :", data.shape)

    centers, indices = kmeans_plusplus(data, n_clusters=num_clusters, random_state=59)

    # df = pd.DataFrame(centers)
    # df.to_csv("crop_cen.csv", sep=",", index=False, header=False)

    print("Centers shape: ", centers.shape)

    kmlb_start_time = time.time()
    # _, _, _ = DCKMeans(data, num_clusters, threshold, num_iterations, seed)
    kmlb_centroids, kmlb_iter, dckm_calc = DCKMeans(data, num_clusters, threshold, num_iterations, centers, seed)
    kmlb_TraningTime = round(time.time() - kmlb_start_time, 2)

    centers, indices = kmeans_plusplus(data, n_clusters=num_clusters, random_state=59)

    km_start_time = time.time()
    km_centroids, km_iter = Kmeans(data, num_clusters, threshold, num_iterations, centers, seed)
    km_TraningTime = round(time.time() - km_start_time, 5)

    # print(kmlb_centroids)
    # print("Distance calculations by KMeans: ", num_clusters*data.shape[0]*km_iter)
    # print("Distance calculations by DCKMeans: ", dckm_calc)
=======
    # print("Counter: ", counter, " File: ", data_file)
    print(data.shape)

    for seed in seeds:

        km_start_time = time.time()
        km_centroids, km_iter, km_cacl = Kmeans(data, num_clusters, threshold, num_iterations, seed)
        km_TraningTime = round(time.time() - km_start_time, 2)

        kmlb_start_time = time.time()
        kmlb_centroids, kmlb_iter, dckm_calc = DCKMeans(data, num_clusters, threshold, num_iterations, seed)
        kmlb_TraningTime = round(time.time() - kmlb_start_time, 2)
>>>>>>> origin

        # if (km_iter!=kmlb_iter):
        #     print("Difference found: ", seed)
        #     break

<<<<<<< HEAD
    print(km_TraningTime, kmlb_TraningTime)
    # print(kmlb_centroids)
    # print(km_centroids)
    print("Dev: ", round(np.mean(np.square(km_centroids - kmlb_centroids))), 3)
=======
        # print("Distance calculations by KMeans: ", num_clusters*data.shape[0]*km_iter)
        # print("Distance calculations by DCKMeans: ", dckm_calc)

        # acc, new_labels1 = calc_raw_accuracy(labels, assign1, data)
        # print("KM: ", acc, check_ARI(new_labels1, labels))
        #
        # acc, new_labels2 = calc_raw_accuracy(labels, assign2, data)
        # print("KMLB: ", acc, check_ARI(new_labels2, labels))
        #
        # print("Diff in clustering: ", len(np.where(new_labels1 != new_labels2)[0]))
        # print("Diff in clustering: ", len(np.where(assign1 != assign2)[0]))
        # print(assign1[np.where(assign1 != assign2)[0]], assign2[np.where(assign1 != assign2)[0]])

        # print(kmlb_TraningTime)
        print(km_TraningTime, kmlb_TraningTime)
        print(km_cacl, dckm_calc)
        print("Dev: ", round(np.sqrt(np.mean(np.square(km_centroids - kmlb_centroids))), 3))

    counter += 1
>>>>>>> origin
