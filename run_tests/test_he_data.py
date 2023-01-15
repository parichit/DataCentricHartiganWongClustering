import sys
sys.path.append("../")

from utils.dataIO import *
from base.HWKmeans import *
from pathlib import Path
import time

'''
Algo1: KMeans
Algo2: DCKMeans
'''



# file_list = ['test_data_case1.csv']
file_list = ['50_2_10.csv']

# file_list = ['ijcnn.csv']
# file_list = ['magic.csv']
# file_list = ['user_knowledge_train.csv']
# file_list = ['hapt_train.csv']
# file_list = ['covertype.csv']
# file_list = ['spambase.csv']
# file_list = ['crop.csv']

data_path = "/Users/schmuck/Documents/OneDrive - Indiana University/Box Sync/PhD/DATASETS"
data_path = "/Users/schmuck/Library/CloudStorage/OneDrive-IndianaUniversity/Box Sync/PhD/DataCentricHartiganWongClustering"

# Make changes for adjusting the current directory here
file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(Path(__file__).parents[1], "benchmark", "scal_data")
file_path = os.path.join(data_path, "real_data")
file_path = os.path.join(data_path, "sample_data")
# file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(data_path, "data")


# Set parameters
threshold = 0.001
num_iterations = 100
num_clusters = 8

seed = 1245

seeds = np.random.randint(1, 1200, 1000)
seeds = [1]
counter = 1


for data_file in file_list:

    data, labels = read_simulated_data(os.path.join(file_path, data_file))
    # data = np.load(os.path.join(file_path, "264792_4_0.001_1000000000_.npy"))
    # data = np.load(data_file)
    print("Data Shape :", data.shape)

    for seed in seeds:
        # kmdc_start_time = time.time()
        # kmdc_centroids, kmdc_iter, dckm_calc = DCKMeans(data, num_clusters, threshold, num_iterations, centers, seed)
        # kmdc_TraningTime = round(time.time() - kmdc_start_time, 2)

        km_start_time = time.time()
        km_centroids, km_iter = HWKmeans_test(data, num_clusters, num_iterations, seed)
        km_TraningTime = round(time.time() - km_start_time, 5)

        # km_start_time = time.time()
        # km_centroids, km_iter = HWKmeans_1(data, num_clusters, num_iterations, seed)
        # km_TraningTime = round(time.time() - km_start_time, 5)

        print(km_centroids)
        # print("Time", km_TraningTime)
        # print(km_cacl, dckm_calc)
        # print("Dev: ", round(np.sqrt(np.mean(np.square(km_centroids - kmlb_centroids))), 3))

    counter += 1

