import sys
sys.path.append("../")

from utils.dataIO import *
from base.HWKmeans import *
from base.DCHWKmeans import *
from pathlib import Path
import time

'''
Algo1: KMeans
Algo2: DCKMeans
'''



# file_list = ['test_data_case1.csv']
file_list = ['50_2_10.csv']

# file_list = ['ijcnn.csv']
file_list = ['magic.csv']
# file_list = ['user_knowledge_train.csv']
# file_list = ['hapt_train.csv']
# file_list = ['covertype.csv']
# file_list = ['spambase.csv']
# file_list = ['crop.csv']

data_path = "/Users/schmuck/Documents/OneDrive - Indiana University/Box Sync/PhD/DATASETS"
# data_path = "/Users/schmuck/Library/CloudStorage/OneDrive-IndianaUniversity/Box Sync/PhD/DataCentricHartiganWongClustering"

# Make changes for adjusting the current directory here
file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(Path(__file__).parents[1], "benchmark", "scal_data")
file_path = os.path.join(data_path, "real_data")
# file_path = os.path.join(data_path, "sample_data")
# file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(data_path, "data")


# Set parameters
threshold = 0.001
num_iterations = 100
clusters = [i for i in range(1, 21)]
clusters = [5, 10, 15, 25, 30, 35]

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

        for num_clusters in clusters:

            print("\nNum clusters: ", num_clusters, "\n")

            hw_start_time = time.time()
            hw_centroids, hw_iter = HWKmeans(data, num_clusters, num_iterations, seed)
            hw_TraningTime = round(time.time() - hw_start_time, 5)

            dchw_start_time = time.time()
            dchw_centroids, dchw_iter = DCHWKmeans(data, num_clusters, num_iterations, seed)
            dchw_TraningTime = round(time.time() - dchw_start_time, 5)

            dev = np.sum(np.square(hw_centroids- dchw_centroids))
            
            if dev != 0:
                print("Deviation not zero for: ", num_clusters )
                break
            else:
                print(dev)

        # print("Time", dchw_TraningTime)
        # print(km_cacl, dckm_calc)
        # print("Dev: ", round(np.sqrt(np.mean(np.square(km_centroids - kmlb_centroids))), 3))

    counter += 1

