import sys
sys.path.append("../")

from utils.dataIO import *
from base.HWKmeans import *
from base.DCHWKmeans import *
from base.test_both_algos import *
from pathlib import Path
import time

'''
Algo1: HWKMeans
Algo2: HWDCKMeans
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

data_path = "/Users/schmuck/Documents/OneDrive - Indiana University/Box Sync/PhD/DATASETS/"
# data_path = "/Users/schmuck/Library/CloudStorage/OneDrive-IndianaUniversity/Box Sync/PhD/DataCentricHartiganWongClustering"

# Make changes for adjusting the current directory here
file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(Path(__file__).parents[1], "benchmark", "scal_data")
file_path = os.path.join(data_path, "real_data")
# file_path = os.path.join(data_path, "sample_data")
# file_path = os.path.join(data_path, "clustering_data")
# file_path = os.path.join(data_path, "data")


# Set parameters
num_iterations = 100
clusters = [i for i in range(5, 30, 5)]
clusters = [11]
 
# seeds = np.random.randint(1, 1200, 1000)
seeds= [9]


for data_file in file_list:

    data = read_simulated_data123(os.path.join(file_path, data_file))
    print("Data Shape :", data.shape)

    for seed in seeds:

        for num_clusters in clusters:

            print("\nNum clusters: ", num_clusters, "\n")

            hw_start_time = time.time()
            hw_centroids, hw_iter, hw_sse, hw_labels = HWKmeans(data, num_clusters, num_iterations, seed)
            hw_TraningTime = round(time.time() - hw_start_time, 5)
            # print(hw_centroids)

            dchw_start_time = time.time()
            dchw_centroids, dchw_iter, dchw_sse, dchw_labels = DCHWKmeans(data, num_clusters, num_iterations, seed)
            dchw_TraningTime = round(time.time() - dchw_start_time, 5)

            dev = np.sum(np.square(hw_centroids - dchw_centroids))
            
            if dev != 0:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("Deviation is ", dev, " for ",  num_clusters, " clusters")
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
                
                # temp1 = check_ARI(labels, hw_labels)
                # temp2 = check_ARI(labels, dchw_labels)
                temp =  check_ARI(hw_labels, dchw_labels)
                print("ARI: ", temp)

            print(dchw_sse)
            print("Time", hw_TraningTime, dchw_TraningTime, hw_sse, dchw_sse)

            # hw_start_time = time.time()
            # hw_centroids, hw_iter, hw_labels = HWKmeans_test123(data, num_clusters, num_iterations, seed)
            # hw_TraningTime = round(time.time() - hw_start_time, 5)

