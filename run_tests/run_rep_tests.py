from utils.dataIO import *
from base.KMeans import *
import time
from pathlib import Path
import gc

'''
Algo1: KMeans
Algo2: DEKMeans, 1 step look-back
Algo3: DEKMeans, probabilistic 1 step look back
Algo4: DEKMeans, Stochastic 1 step look back
'''

# file_dicts = {0: ['10k_50_5.txt', '10k_50_5_centers.txt'],
#               1: ['50k_50_5.txt', '50k_50_5_centers.txt'],
#               2: ['100k_50_5.txt', '100k_50_5_centers.txt'],
#               3: ['200k_50_5.txt', '200k_50_5_centers.txt'],
#               4: ['300k_50_5.txt', '300k_50_5_centers.txt'],
#               5: ['400k_50_5.txt', '400k_50_5_centers.txt'],
#               6: ['500k_50_5.txt', '500k_50_5_centers.txt']}

file_list = ['10_clusters.csv', '50_clusters.csv']


seed_array = np.array(np.random.randint(1, 10000, 70)).reshape(7, 10)
seeds = [546, 123, 74125]

# Set parameters
threshold = 0.01
num_iterations = 100
data_threshold = 3

avg_result_df = pd.DataFrame(columns=["Algorithm", 'Num_clusters', 'Runtime', 'Iterations',
                          'ARI', 'Centroids_dev'])

all_results_df = pd.DataFrame(columns=["Algorithm", 'Num_clusters', 'Runtime', 'Iterations',
                                'ARI', 'Centroids_dev'])
algorithms = ['KMeans', 'DEKMeans-LB', 'DEKMeans-PROB', 'DEKMeans-STO']

# Make changes for adjusting the current directory here
file_path = os.path.join(Path(__file__).parents[1], "benchmark", "clustering_data")

# for i in range(len(file_dicts.keys())):
for data_file in file_list:

    # data_file = file_dicts[i][0]
    # data, labels = read_data(data_file)
    # num_clusters = int(data_file.split(".")[0].split("_")[2])

    data, labels = read_simulated_data(os.path.join(file_path, data_file))
    num_clusters = int(data_file.split(".")[0].split("_")[0])

    for i_seed in seeds:

        # centroids2 = init_centroids(data, num_clusters, seeds[i])

        km_start_time = time.time()
        km_centroids, km_iter = Kmeans(data, num_clusters, threshold, num_iterations, [], False, i_seed)
        km_TraningTime = round(time.time() - km_start_time, 2)

        kmlb_start_time = time.time()
        kmlb_centroids, kmlb_iter = DEKmeans_lb(data, num_clusters, threshold, data_threshold, num_iterations, i_seed)
        kmlb_TraningTime = round(time.time() - kmlb_start_time, 2)

        kmprob_start_time = time.time()
        kmprob_centroids, kmprob_iter = DEKmeans_prob(data, num_clusters, threshold, data_threshold,
                                                 num_iterations, i_seed)
        kmprob_TraningTime = round(time.time() - kmprob_start_time, 2)

        kmsto_start_time = time.time()
        kmsto_centroids, kmsto_iter = DEKmeans_sto(data, num_clusters, threshold, data_threshold,
                                               num_iterations, i_seed)
        kmsto_TraningTime = round(time.time() - kmsto_start_time, 2)

        km_ari = check_ARI(pred_membership(data, km_centroids), labels)
        kmlb_ari = check_ARI(pred_membership(data, kmlb_centroids), labels)
        kmsto_ari = check_ARI(pred_membership(data, kmsto_centroids), labels)
        kmprob_ari = check_ARI(pred_membership(data, kmprob_centroids), labels)

        km_amis = check_amis(pred_membership(data, km_centroids), labels)
        kmlb_amis = check_amis(pred_membership(data, kmlb_centroids), labels)
        kmsto_amis = check_amis(pred_membership(data, kmsto_centroids), labels)
        kmprob_amis = check_amis(pred_membership(data, kmprob_centroids), labels)

        kmlb_dev = np.sqrt(np.mean(np.square(km_centroids - kmlb_centroids)))
        kmsto_dev = np.sqrt(np.mean(np.square(km_centroids - kmsto_centroids)))
        kmprob_dev = np.sqrt(np.mean(np.square(km_centroids - kmprob_centroids)))

        all_results_df = all_results_df.append({'Algorithm': 'KMeans', 'Num_clusters': num_clusters,
                                                'Runtime': km_TraningTime, 'Iterations': km_iter,
                                  'ARI': km_ari, 'Centroids_dev': 0}, ignore_index=True)

        all_results_df = all_results_df.append({'Algorithm': 'DEKMeans-LB', 'Num_clusters': num_clusters,
                                                'Runtime': kmlb_TraningTime, 'Iterations': kmlb_iter,
                                  'ARI': kmlb_ari, 'Centroids_dev': kmlb_dev}, ignore_index=True)

        all_results_df = all_results_df.append({'Algorithm': 'DEKMeans-PROB', 'Num_clusters': num_clusters,
                                                'Runtime': kmprob_TraningTime, 'Iterations': kmprob_iter,
                                  'ARI': kmprob_ari, 'Centroids_dev': kmprob_dev}, ignore_index=True)

        all_results_df = all_results_df.append({'Algorithm': 'DEKMeans-STO', 'Num_clusters': num_clusters,
                                                'Runtime': kmsto_TraningTime, 'Iterations': kmsto_iter,
                                  'ARI': kmsto_ari, 'Centroids_dev': kmsto_dev}, ignore_index=True)


    # Take the average of 10 runs and write to a file
    # and also store in the result dataframe

    for algo in algorithms:

        subset = all_results_df.loc[all_results_df['Algorithm'] == algo, ['Runtime', 'Iterations',
                          'ARI', 'Centroids_dev']]

        subset = subset.mean().values

        avg_result_df = avg_result_df.append({'Algorithm': algo, 'Num_clusters': num_clusters,
                                              'Runtime': subset[0], 'Iterations': subset[1],
                                  'ARI': subset[2], 'Centroids_dev': subset[3]},
                                             ignore_index=True)

    gc.collect()

    print("Completed the analysis for", num_clusters, "clusters")

## Write the results to a file

avg_result_df.to_csv("avg_clustering_experiments.csv", index=False, sep="\t")
all_results_df.to_csv("all_results.csv", index=False, sep="\t")
