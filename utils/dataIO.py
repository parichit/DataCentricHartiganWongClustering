import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path


def read_data(input_loc):

    data = pd.read_csv(input_loc, header=0, sep="\t")

    # Get the label column from the data
    labels = list(data['labels'].values)
    data.drop(['labels'], inplace=True, axis=1)

    # Subset the feature columns from the data
    data = np.array(data, dtype=float)

    # Cast labels to a numpy array
    labels = np.array(labels)
    print("Data shape: ", data.shape)
    return data, labels



def read_simulated_data(file_path):

    data = pd.read_csv(file_path, sep=",")
    labels = data['labels'].to_list()
    data.drop(["labels"], inplace=True, axis=1)

    return np.array(data), labels


def read_simulated_data123(file_path):
    data = pd.read_csv(file_path, sep=",")
    return np.array(data)


def write_result_data(result, output_loc, result_type):

    if result_type == "clustering_results":
        header = "Clusters,Time,Iterations,ARI,Algorithm\n"
        outfile = "clustering_benchmark.csv"
    elif result_type == "scal_results":
        header = "Num_Points,Time,Iterations,ARI,Algorithm\n"
        outfile = "scalability_benchmark.csv"
    elif result_type == "dims_results":
        header = "Dimensions,Time,Iterations,ARI,Algorithm\n"
        outfile = "dimsionality_benchmark.csv"

    with open(os.path.join(output_loc, outfile), "w") as file:

        file.write(header)

        for i in range(len(result)):
            file.write(str(result[i][0])+","+str(result[i][1])+","+str(result[i][2]) + "," +
                              str(result[i][3])+","+result[i][4] + "\n")

    print(result_type, ": File written to disk")


def read_real_data(dataset_loc, dataset_name):

    if dataset_name[0] == "spambase.csv":

        data = pd.read_csv(dataset_loc, sep=",")
        # data = data.sample(frac=1).reset_index(drop=True)

        labels = list(data['labels'].values)
        data.drop("labels", axis=1, inplace=True)

        # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=4152, test_size=0.1)

        return np.array(data), labels

    elif dataset_name[0] == "magic.csv":

        data = pd.read_csv(dataset_loc, sep=",")
        # data = data.sample(frac=1).reset_index(drop=True)

        labels = list(data['labels'].values)
        data.drop("labels", axis=1, inplace=True)

        # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1475, test_size=0.1)

        return np.array(data), labels

    elif dataset_name[0] == "hapt_train.csv":

        data = pd.read_csv(dataset_loc, sep=",")
        # data = data.sample(frac=1).reset_index(drop=True)

        labels = list(data['labels'].values)
        data.drop("labels", axis=1, inplace=True)

        # test_data = pd.read_csv(os.path.join(os.path.dirname(dataset_name), dataset_name[2]), sep=",")
        # test_labels = list(test_data['labels'].values)
        # test_data.drop("labels", axis=1, inplace=True)
        #
        # train_data = np.array(train_data)
        # test_data = np.array(test_data)

        return np.array(data), labels

    elif dataset_name[0] == "user_knowledge_train.csv":

        data = pd.read_csv(dataset_loc, sep=",")
        # data = data.sample(frac=1).reset_index(drop=True)

        labels = list(data["UNS"].values)
        data.drop("UNS", axis=1, inplace=True)

        # test_data = pd.read_csv(os.path.join(os.path.dirname(dataset_loc), dataset_name[2]), sep=",")
        # test_labels = list(test_data["UNS"].values)
        # test_data.drop("UNS", axis=1, inplace=True)

        return np.array(data), labels

    elif dataset_name[0] == "crop.csv":

        data = pd.read_csv(dataset_loc, sep=",")
        # data = data.sample(frac=1).reset_index(drop=True)

        labels = list(data['labels'].values)
        data.drop(['labels'], inplace=True, axis=1)

        # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=5532, test_size=0.1)

        return np.array(data), labels




