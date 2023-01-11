import sklearn.datasets as skd
import pandas as pd
from pathlib import Path
import os

file_path = os.path.join(Path(__file__).parents[1], "sample_data")

data, labels, centers = skd.make_blobs(n_samples=100,
                             n_features=5,
                             centers=10,
                             return_centers=True)


# vis_PCA(data, label)
# view_3d(data, labels)

data = pd.DataFrame(data)
data['labels'] = labels
centers = pd.DataFrame(centers)
data.to_csv(os.path.join(file_path, "test_100_5_10.csv"), sep=",", index=False)