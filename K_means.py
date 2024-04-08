import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display_functions import clear_output
from sklearn.decomposition import PCA


df = pd.read_excel("US_population_dataset_cor.xlsx")
df = df.drop(df.columns[-1], axis=1)

index_split = int(len(df) * 0.7)
x_trains = df[: index_split]
x_tests = df[index_split:]

class K_MEANS:
    def __init__(self, k):
        self.k = k

    def fit(self, x, iterations):
        centroids_old = pd.DataFrame()
        centroids = K_MEANS.random_centroid(x, self.k)

        for i in range(iterations):
            if all(a == b for a, b in zip(centroids, centroids_old)):
                break
            centroids_old = centroids
            label = K_MEANS.get_labels(x, centroids_old)
            centroids = K_MEANS.new_centroid(x, label)
            K_MEANS.plot_cluster(x,label,centroids_old,i)


    def predict(self, y, iteration):
       pass


    @staticmethod
    def random_centroid(data, k):
        centroids = []
        for i in range(k):
            centroid = data.sample().values.tolist()
            centroids.append(centroid)
        return centroids

    @staticmethod
    def get_labels(data, centroids):
        distances = []
        for centroid in centroids:
            distance = np.sqrt(((data - centroid) ** 2).sum(axis=1))
            distances.append(distance)
        return np.argmin(np.array(distances), axis=0)

    @staticmethod
    def new_centroid(data, label):
        return data.groupby(label).apply(lambda x: np.exp(np.log(x).mean())).values

    @staticmethod
    def plot_cluster(data, labels, centroids, iteration):
        pca=PCA(2)
        data=pca.fit_transform(data)
        centroids=pca.transform(centroids.T)
        clear_output(wait=True)
        plt.title(f'Iteration {iteration}')
        plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
        plt.scatter(x=centroids[:, 0], y=centroids[:, 1])
        plt.show()


if __name__ == "__main__":
    K_MEANS_model = K_MEANS(k=3)
    K_MEANS_model.fit(x_trains, 10)
