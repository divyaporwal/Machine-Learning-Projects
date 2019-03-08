from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture

def spectral_clustering(x, k):
    X = x[0]
    Y = x[1]
    features = 2
    A = metrics.pairwise.rbf_kernel(X, X, gamma=100)
    d = np.sum(A, axis=0)
    D = np.diag(d)
    L = D - A
    centered_matrix = L - L.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = eigvals.argsort()[::1]
    eigenVectors = eigvecs[:, idx]
    newX = np.column_stack((eigenVectors[:,0], eigenVectors[:,1]))
    krows = eigenVectors[0:features:1]
    Xdat = np.matmul(X, krows)
    Xreal = Xdat.real
    kmeans = KMeans(n_clusters=k).fit(newX.real)
    y_kmeans = kmeans.predict(newX.real)
    plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    
n_samples = 1500
x = datasets.make_circles(n_samples =1500, factor=.5, noise=.05)
X = x[0]
Y = x[1]

# Running spectral_clustering on the circles data
spectral_clustering(x, 2)

# Running kmeans clustering on circles data
kmeans = KMeans(n_clusters=2).fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
