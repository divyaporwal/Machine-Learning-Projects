import cv2
from sklearn import metrics
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
Image segmentation using spectral clustering

"""
img = cv2.imread('seg.jpg', 0)
original_shape = img.shape
X = img
X = X/255
A = metrics.pairwise.rbf_kernel(X, X, gamma=3400)
d = np.sum(A, axis=0)
D = np.diag(d)
L = D - A
centered_matrix = L - L.mean(axis=1)[:, np.newaxis]
cov = np.dot(centered_matrix, centered_matrix.T)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = eigvals.argsort()[::1]
eigenVectors = eigvecs[:, idx]
samples = np.column_stack([eigenVectors.flatten()])
kmeans = KMeans(n_clusters=2).fit(samples)
samples = np.column_stack([X.flatten()])
labels = kmeans.predict(samples).reshape(X.shape)
plt.imshow(labels)
plt.show()
