import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import graphviz

class DataSet:
    def __init__(self, data_set):
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None, 'valid' : None}

        # The training and test examples
        self.examples = {'train': None, 'test': None, 'valid' : None}

        # Load all the data for this data set
        for data in ['train', 'test', 'valid']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.num_valid = self.examples['valid'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, dtype=float, delimiter=',')

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))

def plot_digits(data):
    plt.clf()
    fig, axes = plt.subplots(4, 4, figsize=(15, 5),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(16, 16))
        print("plotting "+str(i))
    plt.show()

def compute_error(y_true, y_pred, weights=None):
    if weights is None:
        weights = [1] * len(y_true)
    
    weights = np.array(weights)
    wsum = np.sum(weights)
    error_idx = np.where(y_true != y_pred)
    error_val = np.sum(weights[error_idx])
    return error_val/wsum
        
if __name__ == '__main__':
	"""
	Performing PCA decomposition for dimensionality reduction and then running 
	SDG classifier with hinge loss to create a SVM
	
	"""
    data = DataSet('usps')
    X = data.examples['train']
    h, w = X.shape
    Y = data.labels['train']
    n_components = 50
    
    alp = [0.1, 0.01, 0.001, 0.0001]
    x_v = [0.7, 0.8, 0.9, 1]

    print("X  |  Alpha  | Test Error")
    for alph in alp:
        for var in x_v:
            pca = PCA(var)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_new = pca.inverse_transform(X_pca)
            if var == 1:
                X_new = X
            
            clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, alpha=alph)
            clf.fit(X_new, Y)
            y_pred = clf.predict(data.examples['test'])
            pred_err = compute_error(data.labels['test'], y_pred)
            print(str(var*100)+"   "+str(alph)+"   "+str(pred_err))
            

    print("X  |  Alpha  | Validation Error")
    for alph in alp:
        for var in x_v:
            pca = PCA(var)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_new = pca.inverse_transform(X_pca)
            if var == 1:
                X_new = X
            
            clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, alpha=alph)
            clf.fit(X_new, Y)
            y_pred = clf.predict(data.examples['valid'])
            pred_err = compute_error(data.labels['valid'], y_pred)
            print(str(var*100)+"    "+str(alph)+"    "+str(pred_err))