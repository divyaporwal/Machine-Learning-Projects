import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import graphviz 

class DataSet:
    def __init__(self, data_set):
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21), dtype=int, delimiter=',')

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))

if __name__ == '__main__':
    # Load a data set
    data = DataSet('mushroom')

    num_trees = 10
    seed = 7
    depth = [3, 5]
    num_trees = [10, 20]
    for (index, dep) in enumerate(depth):
        for (idx, num) in enumerate(num_trees):
            clf = tree.DecisionTreeClassifier(max_depth=dep, min_samples_split=2)
            bagmodel = BaggingClassifier(base_estimator=clf, n_estimators=num, random_state=seed)
            bagmodel = bagmodel.fit(data.examples['train'], data.labels['train'])
            y_predict = bagmodel.predict(data.examples['test'])
            tn, fp, fn, tp = confusion_matrix(data.labels['test'], y_predict).ravel()
            print("Confusion Matrix for bagging model using scikit-learn depth={0}, num_trees={1}".format(dep,num))
            print("      Positive  Negative")
            print("Positive "+str(tp)+" "+str(fn))
            print("Negative "+str(fp)+" "+str(tn))


    d = 1
    depth = [1, 2]
    num_trees = [20, 40]
    for (index, dep) in enumerate(depth):
        for (idx, num) in enumerate(num_trees):
            clf_stump = tree.DecisionTreeClassifier(max_depth=dep, min_samples_split=2)
            adaboost = AdaBoostClassifier(base_estimator=clf_stump,n_estimators=num)
            adaboost.fit(data.examples['train'], data.labels['train'])
            adaboost.score(data.examples['test'], data.labels['test'])
            y_predict = adaboost.predict(data.examples['test'])
            tn, fp, fn, tp = confusion_matrix(data.labels['test'], y_predict).ravel()
            print("Confusion Matrix for boosting model using scikit-learn depth={0}, num_trees={1}".format(dep,num))
            print("      Positive  Negative")
            print("Positive "+str(tp)+" "+str(fn))
            print("Negative "+str(fp)+" "+str(tn))