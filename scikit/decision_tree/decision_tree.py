import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
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
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0,
                                          usecols=(0, 1, 2, 3, 4, 5, 6), dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))

if __name__ == '__main__':
    # Load a data set
    data = DataSet('monks-1')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))

    # Learn a decision tree of depth 3
    d = 3
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data.examples['train'], data.labels['train'])
    y_predict = clf.predict(data.examples['test'])

    tn, fp, fn, tp = confusion_matrix(data.labels['test'], y_predict).ravel()
    print("Confusion Matrix using scikit-learn")
    print("      Positive  Negative")
    print("Positive "+str(tp)+" "+str(fn))
    print("Negative "+str(fp)+" "+str(tn))
    
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render()
    graph.view()