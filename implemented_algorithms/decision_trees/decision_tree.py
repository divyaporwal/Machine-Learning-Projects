import numpy as np
import math
import matplotlib.pyplot as plt

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
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, usecols=(0, 1, 2, 3, 4, 5, 6), dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    x_dict = {}
    unique_x = np.unique(x)

    for x_attr in unique_x:
        x_dict[x_attr] = np.where(x == x_attr)[0]

    return x_dict

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    sum = y.sum()
    y0 = (y == 0).sum()
    y1 = (y == 1).sum()
    total = y.size
    p0 = y0/total
    p1 = y1/total
    if (p0 == 0):
        p0log = 0
    else:
        p0log = math.log(p0, 2)
    
    if (p1 == 0):
        p1log = 0
    else:
        p1log = math.log(p1, 2)

    entrpy = -1*(p0 * p0log + p1 * p1log)
    return entrpy

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    entrpy = entropy(y)
    unique_x = np.unique(x)
    x_set = partition(x)
   
    p_y_conditional = 0;
    for x_label in unique_x:
        x_single = x_set[x_label]
        y_val = y[x_single[0:]]
        y_0 = (y_val == 0).sum()
        y_1 = (y_val == 1).sum()
        p_y_0 = y_0/y_val.size
        p_y_1 = y_1/y_val.size
       
        if p_y_0 == 0:
            p_y_0_x = 0
        else:
            p_y_0_x = p_y_0*math.log(p_y_0, 2)

        if p_y_1 == 0:
            p_y_1_x = 0
        else:
            p_y_1_x = p_y_1*math.log(p_y_1, 2)

        p_y_x = x_single.size/y.size*(-1)*(p_y_0_x + p_y_1_x)
        p_y_conditional = p_y_conditional + p_y_x
    
    information_gain = entrpy - p_y_conditional
    return information_gain

def get_best_attribute(x, y, attributes):
    best = attributes[0]
    maxGain = 0.0
    
    attr = 0
    bestAttr = 0
    for x_col in x.T:
        if not attr in attributes:
            attr = attr + 1
            continue
        gain = mutual_information(x_col, y)
        
        if gain >= maxGain:
            maxGain = gain
            bestAttr = attr
        attr = attr + 1

    return bestAttr

def id3(x, y, attributes, max_depth, depth=0):
    if np.unique(y).size == 1:
        return y[0]
    elif (len(attributes) == 0) or (depth == max_depth) or (len(x) == 0):
        counts = np.bincount(y)
        return np.argmax(counts)
    else:
        best_attr = get_best_attribute(x, y, attributes)
        tree = {}
        x_set = partition(x[:,best_attr])
        for x_val in x_set:
            new_attr = attributes[:]
            new_attr.remove(best_attr)
            idx = x_set[x_val]
            subtree = id3(x[idx], y[idx], new_attr, max_depth, depth+1)

            if (best_attr, x_val) in tree:
                tree[best_attr, x_val].append(subtree)
            else:
                tree[best_attr, x_val] = subtree

    return tree

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    for index, val in enumerate(x):
        key = (index, val)

        if key in list(tree.keys()):
            try:
                result = tree[key]
            except:
                return default_output

            result = tree[key]

            if isinstance(result, dict):
                return predict_example(x, result)
            else:
                return result
            
def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return 1/y_true.size*(np.sum(y_true != y_pred))

def confusion_matrix(true, predicted):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    w, h = 2, 2
    matrix = [[0 for x in range(w)] for y in range(h)]
    for t, p in zip(true, predicted):
        if t == 1 and p == 1:
            true_positive =  true_positive + 1
        elif t == 0 and p == 0:
            true_negative = true_negative + 1
        elif t == 1 and p == 0:
            false_negative = false_negative + 1
        elif t == 0 and p == 1:
            false_positive = false_positive + 1

    matrix[0][0] = true_positive
    matrix[0][1] = false_negative
    matrix[1][0] = false_positive
    matrix[1][1] = true_negative
    return matrix

def visualize(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def check_features(x):
    datasize = len(x)
    threshold = (int)(datasize*0.05)
    index = 0

    for x_col in x.T:
        if np.unique(x_col).size > threshold:
            x_col = discretize_data(x_col)
            
        x[:,index] = x_col
        index = index + 1

    return x

def discretize_data(x):
    mean = np.mean(x)
    idx = 0

    for val in x:
        if (val > mean):
            x[idx] = 1
        else:
            x[idx] = 0
        idx = idx + 1
    return x

if __name__ == '__main__':

    # Load a data set
    check_data = "false"
    data = DataSet('monks-1')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))
    
    # Learn a decision tree of depth 3
    if check_data == "true":
        data.examples['train'] = check_features(data.examples['train'])
        data.examples['test'] = check_features(data.examples['test'])

    # Add more values in array to generate trees of different depth
    depth = [3, 4, 5]
    testArray = []
    trainArray = []
    for dep in depth:
        decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx.tolist(), dep)
        visualize(decision_tree)

    # Compute the training error
        trn_pred = [predict_example(data.examples['train'][i, :], decision_tree) for i in range(data.num_train)]
        trn_err = compute_error(data.labels['train'], trn_pred)
        trainArray.append(trn_err)
    
    # Compute the test error
        tst_pred = [predict_example(data.examples['test'][i, :], decision_tree) for i in range(data.num_test)]
        tst_err = compute_error(data.labels['test'], tst_pred)
        testArray.append(tst_err)
        matrix = confusion_matrix(data.labels['test'], tst_pred)
        print('d={0} trn={1}, tst={2}'.format(dep, trn_err, tst_err))

    plt.plot(depth, trainArray, label="Training Error", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='none')
    plt.plot(depth, testArray, label="Test Error", marker ='o', markerfacecolor ='b', markeredgecolor ='b', linestyle ='none')
    plt.xlabel('Depth x-axis')
    plt.ylabel('Error y-axis')
    plt.legend(loc='upper right')
    plt.title('Depth vs Error for monks-1')
    plt.show()
