########################
# DATASET UTILS
########################

import numpy as np
import csv
import os

def csv_2_numpy(filename, path, sep=',', type='int8'):
    """
    Utility to read a dataset in csv format into a numpy array
    """
    file_path = os.path.join(path, filename)
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    array = np.array(x).astype(type)
    return array


def load_train_valid_test_csvs(dataset_name,
                               path,
                               sep=',',
                               type='int32',
                               suffix='data',
                               splits=['train',
                                       'valid',
                                       'test'],
                               verbose=True):
    """
    Loading training, validation and test splits by suffix from csv files
    """

    csv_files = ['{0}.{1}.{2}'.format(dataset_name, ext, suffix) for ext in splits]

    dataset_splits = [csv_2_numpy(file, path, sep, type) for file in csv_files]

    if verbose:
        print('Dataset splits for {0} loaded'.format(dataset_name))
        for data, split in zip(dataset_splits, splits):
            print('\t{0}:\t{1}'.format(split, data.shape))

    return dataset_splits

def load_dataset(dataset):
    if dataset == "toy":
        path = 'datasets/toy/%s' % dataset
    elif dataset[:4] == "amzn":
        path = 'datasets/amzn/%s' % dataset
    else:
        path = 'datasets/DEBD/%s' % dataset
    
    train, valid, test = load_train_valid_test_csvs(dataset, path=path)
    return train, valid, test
