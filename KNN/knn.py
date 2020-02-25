import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features = None
        self.labels = None

    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels

    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        ret_list = []
        for sbl in features:
            list_of_labels = KNN.get_k_neighbors(self, sbl)
            zeros = 0
            ones = 0
            for lbl in list_of_labels:
                if lbl == 0:
                    zeros = zeros + 1
                else:
                    ones = ones + 1
            if zeros >= ones:
                ret_list.append(0)
            else:
                ret_list.append(1)

        return ret_list

    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        dict_of_dists = dict()
        i = 0
        for sbl in self.features:
            dict_of_dists[i] = self.distance_function(point, sbl)
            i = i + 1

        list_of_dists = []
        for val in dict_of_dists.values():
            list_of_dists.append(val)

        list_of_dists.sort()

        set_of_inds = set()

        kftp = min(self.k, len(list_of_dists))

        for j in range(kftp):
            dist = list_of_dists[j]
            for idx, dts in dict_of_dists.items():
                if dist == dts and idx not in set_of_inds:
                    set_of_inds.add(idx)
                    break

        list_of_inds = []
        for idx in set_of_inds:
            list_of_inds.append(idx)

        list_of_inds.sort()

        ret_list = []
        for idx in list_of_inds:
            ret_list.append(self.labels[idx])

        return ret_list


if __name__ == '__main__':
    print(np.__version__)

