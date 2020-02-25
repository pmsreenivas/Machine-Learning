import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    x = np.array(real_labels)
    y = np.array(predicted_labels)

    xy = x * y

    A = sum(xy)
    B = sum(x)
    C = sum(y)

    if B == 0 and C == 0:
        return 1

    dist = (2 * A) / (B + C)
    return dist


class Distances:
    @staticmethod
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = point1
        y = point2
        nr = []
        dr = []
        for i in range(len(x)):
            nr.append(abs(x[i] - y[i]))
            dr.append(abs(x[i]) + abs(y[i]))
        v = []
        for i in range(len(nr)):
            if dr[i] != 0:
                v.append(nr[i] / dr[i])
            else:
                v.append(0)
        dist = sum(v)
        return dist

    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = point1
        y = point2
        nr = []
        for i in range(len(x)):
            nr.append(abs(x[i] - y[i]))
        for i in range(len(nr)):
            nr[i] = nr[i] ** 3
        dist = sum(nr)
        dist = dist ** (1 / 3)
        return dist

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = np.array(point1)
        y = np.array(point2)
        z = x - y
        zz = z * z

        return np.sqrt(sum(zz))

    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        x = np.array(point1)
        y = np.array(point2)
        return sum(x * y)

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        nr = Distances.inner_product_distance(point1, point2)
        x = np.array(point1)
        y = np.array(point2)

        xx = x * x
        yy = y * y

        dr1 = np.sqrt(sum(xx))
        dr2 = np.sqrt(sum(yy))

        cs = nr / (dr1 * dr2)
        cd = 1 - cs

        return cd

    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        v = Distances.euclidean_distance(point1, point2)

        vv = v * v
        vvv = -0.5 * vv
        rvp = np.exp(vvv)
        rv = -1 * rvp
        return rv


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        list_of_models = []
        for fnc in set(distance_funcs.values()):
            for j in range(1, 30, 2):
                model = KNN(j, fnc)
                model.train(x_train, y_train)
                predicted_labels = model.predict(x_val)
                f1s = f1_score(y_val, predicted_labels)
                list_of_models.append((fnc, j, f1s, model))
        maxf1 = float('-inf')
        for _, _, f1s1, _ in list_of_models:
            if f1s1 > maxf1:
                maxf1 = f1s1
        filtered = []
        for ele in list_of_models:
            (fnc2, k2, f1s2, mdl2) = ele
            if f1s2 == maxf1:
                filtered.append((fnc2, k2, f1s2, mdl2))

        if len(filtered) == 1:
            (fnc3, k3, f1s3, mdl3) = filtered[0]
            self.best_k = k3
            for k, v in distance_funcs.items():
                if v == fnc3:
                    self.best_distance_function = k
                    break
            self.best_model = mdl3
            return

        ca_cands = []
        mi_cands = []
        eu_cands = []
        ga_cands = []
        in_cands = []
        co_cands = []

        for ele in filtered:
            (fnc4, k4, f1s4, mdl4) = ele
            if fnc4 == Distances.canberra_distance:
                ca_cands.append((fnc4, k4, f1s4, mdl4))
            elif fnc4 == Distances.minkowski_distance:
                mi_cands.append((fnc4, k4, f1s4, mdl4))
            elif fnc4 == Distances.euclidean_distance:
                eu_cands.append((fnc4, k4, f1s4, mdl4))
            elif fnc4 == Distances.gaussian_kernel_distance:
                ga_cands.append((fnc4, k4, f1s4, mdl4))
            elif fnc4 == Distances.inner_product_distance:
                in_cands.append((fnc4, k4, f1s4, mdl4))
            elif fnc4 == Distances.cosine_similarity_distance:
                co_cands.append((fnc4, k4, f1s4, mdl4))

        if ca_cands:
            min_k = 34
            for ele in ca_cands:
                (fnc5, k5, f1s5, mdl5) = ele
                if k5 < min_k:
                    min_k = k5
            for ele in ca_cands:
                (fnc51, k51, f1s51, mdl51) = ele
                if k51 == min_k:
                    self.best_k = k51
                    self.best_distance_function = "canberra"
                    self.best_model = mdl51
                    return
        elif mi_cands:
            min_k = 34
            for ele in mi_cands:
                (fnc, k, f1s, mdl) = ele
                if k < min_k:
                    min_k = k
            for ele in mi_cands:
                (fnc, k, f1s, mdl) = ele
                if k == min_k:
                    self.best_k = k
                    self.best_distance_function = "minkowski"
                    self.best_model = mdl
                    return
        elif eu_cands:
            min_k = 34
            for ele in eu_cands:
                (fnc, k, f1s, mdl) = ele
                if k < min_k:
                    min_k = k
            for ele in eu_cands:
                (fnc, k, f1s, mdl) = ele
                if k == min_k:
                    self.best_k = k
                    self.best_distance_function = "euclidean"
                    self.best_model = mdl
                    return
        elif ga_cands:
            min_k = 34
            for ele in ga_cands:
                (fnc, k, f1s, mdl) = ele
                if k < min_k:
                    min_k = k
            for ele in ga_cands:
                (fnc, k, f1s, mdl) = ele
                if k == min_k:
                    self.best_k = k
                    self.best_distance_function = "gaussian"
                    self.best_model = mdl
                    return
        elif in_cands:
            min_k = 34
            for ele in in_cands:
                (fnc, k, f1s, mdl) = ele
                if k < min_k:
                    min_k = k
            for ele in in_cands:
                (fnc, k, f1s, mdl) = ele
                if k == min_k:
                    self.best_k = k
                    self.best_distance_function = "inner_prod"
                    self.best_model = mdl
                    return
        elif co_cands:
            min_k = 34
            for ele in co_cands:
                (fnc, k, f1s, mdl) = ele
                if k < min_k:
                    min_k = k
            for ele in co_cands:
                (fnc, k, f1s, mdl) = ele
                if k == min_k:
                    self.best_k = k
                    self.best_distance_function = "cosine_dist"
                    self.best_model = mdl
                    return

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        # raise NotImplementedError

        list_of_models = []
        for fnc in set(distance_funcs.values()):
            for j in range(1, 30, 2):
                for sclf in set(scaling_classes.values()):
                    scaler = sclf()
                    x_train_scaled = scaler(x_train)
                    model = KNN(j, fnc)
                    model.train(x_train_scaled, y_train)
                    x_val_scaled = scaler(x_val)
                    predicted_labels = model.predict(x_val_scaled)
                    f1s = f1_score(y_val, predicted_labels)
                    list_of_models.append((fnc, j, f1s, model, sclf))
        maxf1 = float('-inf')
        for _, _, f1s1, _, _ in list_of_models:
            if f1s1 > maxf1:
                maxf1 = f1s1
        filtered = []
        for ele in list_of_models:
            (fnc2, k2, f1s2, mdl2, sclf2) = ele
            if f1s2 == maxf1:
                filtered.append((fnc2, k2, f1s2, mdl2, sclf2))

        if len(filtered) == 1:
            (fnc3, k3, f1s3, mdl3, sclf3) = filtered[0]
            self.best_k = k3
            for k, v in distance_funcs.items():
                if v == fnc3:
                    self.best_distance_function = k
                    break
            for k, v in scaling_classes.items():
                if v == sclf3:
                    self.best_scaler = k
                    break
            self.best_model = mdl3
            return

        no_cands = []
        mm_cands = []

        ca_cands = []
        mi_cands = []
        eu_cands = []
        ga_cands = []
        in_cands = []
        co_cands = []

        for ele in filtered:
            (fnc, k, f1s, mdl, sclf) = ele
            if sclf == NormalizationScaler:
                no_cands.append(ele)
            else:
                mm_cands.append(ele)

        if mm_cands:

            for ele in mm_cands:
                (fnc4, k4, f1s4, mdl4, sclf) = ele
                if fnc4 == Distances.canberra_distance:
                    ca_cands.append(ele)
                elif fnc4 == Distances.minkowski_distance:
                    mi_cands.append(ele)
                elif fnc4 == Distances.euclidean_distance:
                    eu_cands.append(ele)
                elif fnc4 == Distances.gaussian_kernel_distance:
                    ga_cands.append(ele)
                elif fnc4 == Distances.inner_product_distance:
                    in_cands.append(ele)
                elif fnc4 == Distances.cosine_similarity_distance:
                    co_cands.append(ele)

            if ca_cands:
                min_k = 34
                for ele in ca_cands:
                    (fnc5, k5, f1s5, mdl5, sclf) = ele
                    if k5 < min_k:
                        min_k = k5
                for ele in ca_cands:
                    (fnc51, k51, f1s51, mdl51, sclf) = ele
                    if k51 == min_k:
                        self.best_k = k51
                        self.best_distance_function = "canberra"
                        self.best_model = mdl51
                        self.best_scaler = "min_max_scale"
                        return
            elif mi_cands:
                min_k = 34
                for ele in mi_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in mi_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "minkowski"
                        self.best_model = mdl
                        self.best_scaler = "min_max_scale"
                        return
            elif eu_cands:
                min_k = 34
                for ele in eu_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in eu_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "euclidean"
                        self.best_model = mdl
                        self.best_scaler = "min_max_scale"
                        return
            elif ga_cands:
                min_k = 34
                for ele in ga_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                    for ele in ga_cands:
                        (fnc, k, f1s, mdl, sclf) = ele
                        if k == min_k:
                            self.best_k = k
                            self.best_distance_function = "gaussian"
                            self.best_model = mdl
                            self.best_scaler = "min_max_scale"
                            return
            elif in_cands:
                min_k = 34
                for ele in in_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in in_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "inner_prod"
                        self.best_model = mdl
                        self.best_scaler = "min_max_scale"
                        return
            elif co_cands:
                min_k = 34
                for ele in co_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in co_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "cosine_dist"
                        self.best_model = mdl
                        self.best_scaler = "min_max_scale"
                        return

        elif no_cands:

            for ele in no_cands:
                (fnc4, k4, f1s4, mdl4, sclf) = ele
                if fnc4 == Distances.canberra_distance:
                    ca_cands.append(ele)
                elif fnc4 == Distances.minkowski_distance:
                    mi_cands.append(ele)
                elif fnc4 == Distances.euclidean_distance:
                    eu_cands.append(ele)
                elif fnc4 == Distances.gaussian_kernel_distance:
                    ga_cands.append(ele)
                elif fnc4 == Distances.inner_product_distance:
                    in_cands.append(ele)
                elif fnc4 == Distances.cosine_similarity_distance:
                    co_cands.append(ele)

            if ca_cands:
                min_k = 34
                for ele in ca_cands:
                    (fnc5, k5, f1s5, mdl5, sclf) = ele
                    if k5 < min_k:
                        min_k = k5
                for ele in ca_cands:
                    (fnc51, k51, f1s51, mdl51, sclf) = ele
                    if k51 == min_k:
                        self.best_k = k51
                        self.best_distance_function = "canberra"
                        self.best_model = mdl51
                        self.best_scaler = "normalize"
                        return
            elif mi_cands:
                min_k = 34
                for ele in mi_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in mi_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "minkowski"
                        self.best_model = mdl
                        self.best_scaler = "normalize"
                        return
            elif eu_cands:
                min_k = 34
                for ele in eu_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in eu_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "euclidean"
                        self.best_model = mdl
                        self.best_scaler = "normalize"
                        return
            elif ga_cands:
                min_k = 34
                for ele in ga_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                    for ele in ga_cands:
                        (fnc, k, f1s, mdl, sclf) = ele
                        if k == min_k:
                            self.best_k = k
                            self.best_distance_function = "gaussian"
                            self.best_model = mdl
                            self.best_scaler = "normalize"
                            return
            elif in_cands:
                min_k = 34
                for ele in in_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in in_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "inner_prod"
                        self.best_model = mdl
                        self.best_scaler = "normalize"
                        return
            elif co_cands:
                min_k = 34
                for ele in co_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k < min_k:
                        min_k = k
                for ele in co_cands:
                    (fnc, k, f1s, mdl, sclf) = ele
                    if k == min_k:
                        self.best_k = k
                        self.best_distance_function = "cosine_dist"
                        self.best_model = mdl
                        self.best_scaler = "normalize"
                        return


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        ftA = np.array(features)
        normalized = []
        for ary in ftA:
            B = ary
            all_zeros = True
            for i in range(len(B)):
                if (B[i] == 0):
                    continue
                else:
                    all_zeros = False
                    break
            C = B.tolist()
            if (all_zeros):
                normalized.append(C)
                continue
            BB = B * B
            D = np.sqrt(sum(BB))
            B = B / D
            C = B.tolist()
            normalized.append(C)
        return normalized


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_run = True
        self.min = []
        self.max = []
        self.num_features = 0

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        if self.first_run:
            self.first_run = False
            self.num_features = len(features[0])
            for i in range(self.num_features):
                m = float('inf')
                M = float('-inf')
                for sbl in features:
                    if m > sbl[i]:
                        m = sbl[i]
                    if M < sbl[i]:
                        M = sbl[i]
                self.min.append(m)
                self.max.append(M)
        assert self.num_features == len(self.min)
        assert self.num_features == len(self.max)
        normalized = []
        for sbl in features:
            newl = []
            for i in range(self.num_features):
                m = self.min[i]
                M = self.max[i]
                c = sbl[i]
                if M - m == 0:
                    val = c - m
                else:
                    val = (c - m) / (M - m)
                newl.append(val)
            normalized.append(newl)
        assert len(normalized) == len(features)
        return normalized

