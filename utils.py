# -- coding: utf-8 --

import os
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from transform import *
from visualization import *
from fragment import Fragment

class Utils:

    def __init__(self, prefix, k=12):
        self.prefix = prefix
        self.k = k
        self.datas = []
        self.fragments = []
        self.tv = TridimensionalVisualization()
        self.transform = Transform()


    # @property
    # def datas(self):
    #     return self.datas

    
    def read_stl(self, file_name):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()

        poly_data = reader.GetOutput()
        return poly_data


    def load_all_fragment(self):
        file_names_of_plates = os.listdir(const_values.FLAGS.dir_of_plates)
        file_names_of_fractures = os.listdir(const_values.FLAGS.dir_of_fractures)
        names_of_fragment = []
        
        for name_of_plate in file_names_of_plates:
            if name_of_plate.endswith(self.prefix):
                file_names_of_fragments = os.listdir(const_values.FLAGS.dir_of_plates + name_of_plate)
                for name_of_fragment in file_names_of_fragments:
                    if name_of_fragment.endswith('.stl'):
                        names_of_fragment.append(name_of_plate + '/' + name_of_fragment)
        # print(names_of_fragment)
        names_of_fracture = [[] for _ in range(len(names_of_fragment))]
        for name_of_fracture in file_names_of_fractures:
            s = name_of_fracture.split('-')
            if s[0] == self.prefix:
                names_of_fracture[int(s[1]) - 1].append(name_of_fracture)
        # print(names_of_fracture)
        for i in range(len(names_of_fragment)):
            # print('fragment: ', const_values.FLAGS.dir_of_plates + names_of_fragment[i])
            # print('fracture: ', const_values.FLAGS.dir_of_fractures + names_of_fracture[i][0])
            fragment = self.read_stl(const_values.FLAGS.dir_of_plates + names_of_fragment[i])
            fractures = [self.read_stl(const_values.FLAGS.dir_of_fractures + j) for j in names_of_fracture[i]]
            self.fragments.append(Fragment(fragment, fractures, self.prefix))

    def get_control_points(self, poly_data):
        n = poly_data.GetNumberOfPoints()
        X = np.zeros((n, 3), dtype=np.float64)
        
        points = poly_data.GetPoints()
        for i in range(n):
            X[i] = points.GetPoint(i)

        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        # serialize the center points
        centers = self.serialize_centers(centers)
        return X, centers

    def transform_pair(self, name_of_fixed, name_of_float):
        # parse name
        fixed_fragment_id, float_fragment_id = int(name_of_fixed.split('-')[1]) - 1, int(name_of_float.split('-')[1]) - 1
        fixed_fracture_id, float_fracture_id = ord(name_of_fixed.split('-')[2]) - ord('a'), ord(name_of_float.split('-')[2]) - ord('a')
        fixed_fragment, float_fragmet = self.fragments[fixed_fragment_id], self.fragments[float_fragment_id]
        fixed_fracture, float_fracture = fixed_fragment.fractures[fixed_fracture_id], float_fragmet.fractures[float_fracture_id]

        fixed_points, _, = self.get_control_points(fixed_fracture)
        float_points, _ = self.get_control_points(float_fracture)
        float_points_, bias, identification, rotate_matrices, translate_matrices = self.transform.collimate_axis_general(fixed_points, float_points, '')

        fixed_fragment.visualization_contrast(float_fragmet)

        if identification is const_values.const.AXIS_TYPE['main'] or identification is const_values.const.AXIS_TYPE['secondary']:
            float_fragmet.self_transform(rotate_matrices)
            float_fragmet.self_transform(translate_matrices)
        elif identification is const_values.const.AXIS_TYPE['both']:
            for rotate_matrix, translate_matrix in zip(rotate_matrices, translate_matrices):
                float_fragmet.self_transform(rotate_matrix)
                float_fragmet.self_transform(translate_matrix)
        
        # test
        fixed_fragment.visualization_contrast(float_fragmet)


    def serialize_centers(self, centers):
        k = len(centers)
        kdtree = KDTree(centers, leaf_size=30, metric='euclidean')
        distances, mapping = kdtree.query(centers, k=k, return_distance=True)
        
        start = np.where(distances[:, k - 1] == max(distances[:, k - 1]))
        start = start[0][0]
        end = mapping[start][k - 1]

        serialized_indices = [start]
        while start != end:
            i = 1
            while mapping[start][i] in serialized_indices:
                i += 1
            start = mapping[start][i]
            serialized_indices.append(start)

        return np.array([centers[x] for x in serialized_indices])


    def length_of_fracture(self, centers):
        return np.linalg.norm(centers[0] - centers[-1])


    def random_sample(self, X, l=100):
        res = []
        indices = np.random.randint(len(X), size=l)
        for index in indices:
            res.append(X[index])
        return np.array(res)


    def make_pair(self, fixed_points, float_points, indices):
        assert len(float_points) == len (indices) and len(fixed_points) == len(indices)
        X = []
        Y = []
        for i in range(len(indices)):
            X.append(fixed_points[i])
            Y.append(float_points[indices[i]])
        return np.array(X), np.array(Y)


    def value_of_k(self, nums):
        pass


    def generate_datas(self, is_decrease=False):
        all_points = []
        all_random_points = []
        all_centers = []
        all_length = []

        file_names = os.listdir(const_values.FLAGS.dir_of_fractures)

        for file_name in file_names:
            if file_name.startswith(self.prefix):
                poly_data = self.read_stl(const_values.FLAGS.dir_of_fractures + file_name)
                self.datas.append(poly_data)
                n = poly_data.GetNumberOfPoints()
                X = np.zeros((n, 3), dtype=np.float64)
                # visualization(data)
                points = poly_data.GetPoints()
                for i in range(n):
                    X[i] = points.GetPoint(i)

                all_points.append(X)

                # random sample
                l = min([len(x) for x in all_points])
                for points in all_points:
                    all_random_points.append(self.random_sample(points, l))
        len_of_points_set = np.array([len(x) for x in all_points])
        avg_num = np.mean(np.array([len(x) for x in all_points]))
        for X in all_points:
            k = self.k
            if len(X) < avg_num and is_decrease:
                k = self.k // 2 + 1
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_

            # serialize the center points
            centers = self.serialize_centers(centers)

            # compute the length of each fracture
            all_length.append(self.length_of_fracture(centers))
            all_centers.append(centers)

        return np.array(all_points), np.array(all_random_points), np.array(all_centers), np.array(all_length),\
            [name for name in file_names if name.startswith(self.prefix)]


    def comparsion(self, is_decrease, is_save=False):
        all_points, all_random_points, all_centers, all_length, file_names = self.generate_datas(is_decrease)
        centers = np.array([np.mean(center, 0) for center in all_centers])

        icp = ICP()
        transform = Transform()

        points = all_centers
        for i in range(len(points)):
            min_bais = []
            alternative_pair = {}
            for j in range(len(points)):
                if i != j and abs(i - j) != 1:
                    fixed_points, float_points = points[i], points[j]
                    # make dir
                    path = const_values.FLAGS.dir_of_axis_transformation_pics + file_names[i][:-4] + '&' + file_names[j][:-4] + '/'
                    if not os.path.exists(path):
                        print(path)
                        os.makedirs(path)

                    float_points_, bias, identification, _, _ = transform.collimate_axis_general(fixed_points, float_points, is_save=is_save)
                    min_bais.append(bias)
                    if is_save:
                        transform.save_fig(const_values.FLAGS.dir_of_fracture_comparsion_pics, file_names[i] + ' compares to ' + file_names[j] + '.png', fixed_points, float_points_)

                else:
                    alternative_pair[file_names[j]] = np.inf
                    min_bais.append(np.inf)
            alternative_pair = sorted(alternative_pair.items(), key = lambda x:x[1])
            for pair in alternative_pair:
                if pair[1] is not np.inf:
                    print(file_names[i], ' compare to ', pair[0], ' with bias ', pair[1])
            print('---', file_names[i], ' - ', file_names[min_bais.index(min(min_bais))], ': ', min(min_bais), '\n')

