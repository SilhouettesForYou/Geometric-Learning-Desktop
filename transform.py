import const_values
import math
import os
import vtk

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from visualization import FlatVisualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Transform:

    def get_matrix(self, T):
        '''
        convert the transform matrix to the form of vtkMatrix4x4

        Args:
            T: the transform matrix
        Return:
            matrix: the form of vtkMatrix4x4
        '''
        matrix = vtk.vtkMatrix4x4()

        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, T[i][j])

        return matrix

    
    def transform_data(self, matrix, data):
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(data)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        return transformFilter.GetOutput()

    
    def transform_points(self, matrix, points):
        '''
        transform the two point sets to overlap approximately

        Args:
            matrix: transformation matrix
            points_fiexed: the fixed points data
            point_float: the points data to be transformed
        Return:
            the points after transformed 
        '''
        res = []
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        for point in points:
            point_float = [0, 0, 0]
            transform.TransformPoint(point, point_float)
            res.append(point_float)
        return np.array(res)

    def transform_point(self, matrix, point):
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformed_point = [0, 0, 0]
        transform.TransformPoint(point, transformed_point)
        return transformed_point

    def angle_of_normal(self, a, b):
        return math.acos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    def rotate_by_any_axis(self, axis, theta):
        '''
        rotate the float normal to be parallel to the fixed normal

        Args:
            axis: the axis to te rotated
            theta: the angle of rotation

        Return:
            matrix: transformation matrix
        '''

        matrix = vtk.vtkMatrix4x4()

        a = axis[0]
        b = axis[1]
        c = axis[2]

        
        matrix.SetElement(0, 0, a ** 2+ (1 - a ** 2) * math.cos(theta))
        matrix.SetElement(0, 1, a * b * (1 - math.cos(theta)) + c * math.sin(theta))
        matrix.SetElement(0, 2, a * c * (1 - math.cos(theta)) - b * math.sin(theta))
        matrix.SetElement(0, 3, 0)
        
        matrix.SetElement(1, 0, a * b * (1 - math.cos(theta)) - c * math.sin(theta))
        matrix.SetElement(1, 1, b ** 2 + (1 - b ** 2) * math.cos(theta))
        matrix.SetElement(1, 2, b * c * (1 - math.cos(theta)) + a * math.sin(theta))
        matrix.SetElement(1, 3, 0)
        
        matrix.SetElement(2, 0, a * c * (1 - math.cos(theta)) + b * math.sin(theta))
        matrix.SetElement(2, 1, b * c * (1 - math.cos(theta)) - a * math.sin(theta))
        matrix.SetElement(2, 2, c ** 2 + (1 - c ** 2) * math.cos(theta))
        matrix.SetElement(2, 3, 0)
        
        matrix.SetElement(3, 0, 0)
        matrix.SetElement(3, 1, 0)
        matrix.SetElement(3, 2, 0)
        matrix.SetElement(3, 3, 1)

        return matrix

    def translate(self, start, end):
        '''
        translate the start point to overlap with end point

        Args:
            start: start point (float)
            end: end point (fixed)

        Return:
            matrix: transformation matrix
        '''

        matrix = vtk.vtkMatrix4x4()

        for i in range(4):
            for j in range(4):
                if i is j:
                    matrix.SetElement(i, j, 1)
                elif j is 3:
                    matrix.SetElement(i, j, start[i] - end[i])
                else:
                    matrix.SetElement(i, j, 0)
        
        return matrix

    def compute_axis(self, points):
        pca = PCA(n_components=3)
        pca.fit(points)

        return pca.components_[0], pca.components_[1]


    def implicit_transformation(self, float_points, fixed_axis, float_axis, fixed_center):
        theta = self.angle_of_normal(fixed_axis, float_axis)
        axis = np.cross(fixed_axis, float_axis)
        rotate_matrix = self.rotate_by_any_axis(axis, theta)

        points = self.transform_points(rotate_matrix, float_points)
        center = np.mean(points, 0)

        translate_matrix = self.translate(fixed_center, center)
        points = self.transform_points(translate_matrix, points)
        return points, rotate_matrix, translate_matrix

    def save_fig(self, path, title, fixed_points, float_points_):
        print(path + title + '.png')
        plt.style.use('ggplot')
        fig = plt.figure()
        ax = Axes3D(fig)
        fv = FlatVisualization(ax)
        fv.paint_two_points(fixed_points, float_points_, title= title)
        fig.savefig(path + title + '.png', dpi=200)
        plt.close(fig)


    def collimate_axis(self, fixed_points, float_points, path, is_save=False):
        '''
        collimate the fixed main axis and fixed secondary axis with the float ones
        there are three times transformation totaly
        Args:
            fixed_points: the baseline points ont the fracture
            float_points: the transforming points ont the fracture

        Return:
            points: the transformed points of float fracture
            matrices: kinds of transformation matrices
        '''
        # record the transformation matrix of each step
        index = 0
        indices = np.zeros(len(fixed_points))
        indices_main = np.zeros(len(fixed_points))
        indices_secondary = np.zeros(len(fixed_points))
        
        rotate_matrices = []
        translate_matrices = []
        proper_fisrt_index, proper_second_index = 0, 0

        float_points_ = float_points
        points_main, points_secondary = float_points, float_points
        main_rotate_matrix, main_translate_matrix = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
        secondary_rotate_matrix, secondary_translate_matrix = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
        bias = np.inf
        all_bias = []
        icp = ICP()
        # compute the main axis of fixed points
        fixed_components = self.compute_axis(fixed_points)
        float_components = self.compute_axis(float_points)
        fixed_main_axis = np.array([fixed_components[0], -fixed_components[0]])
        float_main_axis = np.array([float_components[0], -float_components[0]])
        fixed_secondary_axis = np.array([fixed_components[1], -fixed_components[1]])
        fixed_center = np.mean(fixed_points, 0)

        tip = ['main', 'reversed main', 'secondary', 'reversed secondary']
        # rotation by axes include main axis and secondary
        for i in range(2):
            for j in range(2):
                title1 = '① FXMA-' + str(chr(97 + i)) + ' algin with FLMA-' + str(chr(97 + j)) + ' '
                if i is 0 and j is 0 or i is 1 and j is 0:
                    points, rotate_matrix, translate_matrix = \
                        self.implicit_transformation(float_points, fixed_main_axis[i], float_main_axis[j], fixed_center)
                    rotate_matrices.append(rotate_matrix)
                    translate_matrices.append(translate_matrix)
                    # compute the axis after main axis transformation
                    float_components = self.compute_axis(points)
                    float_secondary_axis = np.array([float_components[1], -float_components[1]])
                    for k in range(2):
                        for l in range(2):
                            title2 = '② FXSA-' + str(chr(97 + k)) + ' algin with FLSA-' + str(chr(97 + l))
                            title = title1 + title2
                            if k is 0 and l is 0 or k is 1 and l is 0:
                                points, rotate_matrix, translate_matrix = \
                                    self.implicit_transformation(points, fixed_secondary_axis[k], float_secondary_axis[l], fixed_center)
                                rotate_matrices.append(rotate_matrix)
                                translate_matrices.append(translate_matrix)
                                distances, indices_ = icp.nearest_neighbor(fixed_points, points)
                                all_bias.append(np.mean(distances))
                                if np.mean(distances) < bias:
                                    proper_fisrt_index, proper_second_index = i + j, 2 + k + l
                                    bias = np.mean(distances)
                                    float_points_ = points
                                #save the transformation results to directory
                                if is_save:
                                    self.save_fig(path, title, fixed_points, points)

        # rotation by only the one axis, main axis or secondary axis
        bias_main = np.inf
        for i in range(len(fixed_main_axis)):
            for j in range(len(float_main_axis)):
                points, rotate_matrix, translate_matrix = \
                    self.implicit_transformation(points, fixed_secondary_axis[k], float_secondary_axis[l], fixed_center)
                distances, indices_ = icp.nearest_neighbor(fixed_points, points)
                if np.mean(distances) < bias_main:
                    main_rotate_matrix, main_translate_matrix = rotate_matrix, translate_matrix
                    bias_main = np.mean(distances)
                    points_main = points

                title = 'FXMA-' + str(chr(97 + i)) + ' algin with FLMA-' + str(chr(97 + j))
                if is_save:
                    self.save_fig(path, title, fixed_points, points)
                
        bias_secondary = np.inf
        for i in range(len(fixed_secondary_axis)):
            for j in range(len(float_secondary_axis)):
                points, rotate_matrix, translate_matrix = \
                    self.implicit_transformation(points, fixed_secondary_axis[k], float_secondary_axis[l], fixed_center)
                distances, indices_ = icp.nearest_neighbor(fixed_points, points)
                if np.mean(distances) < bias_main:
                    secondary_rotate_matrix, secondary_translate_matrix = rotate_matrix, translate_matrix
                    bias_main = np.mean(distances)
                    points_secondary = points
                    
                title = 'FXMA-' + str(chr(97 + i)) + ' algin with FLMA-' + str(chr(97 + j))
                if is_save:
                    self.save_fig(path, title, fixed_points, points)
                    
        if bias < bias_main and bias < bias_secondary:
            return float_points_, bias, const_values.const.AXIS_TYPE['both'], \
                [rotate_matrices[proper_fisrt_index], rotate_matrices[proper_second_index]], \
                [translate_matrices[proper_fisrt_index], translate_matrices[proper_second_index]]
        elif bias_main < bias and bias_main < bias_secondary:
            return points_main, bias_main, const_values.const.AXIS_TYPE['main'], \
                main_rotate_matrix, main_translate_matrix
        else:
            return points_secondary, bias_secondary, const_values.const.AXIS_TYPE['secondary'], \
                secondary_rotate_matrix, secondary_translate_matrix


    def collimate_axis_general(self, fixed_points, float_points, path, is_save=False):
        # select the same points from fixed points and float points
        m, n = len(fixed_points), len(float_points)
        float_points_ = float_points
        bias = 0
        identification = 0
        rotate_matrices, translate_matrices = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
        if m < n:
            points_1, bias_1, identification_1, rotate_matrices_1, translate_matrices_1 = self.collimate_axis(fixed_points, float_points[:m], path + '1-', is_save=is_save)
            points_2, bias_2, identification_2, rotate_matrices_2, translate_matrices_2 = self.collimate_axis(fixed_points, float_points[n - m:], path + '2-', is_save=is_save)
            if bias_1 < bias_2: float_points_, bias, identification, rotate_matrices, translate_matrices = points_1, bias_1, identification_1, rotate_matrices_1, translate_matrices_1
            else: float_points_, bias, identification, rotate_matrices, translate_matrices = \
                points_2, bias_2, identification_2, rotate_matrices_2, translate_matrices_2
        elif m > n:
            points_1, bias_1, identification_1, rotate_matrices_1, translate_matrices_1 = self.collimate_axis(fixed_points[:n], float_points, path + '3-', is_save=is_save)
            points_2, bias_2, identification_2, rotate_matrices_2, translate_matrices_2 = self.collimate_axis(fixed_points[m - n:], float_points, path + '4-', is_save=is_save)
            if bias_1 < bias_2: float_points_, bias, identification, rotate_matrices, translate_matrices = points_1, bias_1, identification_1, rotate_matrices_1, translate_matrices_1
            else: float_points_, bias, identification, rotate_matrices, translate_matrices = \
                points_2, bias_2, identification_2, rotate_matrices_2, translate_matrices_2
        else:
            return self.collimate_axis(fixed_points, float_points, path, is_save=is_save)
        return float_points_, bias, identification, rotate_matrices, translate_matrices

    def turn_over_by_axis(self, axis):
        return self.rotate_by_any_axis(axis, np.pi / 2)




class ICP:
    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.001):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1,A.shape[0]))
        dst = np.ones((m + 1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _ , _ = self.best_fit_transform(A, src[:m,:].T)

        return T, distances, i