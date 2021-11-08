from collections import namedtuple
from transform import *
from visualization import TridimensionalVisualization


class Fragment(object):

    def __init__(self, fragment, fractures, prefix):
        self.fragment = fragment
        self.fractures = fractures
        self.trans = Transform()
        self.tv = TridimensionalVisualization()
        self.prefix = prefix


    def self_transform(self, matrix):
        self.fragment = self.trans.transform_data(matrix, self.fragment)
        for fracture in self.fractures:
            fracture = self.trans.transform_data(matrix, fracture)


    def visualization_contrast(self, float_fragment):
        datas = []
        fixed_datas, float_datas = [], []
        fixed_datas.append(self.fragment)
        float_datas.append(float_fragment.fragment)
        fixed_datas.extend(self.fractures)
        float_datas.extend(float_fragment.fractures)
        datas.extend(fixed_datas)
        datas.extend(float_datas)
        self.tv.visualize_models_auto(datas)


    def save_fragment(self, path):
        writer = vtk.vtkWriter()
        writer.SetInputData(self.fragment) 
        writer.SetFileName(path + 'fragment-' + self.prefix + '.stl')
        writer.Write()   