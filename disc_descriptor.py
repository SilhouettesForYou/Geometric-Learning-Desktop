import const_values
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import vtk

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DiscDescriptor:

    def __init__(self, source):
        self.source = source
        self.distance_from_mesh = 1
        self.num_of_points_on_disc = 32
        self.num_of_circle = 32
        self.init_radius = 0.1
        self.radius_delta = 0.1
        self.all_normals = []
        self.all_centers_disc = []
        self.all_radii = []
        self.all_points_mesh = []
        self.all_points_disc = []
        self.all_distances = []

    def points_to_array(self, points):
        return np.array([points[x] for x in range(3)])


    def max_min_normalization(self, distances):
        min_max = MinMaxScaler()
        distances = min_max.fit_transform(distances)


    def standard_scaler(self, distances):
        std = StandardScaler()
        distances = std.fit_transform(distances)

    def save_FMIs(self, func, distances, name):
        func(distances)
        # for i in range(distances.shape[0]):
        #     sub_name = name[0:-4] + '-' + str(i) + name[-4:]
        #     scipy.misc.imsave(sub_name, np.roll(distances.T, -i).T)
        scipy.misc.imsave(name, distances)



    def cell_normal_generator(self, cell_flag=True, point_flag=False):
        generator = vtk.vtkPolyDataNormals()
        generator.SetInputData(self.source)
        if cell_flag: generator.ComputeCellNormalsOn()
        if point_flag: generator.ComputePointNormalsOn()
        generator.Update()
        self.source = generator.GetOutput()

    
    def centers_of_cells(self):
        generator = vtk.vtkCellCenters()
        generator.SetInputData(self.source)
        generator.VertexCellsOn()
        generator.Update()

        return generator.GetOutput().GetPoints()


    def center_of_disc(self, normal, p):
        p = np.array(p)
        delta = np.sqrt(sum(normal ** 2)) / self.distance_from_mesh
        return normal / delta + p

    
    def points_on_disc(self, normal, center, radius):
        points = []
        alpha, beta, gamma = normal
        alpha_prime, beta_prime, gamma_prime = np.sqrt(1 - np.square(normal))
        
        for i in range(self.num_of_points_on_disc):
            t = 2 * i * np.pi / self.num_of_points_on_disc;
            coordinate = [0 for _ in range(3)]
            coordinate[0] = center[0] + radius * (beta / gamma_prime) * np.cos(t) + radius * alpha * (gamma / gamma_prime) * np.sin(t)
            coordinate[1] = center[1] - radius * (alpha / gamma_prime) * np.cos(t) + radius * beta * (gamma / gamma_prime) * np.sin(t)
            coordinate[2] = center[2] - radius * gamma_prime * np.sin(t)
            points.append(np.array(coordinate))
        
        return np.array(points)

    
    def compute_descriptor(self, obb, normal, center_disc, radius):
        points = []
        distances = []
        points_disc = self.points_on_disc(normal, center_disc, radius)
        for p in points_disc:
            point_other_side = -normal * const_values.FLAGS.T + p
            point_another_size = normal * const_values.FLAGS.T + p
            intersected_points, intersected_cells = vtk.vtkPoints(), vtk.vtkIdList()
            obb.SetTolerance(const_values.FLAGS.tolerance_of_obb)
            obb.IntersectWithLine(point_other_side, point_another_size, intersected_points, intersected_cells)

            intersect = []
            if intersected_points.GetNumberOfPoints() > 1:
                min_distance = np.inf
                for i in range(intersected_points.GetNumberOfPoints()):
                    if np.linalg.norm(p - self.points_to_array(intersected_points.GetPoint(i))) < min_distance:
                        min_distance = np.linalg.norm(p - self.points_to_array(intersected_points.GetPoint(i)))
                        intersect = self.points_to_array(intersected_points.GetPoint(i))
                vec = p - intersect
                vec = vec / np.linalg.norm(vec)
                if vec.dot(normal) < 0:
                    distances.append(-np.linalg.norm(p - intersect))
                else:
                    distances.append(np.linalg.norm(p - intersect))
            elif intersected_points.GetNumberOfPoints() == 0:
                intersect = p
                distances.append(0.0)
            else:
                intersect = [intersected_points.GetPoint(0)[x] for x in range(3)]
                distances.append(np.linalg.norm(p - intersect)) 
            intersect = np.array(intersect)

            
            points.append(intersect)
        points_mesh = np.array(points)
        distances = np.array(distances)
        return points_mesh, points_disc, distances


    def mesh_descriptors(self, FMIs_dir, type_of_normalize, random_num=0):
        self.cell_normal_generator(cell_flag=True, point_flag=False)
        normals = self.source.GetCellData().GetNormals()
        centers = self.centers_of_cells()

        obb = vtk.vtkOBBTree()
        # locator = vtk.vtkCellLocator()
        obb.SetDataSet(self.source)
        obb.BuildLocator()

        size = self.source.GetNumberOfCells()
        scope = []
        if random_num is not 0:
            scope = np.random.randint(0, size, random_num)
        else:
            scope = range(size)
        
        print(scope)
        for i in scope:
            normal = np.array([normals.GetTuple(i)[x] for x in range(3)])
            center = np.array([centers.GetPoint(i)[x] for x in range(3)])
            center_disc = self.center_of_disc(normal, center)
            radii = []
            distances = []
            for j in range(self.num_of_circle):
                radius = self.init_radius + j * self.radius_delta
                radii.append(radius)
                points_mesh, points_disc, d = self.compute_descriptor(obb, normal, center_disc, radius)
                # print('distance: ', [np.linalg.norm(x - y) for x, y in zip(points_mesh, points_disc)])
                distances.append(d)
                self.all_points_mesh.append(points_mesh)
                self.all_points_disc.append(points_disc)
            distances = np.array(distances)
            name = FMIs_dir + str(i) + '.bmp'
            if type_of_normalize is 0:
                self.save_FMIs(self.max_min_normalization, distances, name)
            else:
                self.save_FMIs(self.standard_scaler, distances, name)
            # self.hist_distances(distances.reshape((distances.shape[0] * distances.shape[1], 1)))
            self.all_distances.append(distances)
            self.all_normals.append(normal)
            self.all_centers_disc.append(center_disc)
            self.all_radii.append(np.array(radii))


    def draw_points(self, points):
        points_data = vtk.vtkPolyData()
        vertex_filter = vtk.vtkVertexGlyphFilter()
        points_data.SetPoints(points)
        vertex_filter.SetInputData(points_data)
        vertex_filter.Update()

        return vertex_filter.GetOutput()
        

    def draw_circles(self):
        circles = []
        for center, normal in zip(self.all_centers_disc, self.all_normals):
            for radii in self.all_radii:
                for radius in radii:
                    circle = vtk.vtkRegularPolygonSource()
                    circle.SetNormal(normal)
                    circle.SetNumberOfSides(500)
                    circle.SetRadius(radius)
                    circle.SetCenter(center)
                    circle.Update()
                    circles.append(circle.GetOutput())
        return circles


    def draw_lines(self):
        points_datas = []
        line_datas = []
        points_on_mesh, points_on_disc = vtk.vtkPoints(), vtk.vtkPoints()
        # 
        for points1, points2 in zip(self.all_points_mesh, self.all_points_disc):
            points = vtk.vtkPoints()
            for p1, p2 in zip(points1, points2):
                points.InsertNextPoint(p1)
                points.InsertNextPoint(p2)
                points_on_mesh.InsertNextPoint(p1)
                points_on_disc.InsertNextPoint(p2)

            line_data = vtk.vtkPolyData()
            line_data.SetPoints(points)
            lines = vtk.vtkCellArray()
            for i in range(0, points.GetNumberOfPoints(), 2):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)
            line_data.SetLines(lines)
            line_datas.append(line_data)

            points_datas.append(self.draw_points(points_on_mesh))
            points_datas.append(self.draw_points(points_on_disc))
            

        

        return line_datas, points_datas

    def hist_distances(self, distances, label='distances'):
        plt.style.use( 'ggplot')
        plt.hist(distances, bins = len(distances), color = 'steelblue', label = label)

        plt.tick_params(top = 'off', right = 'off')
        plt.legend()
        plt.show()

    def visualize_models(self, datas):
        ren= vtk.vtkRenderer()  
        color_index = np.arange(const_values.const.LEN_OF_COLOR)
        np.random.shuffle(color_index)
        for data, i in zip(datas, range(len(datas))):
            mapper = vtk.vtkPolyDataMapper()  
            mapper.SetInputData(data)  
            
            actor = vtk.vtkActor()  
            actor.SetMapper(mapper) 
            actor.GetProperty().SetPointSize(10) 
            actor.GetProperty().SetColor(np.array(const_values.const.COLOR[color_index[i % const_values.const.LEN_OF_COLOR]]) / 255.0)
            ren.AddActor( actor )  
            # ren.SetBackground( 0 / 255.0, 166 / 255.0, 222 / 255.0 ) 
            ren.SetBackground( 255 / 255.0, 255 / 255.0, 255 / 255.0 ) 
        renWin = vtk.vtkRenderWindow()  
        renWin.AddRenderer( ren )  
        renWin.SetSize( 300, 300 )  
        renWin.Render()  
        
        iren=vtk.vtkRenderWindowInteractor()  
        iren.SetRenderWindow(renWin)  
        
        iren.Initialize()  
        iren.Start()
