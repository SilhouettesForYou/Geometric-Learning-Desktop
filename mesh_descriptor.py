import const_values
import getopt
import numpy as np
import os
import re
import sys
import time
import vtk

from disc_descriptor import DiscDescriptor


def read_stl(file_name):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()

        poly_data = reader.GetOutput()
        return poly_data

def usage():
    print('usage of fractures match')
    print('-d, --dir: specify the directory of the re-assembly plates. e.g. -d ./plates/plate-1/')
    print('-r, --random: draw descriptor randomly, require the random number. e.g. -r 10')
    print('-c, --config: config the options of descriptor.')
    print('             ', '[distance from disc(1)]') 
    print('             ', '[num of points on disc(32)]') 
    print('             ', '[num of circles(32)]') 
    print('             ', '[init radius(0.1)]') 
    print('             ', '[radius delta(0.1)]')
    print('-s, --save: save features to image. [min-max, z-score]. e.g. -s min-max')
    print('-g, --generate: generate data only, do not visualize.')
    print('-v, --visualization: options([p:points, l:lines, c:circles, s:source]). e.g. -v p,l,c,s')
    print('-h, --help: print help message.')


def valid_input(message, value):
    val = input(message)
    if len(val) is 0:
        return
    pattern = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
    while pattern.match(val):
        print('invalid input')
        val = input(message)
    else:
        value = float(val)


def parse_args(argv):
    args = argv[1:]
    try:
        opts, args = getopt.getopt(args, 'd:r:cs:v:gh', ['dir=', 'random=', 'config', 'save=', 'visualization=', 'generate', 'help'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    
    random_num = 0
    distance_from_mesh = 1
    num_of_points_on_disc = 32
    num_of_circle = 32
    init_radius = 0.1
    radius_delta = 0.1
    file_names = []
    FMIs_dirs = []
    visual_flag = False
    generate_flag = False
    type_of_normalize = 0
    visual_opts = []
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif o in ('-d', '--dir'):
            for file_name in os.listdir(a):
                if file_name.endswith('.stl'):
                    file_names.append(a + file_name)
                    print(a + file_name)
                    
        elif o in ('-r', '--random'):
            random_num = int(a)
        elif o in ('-c', '--config'):
            valid_input('distance from disc(1): ', distance_from_mesh)
            valid_input('num of points on disc(32): ', num_of_points_on_disc)
            valid_input('num of circles(32): ', num_of_circle)
            valid_input('init radius(0.1): ', init_radius)
            valid_input('radius delta(0.1): ', radius_delta)
        elif o in('-s', '--save'):
            if len(file_names) is not 0:
                if a == 'max-min':
                    type_of_normalize = 0
                elif a == 'z-score':
                    type_of_normalize = 1
                for name in file_names:
                    path = name[0:-4] + '-FMIs/'
                    FMIs_dirs.append(path)
                    if not os.path.exists(path):
                        print(path)
                        os.makedirs(path)
        elif o in ('-g', '--generate'):
            generate_flag = True
        elif o in ('-v', '--visualization'):
            visual_opts = a.split(',')
            visual_flag = True
        else:
            print('unhandled option')
            sys.exit(3)
    
    for file_name, FMIs_dir in zip(file_names, FMIs_dirs):
        poly_data = read_stl(file_name)
        descriptor = DiscDescriptor(poly_data)
        descriptor.mesh_descriptors(FMIs_dir, type_of_normalize, random_num=random_num)

        # config desriptor
        descriptor.distance_from_mesh = distance_from_mesh
        descriptor.num_of_points_on_disc = num_of_points_on_disc
        descriptor.num_of_circle = num_of_circle
        descriptor.init_radius = init_radius
        descriptor.radius_delta = radius_delta

        # visualization
        if not generate_flag:
            visualized_datas = []
            circles = descriptor.draw_circles()
            lines_datas, points_datas = descriptor.draw_lines()
            if not visual_flag:
                visualized_datas.append(poly_data)
                visualized_datas.extend(circles)
                visualized_datas.extend(points_datas)
                visualized_datas.extend(lines_datas)
            else:
                if 's' in visual_opts:
                    visualized_datas.append(poly_data)
                if 'c' in visual_opts:
                    visualized_datas.extend(circles)
                if 'p' in visual_opts:
                    visualized_datas.extend(points_datas)
                if 'l' in visual_opts:
                    visualized_datas.extend(lines_datas)       
            descriptor.visualize_models(visualized_datas)


def main(argv):
    parse_args(argv)


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')