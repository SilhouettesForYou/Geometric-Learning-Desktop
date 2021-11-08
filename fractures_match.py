import const_values
import getopt
import numpy as np
import os
import sys
import time
from utils import Utils
from visualization import *


def usage():
    print('usage of fractures match')
    print('-p, --prefix(demand): the title prefix of fractures.')
    print('-m, --match: load the match pairs from file. e.g. -m ./match-pair/plate-1.txt')
    print('-v, --visualizaiton: visualize the model in vtk.')
    print('-d, --dir: specify the directory of the re-assembly plates.')
    print('--, --decrease: the flag of decrease the number of cluster.')
    print('-k, --cluster(default 12): the number of clusters.')
    print('-c, --comparsion: begin the comparsion among the fractures.')
    print('-t, --test: process test code without core opration.')
    print('-s, --save: save the comparsion result or not.')
    print('-h, --help: print help message.')

def parse_args(argv):
    args = argv[1:]
    try:
        opts, args = getopt.getopt(args, 'p:m:vd:-k:cth', ['prefix=', 'cluster=', 'match=', 'visualization=', 'dir='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    prefix = ''
    pairs = {}
    n_cluster = const_values.FLAGS.default_n_cluster
    dir_of_plate = ''
    flag = False
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif o in ('-v', '--visualization'):
            utils = Utils(prefix, k=n_cluster)
            tv = TridimensionalVisualization()
            # visualize the control points on the fracture
            all_centers = utils.generate_datas(is_decrease=flag)[2]
            for data, centers in zip(utils.datas, all_centers):
                datas = []
                datas.append(data)
                points_data = tv.convert_points_to_data(centers)
                datas.append(points_data)
                for center in centers:
                    datas.append(tv.draw_sphere(center, 1))
                tv.visualize_models_auto(datas)
            # visualize the pre-alignment of plate
            ploy_datas = []
            if dir_of_plate == '':
                pre_plate_file_names = os.listdir(const_values.FLAGS.dir_of_pre_plates)
                for plate_name in pre_plate_file_names:
                    if plate_name.endswith(prefix):
                        for fragment_name in os.listdir(const_values.FLAGS.dir_of_pre_plates + plate_name):
                            # print(const_values.FLAGS.dir_of_pre_plates + plate_name + '/' + fragment_name)
                            ploy_datas.append(utils.read_stl(const_values.FLAGS.dir_of_pre_plates + plate_name + '/' + fragment_name))
            else:
                if not os.path.exists(dir_of_plate):
                    print('please input the vaild dirtory')
                    sys.exit(2)
                elif os.listdir(dir_of_plate) is None:
                    print('there is no (.stl) file in the dirtory')
                    sys.exit(2)
                for name in os.listdir(dir_of_plate):
                    ploy_datas.append(utils.read_stl(dir_of_plate + name))
            if o == '-v':
                tv.visualize_models_auto(ploy_datas)
            elif o == '--visualization':
                if ',' in a and '-' not in a:
                    indices = a.split(',')
                    tv.visualize_models_auto([ploy_datas[int(i) - 1] for i in indices if int(i) - 1 in range(len(ploy_datas))])
                elif '-' in a and ',' not in a:
                    start = int(a.split('-')[0]) - 1
                    end = int(a.split('-')[1])
                    tv.visualize_models_auto(ploy_datas[start:end])
                elif ',' in a and '-' in a:
                    al = a.split(',')
                    indices = []
                    for s in al:
                        if '-' in s:
                            indices.extend([int(i) for i in range(int(s.split('-')[0]), int(s.split('-')[1]) + 1)])
                        else:
                            indices.append(int(s))
                    tv.visualize_models_auto([ploy_datas[int(i) - 1] for i in indices if int(i) - 1 in range(len(ploy_datas))])
                else:
                    print('input format is wrong')
        elif o in ('-d', '--dir'):
            dir_of_plate = a
        elif o in ('-p', '--prefix'):
            prefix = a
        elif o in ('-m', '--match'):
            # load fragments
            utils = Utils(prefix, k=n_cluster)
            utils.load_all_fragment()

            # read match pairs
            with open(a) as file:
                for line in file:
                    l = line.strip('\n').split('&')
                    pairs[l[0]] = l[1]
            # print(pairs)
            for key in pairs:
                print(key + ' matches with ' + pairs[key])
                utils.transform_pair(key, pairs[key])

        elif o in ('-t', '--test'):
            utils = Utils(prefix, k=n_cluster)
            utils.load_all_fragment()
            for key in pairs:
                utils.transform_pair(key, pairs[key])

            # length, names = utils.generate_datas(is_decrease=flag)[-2::]
            # print(length, names)
            # print(np.mean(length))
        elif o in ('--', '--decrease'):
            flag = True
        elif o in ('-k', '--cluster'):
            n_cluster = int(a)
        elif o in ('-c', '--comparsion'):
            utils = Utils(prefix, k=n_cluster)
            utils.comparsion(flag)
        else:
            print('unhandled option')
            sys.exit(3)


def main(argv):
    parse_args(argv)


if __name__ == "__main__":
    print('test time begin at ' + time.asctime(time.localtime(time.time())) + '\n')
    main(sys.argv)
    print('test time end at ' + time.asctime(time.localtime(time.time())) + '\n')