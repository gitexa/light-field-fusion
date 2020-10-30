# Directory structure needed:
#     -> this file
#     -> data
#         -> data_folder

import torch

data_folder = "Data/Set1/"

#The file is taken from the python-tools of the 4D Light Field Benchmark. It has been adapted to work
#with python3.7


#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################

import configparser
import os
import sys

import numpy as np
import skimage.io
import matplotlib.pyplot as plt


def read_lightfield(data_folder):
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.uint8)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            img = read_img(fpath)
            light_field[idx // params["num_cams_x"], idx % params["num_cams_y"], :, :, :] = img
        except IOError:
            print ("Could not read input file: %s" % fpath)
            sys.exit()

    return light_field


def read_parameters(data_folder):
    params = dict()

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:
        parser = configparser.ConfigParser()
        parser.read_file(f)

        section = "intrinsics"
        params["width"] = int(parser.get(section, 'image_resolution_x_px'))
        params["height"] = int(parser.get(section, 'image_resolution_y_px'))
        params["focal_length_mm"] = float(parser.get(section, 'focal_length_mm'))
        params["sensor_size_mm"] = float(parser.get(section, 'sensor_size_mm'))
        params["fstop"] = float(parser.get(section, 'fstop'))

        section = "extrinsics"
        params["num_cams_x"] = int(parser.get(section, 'num_cams_x'))
        params["num_cams_y"] = int(parser.get(section, 'num_cams_y'))
        params["baseline_mm"] = float(parser.get(section, 'baseline_mm'))
        params["focus_distance_m"] = float(parser.get(section, 'focus_distance_m'))
        params["center_cam_x_m"] = float(parser.get(section, 'center_cam_x_m'))
        params["center_cam_y_m"] = float(parser.get(section, 'center_cam_y_m'))
        params["center_cam_z_m"] = float(parser.get(section, 'center_cam_z_m'))
        params["center_cam_rx_rad"] = float(parser.get(section, 'center_cam_rx_rad'))
        params["center_cam_ry_rad"] = float(parser.get(section, 'center_cam_ry_rad'))
        params["center_cam_rz_rad"] = float(parser.get(section, 'center_cam_rz_rad'))

        section = "meta"
        params["disp_min"] = float(parser.get(section, 'disp_min'))
        params["disp_max"] = float(parser.get(section, 'disp_max'))
        params["frustum_disp_min"] = float(parser.get(section, 'frustum_disp_min'))
        params["frustum_disp_max"] = float(parser.get(section, 'frustum_disp_max'))
        params["depth_map_scale"] = float(parser.get(section, 'depth_map_scale'))

        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params


def read_depth(data_folder, highres=False):
    fpath = os.path.join(data_folder, "gt_depth_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
        print ("Could not read depth file: %s" % fpath)
        sys.exit()
    return data


def read_img(fpath):
    data = skimage.io.imread(fpath)
    return data


def write_pfm(data, fpath, scale=1, file_identifier="Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print (endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write(file_identifier + '\n')
        file.write('%d %d\n' % (width, height))
        file.write('%d\n' % scale)
        file.write(values)


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data


def _get_next_line(f):
    next_line = f.readline().rstrip()
    next_line = next_line.decode('utf-8')
    # ignore comments
    while next_line.startswith('#'):
        next_line = f.readline().rstrip()
    return next_line
    
def read_all_depths(data_folder, highres=False):
    params = read_parameters(data_folder)
    all_depths = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"]), dtype=np.float32)
    views = sorted([f for f in os.listdir(data_folder) if f.startswith("gt_depth_%s_" % ("highres" if highres else "lowres")) and f.endswith(".pfm")])
    
    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            data = read_pfm(fpath)
            all_depths[idx // params["num_cams_x"], idx % params["num_cams_y"], :, :] = data
        except IOError:
            print ("Could not read depth file: %s" % fpath)
            sys.exit()
    return all_depths
