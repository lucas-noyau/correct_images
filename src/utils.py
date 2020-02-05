import json
import numpy as np
from pathlib import Path
import yaml
from cv2 import imread


def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mapping


def save_new_annotations_file(output_filepath, data):
    with open(output_filepath + "annotations.json", "w") as outfile:
        json.dump(data, outfile)


def get_image_and_info(image_path):
    image = imread(image_path)
    shape = image.shape
    image_width = shape[1]
    image_height = shape[0]
    return(image, image_width, image_height)


def get_altitude(image_name, data_filename):
    with open(data_filename, "r") as data_file:
        matched_lines = [line for line in data_file.readlines() if image_name in line]
    if matched_lines == []:
        print("ERROR: image name not found in data file")
        print(image_name)
        exit()
    #altitude information should be in column 11 of the csv file
    altitude = matched_lines[0].split(",")[11].strip()
    return altitude


def initialise_data_dictionary(coco_labels_filepath):
    data = None
    with open(coco_labels_filepath) as json_file:
        data = json.load(json_file)
        data["annotations"] = []
    return data


def cv2np(node):
    rows = node['rows']
    cols = node['cols']
    a = np.asarray(node['data']).reshape((int(rows), int(cols)))
    return a


class MonoCamera:
    def __init__(self, filename=None):
        self.K = np.zeros((3, 3))
        self.d = np.zeros((5, 1))
        self.R = np.eye(3)
        self.P = np.zeros((3, 4))
        self.image_width = 0
        self.image_height = 0
        self.name = ''

        if filename is not None:
            filename = Path(filename)
            with filename.open('r') as stream:
                yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
                d = yaml.load(stream, Loader=yaml.Loader)
            self.from_node(d)

    def from_node(self, node):
        self.name = node['camera_name']
        self.image_width = node['image_width']
        self.image_height = node['image_height']
        self.K = cv2np(node['camera_matrix'])
        self.d = cv2np(node['distortion_coefficients'])
#      self.R = cv2np(node['rectification_matrix'])
#      self.P = cv2np(node['projection_matrix'])

    def to_str(self):
        msg = (""
             + "image_width: " + str(self.image_width) + "\n"
             + "image_height: " + str(self.image_height) + "\n"
             + "camera_name: " + self.name + "\n"
             + "camera_matrix:\n"
             + "  rows: 3\n"
             + "  cols: 3\n"
             + "  data: [" + ", ".join(["%8f" % i for i in self.K.reshape(1, 9)[0]]) + "]\n"
             + "distortion_model: " + ("rational_polynomial" if self.d.size > 5 else "plumb_bob") + "\n"
             + "distortion_coefficients:\n"
             + "  rows: 1\n"
             + "  cols: 5\n"
             + "  data: [" + ", ".join(["%8f" % self.d[i, 0] for i in range(self.d.shape[0])]) + "]\n"
             + "rectification_matrix:\n"
             + "  rows: 3\n"
             + "  cols: 3\n"
             + "  data: [" + ", ".join(["%8f" % i for i in self.R.reshape(1, 9)[0]]) + "]\n"
             + "projection_matrix:\n"
             + "  rows: 3\n"
             + "  cols: 4\n"
             + "  data: [" + ", ".join(["%8f" % i for i in self.P.reshape(1, 12)[0]]) + "]\n"
             + "")
        return msg
