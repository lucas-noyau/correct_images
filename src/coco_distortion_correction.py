import cv2
import os
import numpy as np
from pathlib import Path
import yaml
from pycocotools.coco import COCO
from utils import *
import math



def correct_distortion(src_img, map_x, map_y, dst_bit=8):
    src_img = np.clip(src_img, 0, 2 ** dst_bit - 1)
    dst_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
    return dst_img


def calc_distortion_mapping(camera_parameter_file_path, width, height):
    mono_cam = MonoCamera(camera_parameter_file_path)
    cam_mat, _ = cv2.getOptimalNewCameraMatrix(mono_cam.K, mono_cam.d,
                                               (width, height), 0)
    map_x, map_y = cv2.initUndistortRectifyMap(mono_cam.K,
                                               mono_cam.d,
                                               None, cam_mat,
                                               (width, height), 5)
    return map_x, map_y

def get_camera_matrix(camera_parameter_file_path, width, height):
    mono_cam = MonoCamera(camera_parameter_file_path)
    cam_mat, _ = cv2.getOptimalNewCameraMatrix(mono_cam.K, mono_cam.d,
                                               (width, height), 0)
    return cam_mat





def undistort_point(point, distortion_mapping_x, distortion_mapping_y):
    point = point.reshape(2)
    x_distance = np.abs(distortion_mapping_x-point[0])
    y_distance = np.abs(distortion_mapping_y-point[1])
    total_distance = x_distance + y_distance
    return np.flip(np.unravel_index(total_distance.argmin(), total_distance.shape))


def undistort_points(points, distortion_mapping_x, distortion_mapping_y):
    new_points=[]
    for point in points:
        print('old point: '+str(point))
        new_point = undistort_point(point, distortion_mapping_x, distortion_mapping_y)
        print('new point: '+str(new_point))
        if new_point is not None:
            new_points.append(new_point)
        else:
            print("point outside of edge of image, using original point")
            print(point)
            exit()
            new_points.append(point.reshape(2).tolist())
    print(new_points)
    return new_points 

def coco_labels_distort_correction(
    coco_labels_filepath,
    output_coco_labels_filepath,
    camera_parameter_file_path,
    image_width,
    image_height,
):
    data = initialise_data_dictionary(coco_labels_filepath + "annotations.json")

    # load coco info
    coco = COCO(coco_labels_filepath + "annotations.json")
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    for img_id in img_ids:
        print(img_id)
        # get annotations for image
        anns_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        coco_image = coco.loadImgs([img_id])[0]
        image_width = coco_image["width"]
        image_height = coco_image["height"]
        image_name = coco_image["file_name"].split("/")[1].split(".")[0]
        distortion_mapping_x, distortion_mapping_y  = calc_distortion_mapping(camera_parameter_file_path, image_width, image_height)
        print(distortion_mapping_x.shape)
        print(distortion_mapping_y.shape)
        # loop through, correcting scale on each annotation and appending to new list of annotations
        for ann in anns:
            n_points = int(len(ann["segmentation"][0])/2)
            bbox = np.reshape(np.array(ann["bbox"], dtype=float), [2,2], order='C')
            bbox = np.array(undistort_points(bbox, distortion_mapping_x, distortion_mapping_y), dtype=int)
            seg = np.reshape(np.array(ann["segmentation"][0], dtype=float), [n_points, 1 ,2], order='C')
            seg = np.array(undistort_points(seg, distortion_mapping_x, distortion_mapping_y), dtype=int)
            new_annotation = {
                "segmentation": [seg.flatten(order='C').tolist()],
                "area": ann["area"],
                "iscrowd": 0,
                "image_id": ann["image_id"],
                "bbox": bbox.flatten(order='C').tolist(),
                "category_id": ann["category_id"],
                "id": ann["id"],
            }
            data["annotations"].append(new_annotation)
    # save new annotations file
    with open(output_coco_labels_filepath + "annotations.json", "w") as outfile:
        json.dump(data, outfile)



def distort_correct_all_images(images_directory,
                       camera_parameter_file_path, output_directory):
    map_x = None
    map_y = None
    for image_name in os.listdir(images_directory):
        if image_name[-4:] == '.jpg':
            image = cv2.imread(images_directory + image_name)
            if map_x is None:
                width = image.shape[1]
                height = image.shape[0]
                map_x, map_y = calc_distortion_mapping(camera_parameter_file_path, width, height)
            new_image = correct_distortion(image, map_x, map_y)
            write_status = cv2.imwrite(output_directory+image_name, new_image)
            if write_status is True:
                print("image " + image_name + " written")
            else:
                print("problem writing image " + image_name)

def distortion_correct_entire_coco_directory(
    input_dataset_directory, output_dataset_directory, auv_dive_particle_filter_file
):
    for directory in ["train/", "val/"]:
        Path(output_dataset_directory + directory + "JPEGImages").mkdir(
            parents=True, exist_ok=True
        )
        coco_labels_distort_correction(
            input_dataset_directory + directory,
            output_dataset_directory + directory,
            auv_dive_particle_filter_file,
            1280,
            1024,
        )
        distort_correct_all_images(
            input_dataset_directory + directory + "JPEGImages/",
            auv_dive_particle_filter_file,
            output_dataset_directory + directory + "JPEGImages/",
        )


def main():
    distortion_correct_entire_coco_directory('/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/histogram_normalised/no_distortion_correction/not_rescaled/', '/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/not_rescaled/', '/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/configuration/camera_parameters_unagi6k.yml')


if __name__ == "__main__":
    main()

