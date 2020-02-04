import cv2
import os
import numpy as np
from pathlib import Path
import yaml
from utils import *



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


def coco_labels_distort_correction(
    coco_labels_filepath,
    output_coco_labels_filepath,
    altitude_data_filepath,
    image_width,
    image_height,
):
    data = initialise_data_dictionary(coco_labels_filepath + "annotations.json")

    # load coco info
    coco = COCO(coco_labels_filepath + "annotations.json")
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    for img_id in img_ids:
        # get annotations for image
        anns_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        coco_image = coco.loadImgs([img_id])[0]
        image_width = coco_image["width"]
        image_height = coco_image["height"]
        image_name = coco_image["file_name"].split("/")[1].split(".")[0]
        altitude = get_altitude(image_name, altitude_data_filepath)
        print(
            "altitude: "
            + str(altitude)
            + ", image height: "
            + str(image_height)
            + ", vertical opening angle: "
            + str(vertical_opening_angle)
        )
        pixel_height = get_pixel_size(altitude, image_height, vertical_opening_angle)
        pixel_width = get_pixel_size(altitude, image_width, horizontal_opening_angle)
        vertical_rescale = pixel_height / target_pixel_size
        horizontal_rescale = pixel_width / target_pixel_size
        print(
            "pixel_height - " + str(pixel_height) + ", pixel_width: " + str(pixel_width)
        )
        print(
            "rescaling coco - " + str(vertical_rescale) + " " + str(horizontal_rescale)
        )

        for image in data["images"]:
            if image["id"] == img_id:
                image["height"] = int(image["height"] * vertical_rescale)
                image["width"] = int(image["width"] * horizontal_rescale)
        # loop through, correcting scale on each annotation and appending to new list of annotations
        for ann in anns:
            new_area = ann["area"] * horizontal_rescale * vertical_rescale
            new_bounding_box = [
                p * vertical_rescale if i % 2 else p * horizontal_rescale
                for i, p in enumerate(ann["bbox"])
            ]
            new_segmentation = [
                p * vertical_rescale if i % 2 else p * horizontal_rescale
                for i, p in enumerate(ann["segmentation"][0])
            ]
            new_annotation = {
                "segmentation": [new_segmentation],
                "area": new_area,
                "iscrowd": 0,
                "image_id": ann["image_id"],
                "bbox": new_bounding_box,
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
            auv_dive_particle_filter_file,
            output_dataset_directory + directory,
            1280,
            1024,
        )
        distort_correct_all_images(
            input_dataset_directory + directory + "JPEGImages/",
            auv_dive_particle_filter_file,
            output_dataset_directory + directory + "JPEGImages/",
        )






correct_all_images('./processed/image/i20180805_215810/for_iridis/histogram_normalised/no_distortion_correction/not_rescaled/train/JPEGImages/', './configuration/camera_parameters_unagi6k.yml' , './processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/not_rescaled/train/JPEGImages/')
