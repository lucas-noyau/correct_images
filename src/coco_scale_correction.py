import math
from cv2 import imread, imwrite
from PIL import Image
import numpy as np
from pathlib import Path
import json
from pycocotools.coco import COCO
import os
from utils import *

# Unagi
# horiz 55.96341714   vert 47.81781398
vertical_opening_angle = 47.81781398
horizontal_opening_angle = 55.96341714
target_pixel_size = 0.00139

# SX3 (ae2000)
# hozriz 60.41137263    vert 52.11659949
# vertical_opening_angle = 52.11659949
# horizontal_opening_angle = 60.41137263


# Function for applying rescaling to a coco annotations file, coco_labels_filepath,
# and saving the result to output_coco_labels_filepath
def coco_labels_scale_correction(
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
    save_new_annotations_file(output_coco_labels_filepath, data)
    


def rescale_masks_and_images(
    image_directory,
    mask_directory,
    image_list_filename,
    target_pixel_size,
    data_filename,
    image_output_directory,
    mask_output_directory,
):
    image_list = open(image_list_filename, "r")
    for image_name in image_list.readlines():
        image = rescale_image(
            image_directory, image_name, target_pixel_size, data_filename
        )
        imwrite(image_output_directory + image_name.strip(), image)
        mask = imread(mask_directory + image_name.strip())
        mask = Image.fromarray(mask.astype("uint8"), "RGB")
        mask = np.array(mask.resize((image.shape[1], image.shape[0]), Image.BICUBIC))
        imwrite(mask_output_directory + image_name.strip(), mask)


def rescale_image(image_directory, image_name, target_pixel_size, data_filename):
    altitude = get_altitude(image_name.split(".")[0], data_filename)
    print(image_directory + image_name)
    image, image_width, image_height = get_image_and_info(image_directory + image_name)
    pixel_height = get_pixel_size(altitude, image_height, vertical_opening_angle)
    pixel_width = get_pixel_size(altitude, image_width, horizontal_opening_angle)
    vertical_rescale = pixel_height / target_pixel_size
    horizontal_rescale = pixel_width / target_pixel_size
    print("pixel height = " + str(pixel_height) + ", pixel width = " + str(pixel_width))
    print("rescaling image - " + str(vertical_rescale) + " " + str(horizontal_rescale))
    size = (int(image_width * horizontal_rescale), int(image_height * vertical_rescale))
    image = Image.fromarray(image.astype("uint8"), "RGB")
    image = image.resize(size, Image.BICUBIC)
    return np.array(image)


# uses given opening angle of camera and the altitude parameter to determine the pixel size
# give width & horizontal, or height & vertical
def get_pixel_size(altitude, image_size, opening_angle):
    image_spatial_size = (
        2 * float(altitude) * float(math.tan(math.radians(opening_angle / 2)))
    )
    pixel_size = image_spatial_size / image_size
    return pixel_size


def rescale_many_images(
    image_directory, target_pixel_size, data_filename, output_directory
):
    image_list = [
        filename for filename in os.listdir(image_directory) if filename[-4:] == ".jpg"
    ]
    print(image_list)
    for image_name in image_list:
        imwrite(
            output_directory + image_name,
            rescale_image(
                image_directory, image_name, target_pixel_size, data_filename
            ),
        )


def rescale_entire_coco_directory(
    input_dataset_directory, output_dataset_directory, auv_dive_particle_filter_file
):
    for directory in ["train/", "val/"]:
        Path(output_dataset_directory + directory + "JPEGImages").mkdir(
            parents=True, exist_ok=True
        )
        coco_labels_scale_correction(
            input_dataset_directory + directory,
            output_dataset_directory + directory,
            auv_dive_particle_filter_file,
            1280,
            1024,
        )
        rescale_many_images(
            input_dataset_directory + directory + "JPEGImages/",
            target_pixel_size,
            auv_dive_particle_filter_file,
            output_dataset_directory + directory + "JPEGImages/",
        )


# target_pixel_size = 0.0055
# rescale_entire_coco_directory('./processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/not_rescaled/', './processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/rescaled/', './auv_pf_LC.csv')
# rescale_entire_coco_directory('./processed/image/i20180805_215810/for_iridis/histogram_normalised/no_distortion_correction/not_rescaled/', './processed/image/i20180805_215810/for_iridis/histogram_normalised/no_distortion_correction/rescaled/', './auv_pf_LC.csv')
rescale_entire_coco_directory(
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/greyworld_correction/distortion_correction/not_rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/greyworld_correction/distortion_correction/rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/auv_pf_LC.csv",
)
rescale_entire_coco_directory(
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/not_rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/histogram_normalised/distortion_correction/rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/auv_pf_LC.csv",
)
rescale_entire_coco_directory(
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/attenuation_correction/distortion_correction/not_rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/processed/image/i20180805_215810/for_iridis/attenuation_correction/distortion_correction/rescaled/",
    "/Volumes/jw22g14_phd/fk2018/tunasand/20180805_215810_ts_un6k/auv_pf_LC.csv",
)
# rescale_entire_coco_directory(
#     "./processed/image/i20180805_215810/for_iridis/attenuation_correction/no_distortion_correction/not_rescaled/",
#     "./processed/image/i20180805_215810/for_iridis/attenuation_correction/no_distortion_correction/rescaled/",
#     "./auv_pf_LC.csv",
# )

target_pixel_size = 0.00139
# rescale_entire_coco_directory('./coco/dive1/D/', './coco/dive1/D_rescaled/', './filelist_auv_nav_D.csv')
# rescale_entire_coco_directory('./coco/dive2/A/', './coco/dive2/A_rescaled/', './filelist_auv_nav_A.csv')
# rescale_entire_coco_directory('./coco/dive2/B/', './coco/dive2/B_rescaled/', './filelist_auv_nav_B.csv')
# rescale_entire_coco_directory('./coco/dive2/C/', './coco/dive2/C_rescaled/', './filelist_auv_nav_C.csv')
# rescale_entire_coco_directory('./coco/dive3/B/', './coco/dive3/B_rescaled/', './filelist_auv_nav_B.csv')
# rescale_entire_coco_directory('./coco/dive3/C/', './coco/dive3/C_rescaled/', './filelist_auv_nav_C.csv')

# rescale_entire_coco_directory('./processed/image/i20180805_215810/attenuation_correction/distortion_correction/dropped_resolution/', './processed/image/i20180805_215810/attenuation_correction/distortion_correction/dropped_resolution/', './auv_pf_LC.csv')
# rescale_entire_coco_directory('./processed/image/i20180805_215810/greyworld_correction/distortion_correction/dropped_resolution/', './processed/image/i20180805_215810/greyworld_correction/distortion_correction/dropped_resolution/', './auv_pf_LC.csv')
# rescale_entire_coco_directory('./processed/image/i20180805_215810/greyworld_correction/no_distortion_correction/dropped_resolution/', './processed/image/i20180805_215810/greyworld_correction/no_distortion_correction/dropped_resolution/', './auv_pf_LC.csv')

# coco_labels_scale_correction('./large_megafauna/val/', './large_megafauna/', './large_megafauna/json_renav/csv/particle_filter/auv_pf_LC.csv', horizontal_opening_angle, vertical_opening_angle)
