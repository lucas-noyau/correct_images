import unittest
import os
import sys
import math
SRC_DIR = os.path.dirname(os.path.abspath('./src/coco_scale_correction.py'))
print(SRC_DIR)
sys.path.append(SRC_DIR)
import coco_scale_correction


class TestCocoScaleCorrection(unittest.TestCase):

    def test_get_pixel_size(self):
        altitude = 5
        image_size = 1000
        opening_angle = 90
        pixel_size = coco_scale_correction.get_pixel_size(altitude, image_size, opening_angle)
        self.assertEqual(pixel_size, 0.009999999999999998)



if __name__ == '__main__':
    unittest.main()
