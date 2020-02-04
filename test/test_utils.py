import unittest
import tempfile
from src import utils
import os
import json

class TestUtils(unittest.TestCase):

    def test_save_new_annotations_file(self):
        output_filepath = tempfile.mkstemp()[1]
        data = {"test":[0,1,5,4]}
        try:
            utils.save_new_annotations_file(output_filepath, data)
            with open(output_filepath+'annotations.json') as json_file:
                result = json.load(json_file)
        finally:
            os.remove(output_filepath)
        self.assertEqual(result, data)



if __name__ == '__main__':
    unittest.main()