import unittest

import numpy as np
from pathlib import Path


class ExampleTests(unittest.TestCase):

    def test_examples(self):
        """
        Test to check all the examples run without errors
        :return:
        """

        for python_file in Path(__file__).parent.parent.glob('examples/*.py'):
            with open(python_file, mode = 'r') as f:
                print(f"Running {python_file.name}")
                try:
                    exec(f.read())
                except Exception as e:
                    raise RuntimeError(f"Error in {python_file.name}") from e
