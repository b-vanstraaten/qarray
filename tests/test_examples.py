import unittest
from pathlib import Path

class ExampleTests(unittest.TestCase):

    def run_all_examples_for_error(self):
        """
        Test to check all the examples run without errors
        :return:
        """

        for python_file in Path(__file__).parent.parent.glob('examples/*.py'):
            with open(python_file, mode = 'r+') as f:
                try:
                    exec(f.read())
                except Exception as e:
                    raise RuntimeError(f"Error in {python_file.name}") from e

if __name__ == '__main__':
    unittest.main()