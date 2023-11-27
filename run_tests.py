import unittest

runner = unittest.TextTestRunner()
exit(not runner.run(unittest.defaultTestLoader.discover("tests", pattern='*_test.py')).wasSuccessful())