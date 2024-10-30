# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import, no-self-use, unused-variable
"""unit tests for features"""
import unittest
from inspect import getsourcefile as _getsourcefile
import os.path as _path

import cv2
import numpy as np
from sympy import Point2D

import funclib.iolib as iolib
import opencvlib.aruco as aruco
from opencvlib.view import show
import opencvlib.imgpipes.generators as gen



class Test(unittest.TestCase):
    """unittest for keypoints"""

    def setUp(self):
        """setup variables etc for use in test cases
        """
        self.pth = iolib.get_file_parts2(_path.abspath(_getsourcefile(lambda: 0)))[0]
        self.modpath = _path.normpath(self.pth)
        self.fidpath = _path.normpath(_path.join(self.modpath, 'bin/images/fid/18_22_49.jpg'))
        self.fidimg = cv2.imread(self.fidpath) #this contains an image with 6 markers (3 are horizontal flip of other 3
        self.fidflippath = _path.normpath(_path.join(self.modpath, 'bin/images/fid/18_22_49_flipped.jpg'))
        self.fidflipimg = cv2.imread(self.fidflippath) #this contains an image with 6 markers (3 are horizontal flip of other 3
        self.aruco_rulers = _path.normpath(_path.join(self.modpath, 'bin/images/aruco_rulers_1.jpg'))
        self.aruco_ruler1 = r'C:\development\peat-auto-level\autolevel\test\bin\images\ruler1'


    @unittest.skip("Temporaily disabled while debugging")
    def test_getmarker(self):
        """test"""
        for n in range(50):
            s = _path.join('C:/Users/Graham Monkman/OneDrive/Documents/PHD/images/fudicial', '%s.jpg' % n)
            dummy = aruco.getmarker(n, 200, 20)
            self.assertTrue(isinstance(dummy, np.ndarray))


    @unittest.skip("Temporaily disabled while debugging")
    def test_Marker(self):
        """test"""
        M = aruco.Marker([[0, 0], [0, 10], [10, 10], [10, 0]], aruco.eMarkerID20mm.Sz25)
        self.assertEqual(M.diagonal_length_mm, aruco.DIAGONAL25mm)
        M = aruco.Marker([[0, 0], [0, 10], [10, 10], [10, 0]], aruco.eMarkerID20mm.Sz30)
        self.assertEqual(M.diagonal_length_mm, aruco.DIAGONAL30mm)
        M = aruco.Marker([[0, 0], [0, 10], [10, 10], [10, 0]], aruco.eMarkerID20mm.Sz50)
        self.assertEqual(M.diagonal_length_mm, aruco.DIAGONAL50mm)


    @unittest.skip("Temporaily disabled while debugging")
    def test_Detect(self):
        """test"""
        D = aruco.Detected(self.fidimg)
        markers = D.detect()
        img = D.image_with_detections
        show(img)

        D = aruco.Detected(self.fidflipimg)
        markers = D.detect()
        img = D.image_with_detections
        show(img)
        pass

    # @unittest.skip("Temporaily disabled while debugging")
    def test_detect_aruco_rulers(self):
        """test aruco rulers"""
        for img, _, _ in gen.FromPaths(self.aruco_ruler1).generate():
            D = aruco.Detected(img)
            markers = D.detect()
            img = D.image_with_detections
            show(img)
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
