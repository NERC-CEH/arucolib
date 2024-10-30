# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import
"""Detect aruco markers from images and
output results to a csv file. Metrics include
length per pixel and pixel dimensions.

Example:
aruco_detect_and_show.py "C:/temp/images"
"""
import argparse
import os.path as _path
from enum import Enum

import numpy as np
import cv2
from sympy.geometry import Segment2D, Triangle

from funclib import iolib

from opencvlib.view import show
import opencvlib.aruco as aruco
import opencvlib.transforms as transforms


from opencvlib.imgpipes.generators import FromPaths




def main():
    """main"""
    cmdline = argparse.ArgumentParser(description=__doc__)  # use the module __doc__

    # named: eg script.py -part head
    cmdline.add_argument('imgpath', help='Path containing the images')
    args = cmdline.parse_args()
    pth = _path.normpath(args.imgpath)
    FP = FromPaths(pth)
    for Img, _, _ in FP.generate():
        Img = transforms.resize(Img, width=768)
        Img = transforms.togreyscale(Img)  # noqa
        Img = transforms.denoise_greyscale(Img)
        Ar = aruco.Detected(Img, side_length_mm=10)
        show(Ar.image_with_detections)  # noqa
        Ar.export('%s/%s' % (pth, 'detections'))


if __name__ == "__main__":
    main()
