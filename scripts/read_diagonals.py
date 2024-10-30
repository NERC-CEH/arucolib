# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import
"""Read aruco marker metrics from images and
output results to a csv file. Metrics include
length per pixel and pixel dimensions.

Example:
read_diagonals.py "C:/temp/images" "aruco_stats.csv"
"""
#read_diagonals.py "C:/Users/Graham Monkman/OneDrive/Documents/PHD/Phase 03 - Novel Applications/Machine Vision/aruco_px_est_imgs" "aruco_stats.csv"
import argparse
import os.path as _path

import numpy as np
import cv2
from scipy import linalg
from enum import Enum

from opencvlib.view import show
from opencvlib.view import KeyBoardInput as Key
from opencvlib.distance import L2dist
from opencvlib.common import draw_str
from opencvlib.imgpipes.generators import FromPaths
from funclib.iolib import get_file_parts2
from plotlib import qplot
from sympy.geometry import Segment2D, Triangle
from funclib import iolib
from funclib.iolib import writecsv
from funclib.iolib import folder_open



def draw_polygon(img, points):
    """(ndarray, tuple|list)
    Join points
    """
    #[10,5],[20,30],[70,20],[50,10]
    points = np.array(points).astype('int32')
    p = points.reshape(-1, 1, 2)
    cv2.polylines(img, p, isClosed=True, color=(0, 0, 255), thickness=20)


class eUsedIDs(Enum):
    """enum for my sizes"""
    Sz25 = 49
    Sz30 = 18
    Sz30_flip = 528
    Sz50 = 22

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)


def main():
    """main"""
    cmdline = argparse.ArgumentParser(description=__doc__) #use the module __doc__

    #named: eg script.py -part head
    cmdline.add_argument('imgpath', help='Path containing the images')
    cmdline.add_argument('outcsv', help='CSV file name in which to save the results. File must not exist. Saved in same folder as images.')
    args = cmdline.parse_args()

    outcsv = _path.normpath(args.outcsv)
    if iolib.file_exists(outcsv):
        print('Output file %s already exists.' % outcsv)
        return

    print('Press "q" to quit')
    FP = FromPaths(args.imgpath)
    results = []
    results.append(['fname', 'marker_sz', 'side_mean_px', 'px_per_mm_from_sides', 'diag_mean_px', 'px_per_mm_from_diag'])
    for frame, fname, _ in FP.generate():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fpath, fname, _ = get_file_parts2(fname)

        res = cv2.aruco.detectMarkers(frame, dictionary)
        #res[0]: Detected corners [0]=topleft [1]=topright [2]=bottomright [3]=bottomleft
        #res[1]: MarkerID
        #res[2]: Rejected Candidates
        if res[0]:
            #print(res[0],res[1],len(res[2]))
            P = np.array(res[0]).squeeze().astype('int32')
            for ind, lbl in enumerate(res[1]):
                if len(P.shape) == 2:
                    pts = P
                else:
                    pts = P[ind]

                if isinstance(lbl, (tuple, list, np.ndarray)):
                    lbl = lbl[0]

                if lbl == eUsedIDs.Sz25.value:
                    v = 25.
                    T = Triangle([0, 0], [25, 0], [25, 25])
                    vdiag = T.hypotenuse.length
                elif lbl in [eUsedIDs.Sz30.value, eUsedIDs.Sz30_flip.value]:
                    v = 30.
                    T = Triangle([0, 0], [30, 0], [30, 30])
                    vdiag = max([s.length.evalf() for s in T.sides])
                elif lbl == eUsedIDs.Sz50.value:
                    v = 50.
                    T = Triangle([0, 0], [50, 0], [50, 50])
                    vdiag = max([s.length.evalf() for s in T.sides])
                else:
                    print('%s, marker detected but unknow. Was the image flipped' % fname)
                    lbl = lbl
                    v = 0
                    vdiag = 0

                ab = L2dist(pts[0], pts[1])
                bc = L2dist(pts[1], pts[2])
                cd = L2dist(pts[2], pts[3])
                da = L2dist(pts[3], pts[0])
                ac = L2dist(pts[0], pts[2]) #diag
                db = L2dist(pts[1], pts[3]) #diag
                diag_mean_px = (ac + db) / 2.
                side_mean_px = sum([ab, bc, cd, da])/4 #mean pixel difference
                px_per_mm_from_sides = v / side_mean_px  #mm's per pixel from sides
                px_per_mm_from_diag = vdiag / diag_mean_px

                results.append([fname, int(v), side_mean_px, px_per_mm_from_sides, diag_mean_px, px_per_mm_from_diag])

                s = 'Marker:{0} mm.  Px:{1:.2f} mm'.format(v, px_per_mm_from_sides)
                draw_str(frame, pts[0][0], pts[0][1], s, color=(0, 0, 0), scale=1.0)

                cv2.aruco.drawDetectedMarkers(frame, res[0], None, borderColor=(0, 0, 255))
        else:
            results.append([fname, '', '', '', '', ''])

       # if Key.check_pressed_key('q', show(frame, title=fname)):
           # return

    outcsv = _path.normpath(_path.join(fpath, outcsv))
    writecsv(outcsv, results, inner_as_rows=False)
    print('Saved results to %s' % outcsv)
    folder_open(fpath)


if __name__ == "__main__":
    main()
