# pylint: disable=C0103, too-few-public-methods, locally-disabled
"""Work with aruco markers

Use the Detected class for detecting markers in an image.

The Marker class represent individual markers detected
from Detected.
"""
import os.path as _path
from enum import Enum as _Enum

import math as _math
import numpy as _np
import cv2 as _cv2
from sympy.geometry import centroid as _centroid

# We use sympy because Point2D supports multiplication by floats,
# plus lots of other good stuff
from sympy.geometry import Point2D as _Point2D

import funclite.iolib as _iolib
import funclite.baselib as _baselib

import opencvlib.geom as _geom
from opencvlib import getimg as _getimg  # noqa
from opencvlib.common import draw_str as _draw_str
from opencvlib.common import draw_points as _draw_points
import opencvlib.color as _color
import opencvlib.view as _view

from opencvlib.geom import order_points as _order_points

_dictionary = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_ARUCO_ORIGINAL)

MARKERS = {0: 'DICT_4X4_50', 1: 'DICT_4X4_100', 2: 'DICT_4X4_250',
           3: 'DICT_4X4_1000', 4: 'DICT_5X5_50', 5: 'DICT_5X5_100', 6: 'DICT_5X5_250', 7: 'DICT_5X5_1000',
           8: 'DICT_6X6_50', 9: 'DICT_6X6_100', 10: 'DICT_6X6_250', 11: 'DICT_6X6_1000', 12: 'DICT_7X7_50',
           13: 'DICT_7X7_100', 14: 'DICT_7X7_250', 15: 'DICT_7X7_1000', 16: 'DICT_ARUCO_ORIGINAL'}


class Marker:
    """
    Represents a single detected marker

    Can be created with points in any order.

    Set the enum Marker.mode to Marker.Mode.mm to get points in real world units.

    Methods:
        diagonal_length_mm: the geommetrically calculated diagonal length of the marker in mm (e.g. 1.41421 ... for a 1cm square)
        side_length_mm: the side length in mm
        vertices_...: sympy Point2D representations of the corners
        mode: Toggle between pixel mode and mm mode. Affect Marker.points


    Raises:
        ErrMarkerExpectes4Points: If instance in initialised with an array of pts of len != 4

    Examples:
        Instantiate a marker print points in real world mm units
        >>> M = Marker([[0,0], [10,10], [10,0], [0,10]], side_length_mm=20)
        >>> M.mode = Marker.Mode.mm
        >>> M.points
        [[0,0], [20,20], [20,0], [0,20]]
    """

    class Mode(_Enum):
        px = 2
        mm = 1

    def __init__(self, pts: (_np.ndarray, list, tuple), markerid, side_length_mm: float):

        if len(pts) != 4:
            raise ErrMarkerExpectes4Points('Marker requires 4 points to create instance, got %s' % str(pts))

        self.side_length_mm = side_length_mm
        self.diagonal_length_mm = diagonal(side_length_mm)
        self.mode = Marker.Mode.px
        p = _order_points(pts)
        self._vertices_topleft = _Point2D(p[0][0], p[0][1])
        self._vertices_topright = _Point2D(p[1][0], p[1][1])
        self._vertices_bottomright = _Point2D(p[2][0], p[2][1])
        self._vertices_bottomleft = _Point2D(p[3][0], p[3][1])
        self.markerid = markerid

    def __repr__(self):
        """pretty print"""
        info = ['Marker "%s"' % self.markerid, str(tuple(self._vertices_topleft * self._factor) if isinstance(self._vertices_topleft, _Point2D) else ''),
                str(tuple(self._vertices_topright * self._factor) if isinstance(self._vertices_topright, _Point2D) else ''),
                str(tuple(self._vertices_bottomright * self._factor) if isinstance(self._vertices_bottomright, _Point2D) else ''),
                str(tuple(self._vertices_bottomleft * self._factor) if isinstance(self._vertices_bottomleft, _Point2D) else '')]
        return ' '.join(info)

    @property
    def vertices_topleft(self):
        """
        Coordinates are give x,y BUT uses opencv origin at top left.

        Obeys Mode.

        Returns:
            sympy.Point2D: top left vertext
        """
        return self._vertices_topleft * self._factor

    @property
    def vertices_topright(self):
        """
        Coordinates are give x,y BUT uses opencv origin at top left.
        Obeys mode
        Returns:
            sympy.Point2D: top right vertext
        """
        return self._vertices_topright * self._factor

    @property
    def vertices_bottomright(self):
        """
        Coordinates are give x,y BUT uses opencv origin at top left.
        Obeys mode
        Returns:
            sympy.Point2D: Bottom right vertext
        """
        return self._vertices_bottomright * self._factor

    @property
    def vertices_bottomleft(self):
        """
        Coordinates are give x,y BUT uses opencv origin at top left.
        Obeys mode

        Returns:
            sympy.Point2D: Bottom left vertext
        """
        return self._vertices_bottomleft * self._factor

    @property
    def diagonal_px(self):
        """mean diagonal length"""
        if isinstance(self._vertices_topleft, _Point2D) and isinstance(self._vertices_bottomright, _Point2D):
            x = abs(self._vertices_topleft.distance(self._vertices_bottomright).evalf())
            y = abs(self._vertices_topleft.distance(self._vertices_bottomright).evalf())
            return (x + y) / 2
        return None

    @property
    def side_px(self):
        """mean side length in px"""
        if isinstance(self._vertices_topleft, _Point2D) and isinstance(self._vertices_bottomright, _Point2D):
            a = abs(self._vertices_topleft.distance(self._vertices_topright).evalf())
            b = abs(self._vertices_topleft.distance(self._vertices_bottomleft).evalf())
            c = abs(self._vertices_bottomright.distance(self._vertices_topright).evalf())
            d = abs(self._vertices_bottomright.distance(self._vertices_bottomleft).evalf())
            return (a + b + c + d) / 4
        return None

    @property
    def angle_horizontal(self) -> float:
        """
        Mean angle in degrees of two vertical lines.

        Invariant to Marker.mode

        Returns:
            float: Mean angle in degrees
        """
        a1 = _geom.angle_between_pts(self._vertices_topleft, self._vertices_topright)
        a2 = _geom.angle_between_pts(self._vertices_bottomleft, self._vertices_bottomright)
        return (a1 + a2) / 2

    @property
    def angle_vertical(self) -> float:
        """
        Mean angle in degrees of two horizontal lines.

        Invariant to Marker.mode

        Returns:
            float: Mean angle in degrees
        """
        a1 = _geom.angle_between_pts(self._vertices_topleft, self._vertices_bottomleft)
        a2 = _geom.angle_between_pts(self._vertices_topright, self._vertices_bottomright)
        return (a1 + a2) / 2

    @property
    def side_difference_vertical(self) -> float:
        """
        Vertical length diff between two vertical sides.

        Respects Marker.mode

        Returns:
            float: Vertical difference between left and right verticals, respects Marker.mode
        """
        return ((self._vertices_bottomleft[1] - self._vertices_topleft[1]) - (self._vertices_bottomright[1] - self._vertices_topright[1])) * self._factor

    @property
    def side_difference_horizontal(self) -> float:
        """
        Horizontal length diff between two vertical sides.

        Respects Marker.mode.

        Returns:
            float: Vertical difference between left and right verticals, respects Marker.mode
        """
        return ((self._vertices_topright[0] - self._vertices_topleft[0]) - (self._vertices_bottomright[0] - self._vertices_bottomleft[0])) * self._factor

    def px_length_mm(self, use_side=False) -> float:
        """
        Estimated pixel length in mm, i.e. the length of a pixel in mm.

        Args:
            use_side (bool): use the mean side pixel length rather than the mean diagonal length

        Returns: float: Size of marker in mm
        """
        if use_side:
            return self.side_length_mm / self.side_px
        return self.diagonal_length_mm / self.diagonal_px

    @property
    def points(self) -> list:
        """
        Get as list of xy points.

        Coordinates are given x,y BUT uses opencv origin at top left.

        Order clockwise from top left as first point.

        Respects Marker.mode.

        Returns:
            list: list of points

        Examples:
            >>> Marker.points
            [[0,10],[10,10],[10,0],[0,0]]
        """
        return [list(self.vertices_topleft), list(self.vertices_topright), list(self.vertices_bottomright), list(self.vertices_bottomleft)]


    @property
    def points_flattened(self) -> list:
        """
        Get as a list of depth 0 x and y values.

        i.e. ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

        Useful for writing out to pandas dataframes etc.

        Respects Marker.mode

        Returns:
            list: list of fully flattened points

        Notes:
            Cooerces into into ints if mode == Mode.px, else coerces to float.

        Examples:
            >>> Marker.points_flattened  # noqa
            [0,10,10,10,10,0,0,0]

        """
        coerce_type = int if self.mode == Marker.Mode.px else float
        pts = _baselib.list_flatten(self.points, coerce_type=coerce_type)
        return pts


    @property
    def centroid(self) -> list:
        """centroid of points, pixel based

        Respects Marker.mode.

        Returns:
            list: centroid point as list, [12.3, 10]
        """
        return list(_centroid(self._vertices_bottomleft * self._factor,
                              self._vertices_bottomright * self._factor,
                              self._vertices_topleft * self._factor,
                              self._vertices_topright * self._factor).evalf())



    @property
    def _factor(self) -> float:
        """Set factor for returning results as mm or px

        Raises:
            ErrInvalidMarkerMode: If self.Mode is no a member of Marker.Mode

        Returns:
            float: Factor to convert to mm, else 1 for pixels (i.e. as-is)
        """
        if self.mode == Marker.Mode.px:
            return 1
        elif self.mode == Marker.Mode.mm:
            return self.px_length_mm()
        else:
            raise ErrInvalidMarkerMode('Marker mode was invald. Ensure you use a member of class Marker.Mode')


class Detected:
    """Detect aruco markers in an image.

    Initalise an instance with an image and then detect
    markers by calling detect on the instance.

    Members:
        image: The original image as an ndarray or a path
        image_with_detections: The image with detections drawn on it
        Markers: A list containing Marker instances. A Marker instance is a detected marker.

    Examples:
        >>> D = Detected('c:/myimg.jpg')
    """

    def __init__(self, img, side_length_mm: float, detect=True):
        self.Markers = []
        self.image = _getimg(img)
        self.image_with_detections = _np.copy(self.image)
        self.side_length_mm = side_length_mm
        if detect:
            self.detect()

    def export(self, fld: str, override_name: str = None) -> str:
        """
        Export image to fld, with detections (if made)

        Creates a random name for the image

        Args:
            fld (str): folder to export to
            override_name (str, None): Explicitly pass the file to save the image to

        Returns: str: File name

        """
        if override_name:
            out = _path.normpath(override_name)
        else:
            fld = _path.normpath(fld)
            _iolib.create_folder(fld)
            tmp = _iolib.get_temp_fname(suffix='.jpg', name_only=True)
            out = _path.normpath(_path.join(fld, tmp))
        _cv2.imwrite(out, self.image_with_detections)
        return out

    def detect(self, expected=()):
        """
        Detect markers, returning those detected
        as a list of Marker class instances

        Args:
            expected (list, tuple): Marker ids, numbers

        Returns:
            list of Marker class instances, represented detected markers
        """
        self.Markers = []
        res = _cv2.aruco.detectMarkers(self.image, _dictionary)
        # res[0]: List of ndarrays of detected corners [][0]=topleft [1]=topright [2]=bottomright [3]=bottomleft. each ndarray is shape 1,4,2
        # res[1]: List containing an ndarray of detected MarkerIDs, eg ([[12, 10, 256]]). Shape n, 1
        # res[2]: Rejected Candidates, list of ndarrays, each ndarray is shape 1,4,2

        if res[0]:
            # print(res[0],res[1],len(res[2]))
            P = _np.array(res[0]).squeeze().astype('int32')

            for ind, markerid in enumerate(res[1]):
                markerid = markerid[0]
                if expected and markerid not in expected:
                    continue

                if len(P.shape) == 2:
                    pts = P
                else:
                    pts = P[ind]
                M = Marker([pts[0], pts[1], pts[2], pts[3]], markerid, side_length_mm=self.side_length_mm)
                self.Markers.append(M)

                # px per mm
                s = '{0} mm. Px:{1:.2f} mm'.format(int(M.side_length_mm), M.px_length_mm())
                _draw_str(self.image_with_detections, pts[0][0], pts[0][1], s, color=(0, 255, 0), scale=0.6)

                # The markerid, centre of marker
                _draw_str(self.image_with_detections, M.centroid[0], M.centroid[1], str(markerid), color=(255, 255, 255), scale=0.7, box_background=(0, 0, 0), centre_box_at_xy=True)

                # the points
                self.image_with_detections = _draw_points(pts, self.image_with_detections, join=True, line_color=(0, 255, 0), show_labels=False)

                # point coords in pixels at each vertext
                for pt in M.points:
                    sPt = '(%s,%s)' % (pt[0], pt[1])
                    _draw_str(self.image_with_detections, pt[0] + 10, pt[1] + 12, sPt, color=(0, 255, 0), scale=0.5)

        else:
            self.Markers = []
            self.image_with_detections = _np.copy(self.image)
        return self.Markers

    def show(self):
        """View detections. detect needs to have been first called.

        Returns: numpy.ndarray
        """
        _cv2.imshow('aruco detections', self.image_with_detections)
        _cv2.waitKey(0)
        _cv2.destroyAllWindows()

    def write(self, fname: str):
        """write image to file system

        Args:
            fname (str): save name

        Returns: None
        """
        fname = _path.normpath(fname)
        _cv2.imwrite(fname, self.image_with_detections)

    def height(self, as_mm=False) -> (int, float):
        """
        Image height. Uses mean pixel length calculated
        across all detected markers for real-world units.

        Args:
            as_mm (bool): Return as mm or pixels

        Returns: image height in pixels or mm
        """
        return self.image.shape[0] if not as_mm else self.image.shape[0] * self.px_length_mm

    def width(self, as_mm: bool = False) -> int:
        """
        Image width. Uses mean pixel length calculated
        across all detected markers for real-world units.

        Args:
            as_mm (bool): Return as mm or pixels

        Returns: image height in pixels or mm
        """
        return self.image.shape[1] if not as_mm else self.image.shape[1] * self.px_length_mm

    def channels(self) -> int:
        """
        Property channels
        Returns: channel nr.
        """
        return self.image.shape[2]

    @property
    def px_length_mm(self) -> (float, None):
        """
        Mean pixel length in mm for all detected markers
        in the image

        Returns:
            float: The mean pixel length
            None: If no detected markers

        """
        if self.Markers:
            px = [mk.px_length_mm() for mk in self.Markers]
            return sum(px) / len(px)
        return None  # noqa

def getmarker(markerid: int, sz_pixels: int = 500, border_sz: int = 0,
              border_color: tuple = _color.CVColors.white,
              borderBits=1, orientation_marker_sz=0,
              orientation_marker_color=_color.CVColors.black, saveas='') -> _np.ndarray:
    """
    Get marker image, i.e. the actual marker
    for use in other applications, for printing
    and saving as a jpg.

    Args:
        markerid (int):
            Dictionary lookup for MARKERS
        sz_pixels (int):
            Size of the whole marker in pixels, including the border as defined by borderBits
        border_sz (int):
            Border added around the marker, usually white, in pixels.
            This is added after the library has created the marker.
            The final image side length will be sz_pixels + border_sz
        border_color:
            tuple (0,0,0)
        borderBits:
            the black padding around the marker, in image pixels, where an
            image pixel is an single aruco marker building block, a
            marker is a 5 x 5 block. This is added by the library and included in the
            sz_pixels
        orientation_marker_sz:
            draw an orientation_point in top left of edge pixel size=orientation_marker_sz
        orientation_marker_color:
            marker color (3-tuple)
        saveas:
            filename to dump img to

    Returns:
        Image as an ndarray
    """
    m = _cv2.aruco.drawMarker(_dictionary, id=markerid, sidePixels=sz_pixels, borderBits=borderBits)
    m = _cv2.cvtColor(m, _cv2.COLOR_GRAY2BGR)
    if border_sz > 0:
        m = _view.pad_image(m, border_sz=border_sz, pad_color=border_color, pad_mode=_view.ePadColorMode.tuple_)

    if orientation_marker_sz > 0:
        m[0:orientation_marker_sz, 0:orientation_marker_sz, :] = orientation_marker_color

    if saveas:
        _cv2.imwrite(saveas, m)
    return m


def diagonal(x: float) -> float:
    """Length of diagonal of square of side length of x
    Args:
        x (float): length of side of marker

    Returns:
        length of diagonal
    """
    return _math.sqrt(x ** 2 + x ** 2)


def dump_markers(markerids: any, fld: str, **kwargs):
    """
    Dump markers to folder fld

    Args:
        markerids (any): tuple, list of even a range function, which is the markerids to dump
        fld:
        **kwargs: keyword arguments, passed to aruco.getmarker

    Returns: None

    Notes:
        markerids can accept a range function, e.g. range(10)
    """
    fld = _path.normpath(fld)
    _iolib.create_folder(fld)
    for id_ in markerids:
        saveas = '%s.jpg' % id_
        saveas = _path.normpath(_path.join(fld, saveas))
        _ = getmarker(id_, saveas=saveas, **kwargs)


# -----------------------
# Errors
# -----------------------
class ErrMarkerExpectes4Points(Exception):
    """Marker requires 4 points"""


class ErrInvalidMarkerMode(Exception):
    """Marker mode was invald. Ensure you use a member of class Marker.Mode"""


# -----------------------


if __name__ == '__main__':
    D = Detected(r'C:\development\peat-auto-level\autolevel\bin\images\v2\rulers_15mm.jpg', 10.05)
