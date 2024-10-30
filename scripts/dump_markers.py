"""Dump markers for rulers to the file system"""
import opencvlib.aruco as aruco

import funclib.iolib as iolib


def main():
    """main"""
    aruco.dump_markers(range(18), fld='C:/temp/markers/ruler1', border_sz=100, borderBits=1)
    aruco.dump_markers(range(18, 18*2), fld='C:/temp/markers/ruler2', border_sz=100, borderBits=1)
    aruco.dump_markers(range(18*2, 18 * 3), fld='C:/temp/markers/ruler3', border_sz=100, borderBits=1)
    aruco.dump_markers(range(18*3, 18*3 + 6), fld='C:/temp/markers/singletons_1', border_sz=100, borderBits=1)

    iolib.folder_open('C:/temp/markers')
    print('done')

if __name__ == '__main__':
    main()