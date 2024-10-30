"""main init file for package opencvlib"""
import os.path as _path

import cv2 as _cv2

IMAGE_EXTENSIONS = ['.bmp',
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.tif',
                    '.tiff',
                    '.pbm',
                    '.pgm',
                    '.ppm']

# IMAGE_EXTENSIONS.extend(list(map(str.upper, IMAGE_EXTENSIONS)))

IMAGE_EXTENSIONS_AS_WILDCARDS = ['*.bmp',
                                 '*.jpg',
                                 '*.jpeg',
                                 '*.png',
                                 '*.tif',
                                 '*.tiff',
                                 '*.pbm',
                                 '*.pgm',
                                 '*.ppm']

# IMAGE_EXTENSIONS_AS_WILDCARDS.extend(list(map(str.upper, IMAGE_EXTENSIONS_AS_WILDCARDS)))



# Global logger
# To use:
# <package>.info("informational message")
# <package>.debug("debug message")
# <package>.critical("informational message")
# try:
#    import funclib.log as _log
#    _logfile = _path.join(_log.RootLogger.USER_TEMP_FOLDER, 'opencvlib.log')
#    _rootlogger = _log.RootLogger(_logfile)
#    Log = _rootlogger.logger
#    print('Logging to', _logfile)
# except Exception as e:
#    print('Logger initialisation failed for file %s.\nError: %s' % (_logfile, str(e)))


# def loginfo():
#    """print log file status"""
#    try:
#        print(_rootlogger)
#    except Exception as e:
#        print('Failed to get log info for file %s.\nError: %s' % (_logfile, str(e)))


def getimg(img, outflag=_cv2.IMREAD_UNCHANGED):
    """(ndarray|str)->ndarray
    tries to load the image if its a path and returns the loaded ndarray
    otherwise returns input img if it is an ndarray

    Also consider using @decs._decs.decgetimg decorator
    """
    if isinstance(img, str):
        return _cv2.imread(_path.normpath(img), outflag)
    return img


__all__ = ['classifiers', 'common',
           'color',
           'decs', 'display_utils', 'distance',
           'edges', 'errors',
           'faces', 'features',
           'geom', 'getimg',
           'histo', 'hogview',
           'IMAGE_EXTENSIONS', 'IMAGE_EXTENSIONS_AS_WILDCARDS', 'info',
           'keypoints',
           'matcher',
           'perspective', 'player',
           'roi',
           'streams',
           'template_match', 'transforms',
           'video',
           'view',
           'winpyr']
