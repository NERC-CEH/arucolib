# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
""" test aruco detection in video"""

import numpy as np
import cv2
from enum import Enum
#from opencvlib.view import show
from opencvlib.distance import L2dist
from opencvlib.common import draw_str

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
    Sz50 = 22

cap = cv2.VideoCapture(0)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.aruco.detectMarkers(frame, dictionary)
    #res[0]: Detected corners [0]=topleft [1]=topright [2]=bottomright [3]=bottomleft
    #res[1]: MarkerID
    #res[2]: Rejected Candidates
    if res[0]:
        #print(res[0],res[1],len(res[2]))
        lbls = [121 for x in res[1]]
        P = np.array(res[0]).squeeze().astype('int32')
        lbls = []
        for ind, lbl in enumerate(res[1]):
            if lbl == eUsedIDs.Sz25.value:
                v = 25
            elif lbl == eUsedIDs.Sz30.value:
                v = 30
            elif lbl == eUsedIDs.Sz50.value:
                v = 50
            else:
                lbl = lbl
                v = 0
            if len(P.shape) == 2:
                pts = P
            else:
                pts = P[ind]
            ab = L2dist(pts[0], pts[1])
            bc = L2dist(pts[1], pts[2])
            cd = L2dist(pts[2], pts[3])
            da = L2dist(pts[3], pts[0])
            mean = sum([ab, bc, cd, da])/4 #mean pixel difference
            px_len = v / mean
            s = 'Marker:{0} mm.  Px:{1:.2f} mm'.format(v, px_len)
            lbls.append(s)
            draw_str(frame, pts[0][0], pts[0][1], s, color=(0, 0, 0), scale=1.0)
            
        
        cv2.aruco.drawDetectedMarkers(frame, res[0], None, borderColor=(0, 0, 255))
        #draw_polygon(gray, res[0])

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
