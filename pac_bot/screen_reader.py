import time

import cv2
import numpy as np
import pyscreenshot as ImageGrab


def capture(origin, terminus, quit_on='q', verbose=False):
    """Start capturing the designated part of the screen.

    Args:
        origin: A tuple or list that contains two elements; this is the coordinate for
            the starting point on the region to be read. This should be the top left
            corner of the box.
        terminus: A tuple or list that contains two elements; this is the coordinate for
            the ending point on the region to be read. This should be the bottom right
            corner of the box.
        quit_on: Quit capturing on this keystroke, defaults q.
        verbose: Set to True to print time each frame.

    Returns:
        None
    """
    while True:
        frame_start_time = time.time()
        screen = np.array(ImageGrab.grab(bbox=(origin, terminus)))
        cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if verbose:
            print('Time since last frame: {}'.format(time.time() - frame_start_time))
        if cv2.waitKey(30) & 0xFF == ord(quit_on):
            cv2.destroyAllWindows()
            break
