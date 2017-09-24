import time

import cv2
import numpy as np
import pyscreenshot as image_grab
from tools.define_box import Definer


def capture(box_end_start=None, quit_on='q', verbose=False):
    """Start capturing the designated part of the screen.

    Args:
        box_end_start: A list of four numbers that are the origin and terminal
            coordinates of the capture area. If this argument is None, then
            you will be prompt to use mouse drag to select an area on the screen.
        quit_on: Quit capturing on this keystroke, defaults q.
        verbose: Set to True to print time each frame.

    Returns:
        None
    """
    if box_end_start is not None:
        origin = box_end_start[:2]
        terminus = box_end_start[2:]
    else:
        definer = Definer()
        definer.define()
        origin = definer.origin
        terminus = definer.terminus

    while True:
        frame_start_time = time.time()
        screen = np.array(image_grab.grab(bbox=(*origin, *terminus)))
        cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if verbose:
            print('Time since last frame: {}'.format(time.time() - frame_start_time))
        if cv2.waitKey(30) & 0xFF == ord(quit_on):
            break
    cv2.destroyAllWindows()
    cv2.waitKey()
