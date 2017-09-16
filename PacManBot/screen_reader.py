import cv2
import numpy as np
import pyscreenshot as ImageGrab
from pymouse import PyMouse, PyMouseEvent


class ScreenReader(object):
    def __init__(self, origin, terminus):
        """Constructor for the screen reader class, this constructor takes two
        arguments, origin and terminus each of which is a coordinate on the screen.

        Args:
            origin: A tuple or list that contains two elements; this is the coordinate for
                the starting point on the region to be read. This should be the top left
                corner of the box.
            terminus: A tuple or list that contains two elements; this is the coordinate for
                the ending point on the region to be read. This should be the bottom right
                corner of the box.
        """
        self.origin = origin
        self.terminus = terminus

    def capture(self, quit_on='q'):
        """Start capturing the designated part of the screen.

        Args:
            quit_on: Quit capturing on this keystroke, defaults q.

        Returns:
            None
        """

        while True:
            screen = np.array(ImageGrab.grab(bbox=(*self.origin, *self.terminus)))
            cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(30) == ord(quit_on):
                cv2.destroyAllWindows()
                break
