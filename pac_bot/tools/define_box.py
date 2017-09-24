from pynput import mouse, keyboard


class Definer(object):
    def __init__(self):
        self._start = False
        self.origin = None
        self.terminus = None

    def _start_defining(self):
        """Private function, used to start defining the capture area.
        The mouse event listener is activate when the designated key is
        pressed.

        Returns:
            None
        """
        def on_press(key):
            if key == keyboard.Key.ctrl:
                self._start = True
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            print('Press [ctrl] to start defining capture area')
            listener.join()

    def _drag_to_define(self):
        """Private function, used to detect mouse action to define the
        capture area.

        Returns:
            None
        """
        print('Drag to define capture area')

        def on_click(x, y, button, pressed):
            if not self._start:
                return False
            elif button == mouse.Button.left and self._start:
                # Drag middle mouse to define capture area
                if pressed:
                    self.origin = (x, y)
                    print('Originated from {}'.format(self.origin))
            if not pressed:
                self.terminus = (x, y)
                print('Ended at {}'.format(self.terminus))
                return False

        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

    def define(self):
        """Calls the two private function to define the capture area.
        Call this function when defining an area. At the meantime,

        Returns:
            None
        """
        self._start_defining()
        self._drag_to_define()
