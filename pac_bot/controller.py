import pyautogui as pag
import time


def commence(countdown=5):
    """Call this method each time before a game to start a countdown

    Args:
        countdown: Time in seconds before any operations. Defaults 5 sec.

    Returns:
        None
    """
    for i in range(countdown):
        print(countdown - i, end=' ')
        time.sleep(0.25)
        print('.', end=' ')
        time.sleep(0.25)
        print('.', end=' ')
        time.sleep(0.5)


def left():
    pag.press('left')


def right():
    pag.press('right')


def up():
    pag.press('up')


def down():
    pag.press('down')
