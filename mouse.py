from pynput.mouse import Button, Controller
import time

mouse = Controller()

def less(start_x, start_y, size):
    mouse.position = (start_x, start_y)
    time.sleep(0.0005)
    mouse.press(Button.left)
    time.sleep(0.0001)
    mouse.move(-size, size)
    time.sleep(0.0001)
    mouse.release(Button.left)
    time.sleep(0.0001)
    mouse.press(Button.left)
    time.sleep(0.0001)
    mouse.move(size, size)
    time.sleep(0.0001)
    mouse.release(Button.left)
    time.sleep(0.4)


def greater(start_x, start_y, size):
    mouse.position = (start_x, start_y)
    time.sleep(0.0005)
    mouse.press(Button.left)
    time.sleep(0.0001)
    mouse.move(size, size)
    time.sleep(0.0001)
    mouse.release(Button.left)
    time.sleep(0.0001)
    mouse.press(Button.left)
    time.sleep(0.0001)
    mouse.move(-size, size)
    time.sleep(0.0001)
    mouse.release(Button.left)
    time.sleep(0.4)



