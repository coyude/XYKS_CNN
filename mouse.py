from pynput.mouse import Button, Controller
import time

mouse = Controller()

def less(start_x, start_y, size):
    mouse.position = (start_x, start_y)
    time.sleep(0.01)  
    mouse.press(Button.left)
    time.sleep(0.01)
    mouse.move(-size, size)
    time.sleep(0.01)
    mouse.release(Button.left)
    time.sleep(0.01)
    mouse.press(Button.left)
    time.sleep(0.03)
    mouse.move(size, size)
    time.sleep(0.05)
    mouse.release(Button.left)
    time.sleep(0.6)


def greater(start_x, start_y, size):
    mouse.position = (start_x, start_y)
    time.sleep(0.01)
    mouse.press(Button.left)
    time.sleep(0.01)
    mouse.move(size, size)
    time.sleep(0.01)
    mouse.release(Button.left)
    time.sleep(0.01)
    mouse.press(Button.left)
    time.sleep(0.03)
    mouse.move(-size, size)
    time.sleep(0.05)
    mouse.release(Button.left)
    time.sleep(0.6)



