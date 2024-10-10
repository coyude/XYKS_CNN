import pyautogui
import time
def less(start_x, start_y, size):
    pyautogui.moveTo(start_x, start_y, duration=0)
    pyautogui.mouseDown()
    pyautogui.moveRel(-size, size, duration=0)
    pyautogui.mouseUp()
    pyautogui.mouseDown()
    pyautogui.moveRel(size, size, duration=0)
    pyautogui.mouseUp()
    time.sleep(0.4)


def greater(start_x, start_y, size):
    pyautogui.moveTo(start_x, start_y, duration=0)
    pyautogui.mouseDown()
    pyautogui.moveRel(size, size, duration=0)
    pyautogui.mouseUp()
    pyautogui.mouseDown()
    pyautogui.moveRel(-size, size, duration=0)
    pyautogui.mouseUp()
    time.sleep(0.4)
