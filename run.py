import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import pyautogui
from torchvision import transforms
import time
from mouse import *



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)


        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10) 

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        x = x.view(-1, 128 * 4 * 4)  
        x = F.relu(self.fc1(x))     
        x = F.relu(self.fc2(x))     
        x = self.fc3(x)            

        return x

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
])


model = CNN().to(device)
model.load_state_dict(torch.load('svhn_cnn1.pth', map_location=device)) 
model.eval()

def preprocess_image(pil_image):
    image = transform(pil_image).unsqueeze(0).to(device)
    return image

def predict_digit(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()




def predict(screenshot_pil, padding=7):
    screenshot = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

  
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    number = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w >= 15 and w<=65) and (h >= 15 and h<=60):
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, screenshot.shape[1] - x)
            h = min(h + 2 * padding, screenshot.shape[0] - y)
            number.append((x, y, w, h))

    if not number:
        print("未检测到数字")
        return []

    number = sorted(number, key=lambda b: b[0])

    predictions = []

    for (x, y, w, h) in number:
        digit_roi = screenshot[y:y+h, x:x+w]



        digit_pil = Image.fromarray(cv2.cvtColor(digit_roi, cv2.COLOR_BGR2RGB))


        # plt.imshow(digit_pil)
        # plt.axis('off') 
        # plt.show()



        image = preprocess_image(digit_pil)
        # show_image_with_cv2(image)

        digit = predict_digit(image)
        predictions.append((digit, (x, y, w, h)))

    #     cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(screenshot, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.9, (0, 255, 0), 2)


    # cv2.imshow('111', screenshot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return predictions

for i in range(30):
    region1 = (786, 332, 128,128)
    region2 = (1020,334,128,128)
  
    screenshot1 = pyautogui.screenshot(region=region1)
    screenshot2 = pyautogui.screenshot(region=region2)
  
    predictions1 = predict(screenshot1)
    predictions2 = predict(screenshot2)
    digits = ''.join([str(d[0]) for d in predictions1])
    digits2 = ''.join([str(d[0]) for d in predictions2])
    if digits and digits2:
        print(f'识别的数字串: {digits,digits2}')
        digits=int(digits)
        digits2=int(digits2)
        if digits>digits2:
            greater(948,723,10)
        elif digits<digits2:
            less(948,723,10)
    else:
        time.sleep(0.4)
    
