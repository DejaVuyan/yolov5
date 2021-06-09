import torch
import cv2
from PIL import Image
import os

# load model
personModel = torch.hub.load('/home/yyz/code/yolov5',
                             'custom',
                             path='/home/yyz/code/yolov5/weights/yolov5x.pt',
                             source='local')

ppeModel = torch.hub.load('/home/yyz/code/yolov5',
                          'custom',
                          path='/home/yyz/code/yolov5/m1.pt',
                          source='local')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images



def check(imgName,prelist):
    # check if the helment and working cloth exist in the list.
    helmentNumber = prelist.count(0.0)
    workingcloNumber = prelist.count(1.0)
    print(f'imgName={imgName},predictList={prelist}  helmentNumber={helmentNumber},workingcloNumber={workingcloNumber}')
    if helmentNumber > 0:
        print("helments at least one in the images")
    else:
        print("no helments")

    if workingcloNumber > 0:
        print("working clothes at least one in the images")
    else:
        print("no working clothes")


#load original images
imgPath = '/home/yyz/code/yolov5/data/images'
originalData = load_images_from_folder(imgPath)

# predict the person and save in the personPath
personPath = 'runs'
personResults = personModel(originalData, size=640)
personResults.crop(personPath)

imgs = load_images_from_folder(os.path.join(personPath, 'crops/person'))
results = ppeModel(imgs, size=640)
print(results.xyxy[0].T[5].tolist())

videoPath = '/home/yyz/code/Yolov5-Deepsort/20210515_152115.mp4'
vidcap = cv2.VideoCapture(videoPath)
if vidcap.isOpened():
    while True:
        success, frame = vidcap.read()
        if not success: break  # if in the end of video
        # funciong: processing image
        # cv2.imshow('frame',frame)
        personPath = 'runs'
        personResults = personModel(frame, size=640)
        personResults.crop(personPath)
        imgs = load_images_from_folder(os.path.join(personPath, 'crops/person'))
        results = ppeModel(imgs, size=640)
        print(results.xyxy[0].T[5].tolist())
        #check(results.xyxy[0].T[5].tolist())
        print('get it ')
        break

else:
    print('视频打开失败')
    exit()