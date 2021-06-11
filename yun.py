import torch
import cv2
from PIL import Image
import os

# load model
personModel = torch.hub.load('/home/yyz/code/yolov5',
                             'custom',
                             path='/home/yyz/code/yolov5/weights/yolov5x.pt',
                             source='local')
personModel.classes = [0]   #filter by classes, only predict person.

ppeModel = torch.hub.load('/home/yyz/code/yolov5',
                          'custom',
                          path='/home/yyz/code/yolov5/m1.pt',
                          source='local')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))[:, :, ::-1]
        if img is not None:
            images.append(img)
    return images



# def check(imgName,prelist):
#     # check if the helment and working cloth exist in the list.
#     helmentNumber = prelist.count(0.0)
#     workingcloNumber = prelist.count(1.0)
#     print(f'imgName={imgName},predictList={prelist}  helmentNumber={helmentNumber},workingcloNumber={workingcloNumber}')
#     if helmentNumber > 0:
#         print("helments at least one in the images")
#     else:
#         print("no helments")
#
#     if workingcloNumber > 0:
#         print("working clothes at least one in the images")
#     else:
#         print("no working clothes")


# #load original images
# imgPath = '/home/yyz/code/yolov5/data/images/bus.jpg'
# #originalData = load_images_from_folder(imgPath)
# originalData = imgPath
# # predict the person and save in the personPath
# personPath = 'runs'
# personModel.classes = [0]   #filter by classes, only predict person.
# personResults = personModel(originalData, size=640)
# personResults.crop(personPath)
#
# imgs = load_images_from_folder(os.path.join(personPath, 'crops/person'))
# results = ppeModel(imgs, size=640)
# print(results.xyxy[0].T[5].tolist())

# read and save each frame in the video, path is it's time
# import cv2
# import os
from models import common   # models is a dir and common.py

videoPath1 = '/run/sdd/Yun/2021/CV/10Ges/作业视频/20210515_163609.mp4'
videoPath = '/home/yyz/code/Yolov5-Deepsort/20210515_152115.mp4'
vidcap = cv2.VideoCapture(videoPath)
fps = vidcap.get(cv2.CAP_PROP_FPS)  # CAP_PROP_后接各种属性
count = 0
a = 0
if vidcap.isOpened():
    success, frame = vidcap.read()
    while success:
        count = count + 1
        times = count * (1 / fps)
        #print(f'count={count} time={times}')
        # can weite a function to pass count and fps and out the minites and seconds
        path = str(times)
        success, frame = vidcap.read()  # read a new farme
        personResults = personModel(frame[:, :, ::-1], size=640)  #cv is BGR, yolo is RGB
        personResults.crop(path)
        cropped_imgs = load_images_from_folder('runs/crops/'+path+'/person')
        results = ppeModel(cropped_imgs, size=160)
        results.display(ppe=True,out_name=path)
        #results = ppeModel(imgs, size=640)
    print('end of video')
else:
    print('视频打开失败')
    exit()
print('hello world')


