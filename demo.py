import torch
import cv2
from PIL import Image

#这是Pytorch写的一个库，用来从github上加载模型，pytorch加载模型要先创建一个相同类型的模型变量，就可以利用这个在最新的github上加载
#模型

# Model
#第一个参数必须是yolov5的github名字，path是自己的模型名字，force_reload会将.cache中的文件删掉，去下载最新版的yolov5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/yyz/code/yolov5/m1.pt')
#model是一个model/common.py中AutoShape的类

#从本地仓库中加载模型，需要加上source=local,比如
model_local = torch.hub.load('/home/yyz/code/yolov5', 'custom', path='/home/yyz/code/yolov5/weights/yolov5x.pt', source='local')

# Image
testImg = '/run/sdd/Yun/2021/CV/YunDataset/test/images/00000007.jpg'
img1 = Image.open(testImg)
results = model(img1,size=640)   #传入图像，自动调用AutoShape.forward,返回一个Detection的类，具体可以传入的格式看原函数
results.print()
results.save()  #把预测出来带框的图片保存在 runs/hub/exp，可以修改。
results.crop('runs') #会自动在给定路径下生成crops/person文件夹，有多少类生成多少类

#预测结果
results.xyxy[0]  # img1 predictions (tensor)
#在控制台可以看到，这是一个list,里面有若干个tensor作为预测结果，results.xyxy[0]就是第一个tensor
#Tensor是(5,6)的大小,5代表检测到了5个目标，6中的最后一个是class
results.pandas().xyxy[0]  # img1 predictions (pandas)

results.xyxy[0].T[5].tolist() #将预测的类别提取出来，返回一个列表
torch.gather()  #这个函数可以从tensor中自己按规律取数来形成一个新的tensor.

model_person = torch.hub.load('ultralytics/yolov5',
                       'custom',
                       path='/home/yyz/code/yolov5/weights/yolov5x/pt',
                       force_reload=True
                       )

torch.tensor()

import cv2
videoPath = '/home/yyz/code/Yolov5-Deepsort/20210515_152115.mp4'
vidcap = cv2.VideoCapture(videoPath)
if vidcap.isOpened():
    while True:
        success, image = vidcap.read()
        if not success:break   #if in the end of video
        # funciong: processing image

    else:
        print('视频打开失败')