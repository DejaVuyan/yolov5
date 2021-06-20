import torch
import cv2
from PIL import Image

#这是Pytorch写的一个库，用来从github上加载模型，pytorch加载模型要先创建一个相同类型的模型变量，就可以利用这个在最新的github上加载
#模型

# Model
#第一个参数必须是yolov5的github名字，path是自己的模型名字，force_reload会将.cache中的文件删掉，去下载最新版的yolov5
from cv2 import CAP_PROP_POS_MSEC

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

#read and save each frame in the video, path is it's time
import cv2
import os
videoPath = '/home/yyz/code/Yolov5-Deepsort/20210515_152115.mp4'
vidcap = cv2.VideoCapture(videoPath)
fps = vidcap.get(cv2.CAP_PROP_FPS)  #CAP_PROP_后接各种属性
count = 0
if vidcap.isOpened():
    success, frame = vidcap.read()
    while success:
        count = count + 1
        times = count*(1/fps)
        print(f'count={count} time={times}')
        #can weite a function to pass count and fps and out the minites and seconds
        path = str(times)
        success, frame = vidcap.read()  #read a new farme
        personResults = personModel(frame, size=640)
        personResults.crop(path)

    print('end of video')
else:
    print('视频打开失败')
    exit()
print('hello world')

# detect ppe in a dir
from models import common   # models is a dir and common.py
import os
import cv2
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

import torch
ppeModel = torch.hub.load('/home/yyz/code/yolov5',
                          'custom',
                          path='/home/yyz/code/yolov5/m1.pt',
                          source='local')
testpath = load_images_from_folder('/home/yyz/code/yolov5/runs/crops/0.10664639055371679/person/')
results = ppeModel(testpath, size=640)
results.display(ppe=True)



###########################################
#检测出有人的部分，通过检测Tensor是否为0来判断
# load model
import torch
import cv2
from PIL import Image
import os

personModel = torch.hub.load('D:\MyWrokspace\code\others\yolov5',
                             'custom',
                             path='D:\MyWrokspace\code\others\yolov5\weights\yolov5s.pt',
                             source='local')
personModel.classes = [0]   #filter by classes, only predict person.

save_path = r'E:\dataset\test'
video_path = r'E:\dataset\Yundian\10Ges\作业视频\安全装备\20210515_161247.mp4'

vidcap = cv2.VideoCapture(video_path)


fps = vidcap.get(cv2.CAP_PROP_FPS)  # CAP_PROP_后接各种属性
w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_writer = cv2.VideoWriter(save_path+'\\test2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
img1 = cv2.imread(r'1.png')[:, :, ::-1]
img2 = cv2.imread(r'7576.jpg')[:, :, ::-1]
personResults1 = personModel(img1, size=640)
personResults2 = personModel(img2, size=640)

print(personResults1.pred)
print(personResults2.pred)

if len(personResults2.pred[0]) == 0:
	print("this tensor is empty")
else:
	print("this tensor is not empty")



