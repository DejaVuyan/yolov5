###########################################
# 检测出有人的部分，截取出视频
# load model

import torch
import cv2
from pyimagesearch.keyclipwriter import KeyClipWriter

personModel = torch.hub.load('D:\MyWrokspace\code\others\yolov5',
                             'custom',
                             path='D:\MyWrokspace\code\others\yolov5\weights\yolov5s.pt',
                             source='local')
personModel.classes = [0]  # filter by classes, only predict person.

save_path = r'E:\dataset\test'
video_path = r'E:\dataset\Yundian\10Ges\作业视频\安全装备\20210515_161247.mp4'
video_test_path = r'E:\dataset\test\source_video\gongdi.mp4'

vidcap = cv2.VideoCapture(video_test_path)

fps = vidcap.get(cv2.CAP_PROP_FPS)  # CAP_PROP_后接各种属性
w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# vid_writer = cv2.VideoWriter(save_path+'\\test3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# bufsize 是要延迟结束录制的时间*fps 设为5S，就是5*fps  int(5 * fps)
bufsize = 32
kcw = KeyClipWriter(bufSize=bufsize)

count = 1
consecFrames = 0

if vidcap.isOpened():
    video_time = (1 / fps) * count
    success, frame = vidcap.read()
    while success:
        times = count * (1 / fps)
        personResults = personModel(frame[:, :, ::-1], size=640)  # cv is BGR, yolo is RGB
        if (len(personResults.pred[0])):
            consecFrames = 0
            # if we are not already recording, start recording
            if not kcw.recording:
                p = "{}/{}_{}.mp4".format(save_path,
                                          'gongdi', times)  # 要保存的文件名
                kcw.start(p, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, w, h)

            print(f'save:{count}/{length}')
        else:
            print(f'not save:{count}/{length}')
            consecFrames += 1
        kcw.update(frame)
        if kcw.recording and consecFrames == bufsize:
            kcw.finish()
        count = count + 1
        success, frame = vidcap.read()  # read a new farme

    if kcw.recording:
        kcw.finish()

    vidcap.release()
    print('剪辑成功！')
else:
    print('视频打开失败')
    exit()
