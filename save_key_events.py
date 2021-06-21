# USAGE
# python save_key_events.py --output output
# This is a usefule data from pyimagesearch
# https://www.pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/

# import the necessary packages
from pyimagesearch.keyclipwriter import KeyClipWriter
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32,
	help="buffer size of video clip writer")
args = vars(ap.parse_args())
# args={"output":"output","picamera":-1,"fps":20,"codec":"MJPG","buffer_size":32}

# initialize the video stream and allow the camera sensor to
# warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# define the lower and upper boundaries of the "green" ball in
# the HSV color space
# greenLower = (29, 43, 46)
# greenUpper = (64, 255, 255)
orangeLower = (11, 43, 46)    # 颜色空间的上下阈值
orangeUpper = (25, 255, 255)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0    # 未包含任何关键事件的连续帧的数值

# keep looping
while True:
	# grab the current frame, resize it, and initialize a
	# boolean used to indicate if the consecutive frames
	# counter should be updated
	frame = vs.read()  # 读入下一个帧
	frame = imutils.resize(frame, width=600)   # 调整大小到600
	updateConsecFrames = True

	# update the key frame clip buffer
	kcw.update(frame)

	# blur the frame and convert it to the HSV color space
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)  # 高斯模糊
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)   # 调整颜色空间从RGB到HSV，可以判断颜色阈值
	# print(hsv.min(axis=(0,1)))
	# print(hsv.max(axis=(0,1)))

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it
		# to compute the minimum enclosing circle
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		updateConsecFrames = radius <= 10

		# only proceed if the redius meets a minimum size
		if radius > 10:
			# reset the number of consecutive frames with
			# *no* action to zero and draw the circle
			# surrounding the object
			consecFrames = 0
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 0, 255), 2)

			# if we are not already recording, start recording
			if not kcw.recording:
				timestamp = datetime.datetime.now()
				p = "{}/{}.mp4".format(args["output"],
					timestamp.strftime("%Y%m%d-%H%M%S"))  # 要保存的文件名
				kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
					args["fps"])

	# otherwise, no action has taken place in this frame, so
	# increment the number of consecutive frames that contain
	# no action
	if updateConsecFrames:
		consecFrames += 1

	# update the key frame clip buffer
	kcw.update(frame)

	# if we are recording and reached a threshold on consecutive
	# number of frames with no action, stop recording the clip
	# 这个函数规定了在关键事件后连续出现多少个帧的没有关键事件，再停止写入
	# 结束这个写入进程，下一次写入就是一个新的文件了
	if kcw.recording and consecFrames == args["buffer_size"]:
		kcw.finish()

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are in the middle of recording a clip, wrap it up
# 已经退出了循环，但是还是recording状态时，释放线程
if kcw.recording:
	kcw.finish()

# do a bit of cleanup
# 释放窗口，结束读取摄像头
cv2.destroyAllWindows()
vs.stop()