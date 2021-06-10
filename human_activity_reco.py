import numpy as np
import argparse
import imutils
import sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="",
	help="optional path to video file")
args = vars(ap.parse_args())

CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

while True:

    frames = []

# loop number of required sample frames
for i in range(0, SAMPLE_DURATION):
		# read frame from stream
		(grabbed, frame) = vs.read()

        # if frame not grabbed, end of video, exit
if not grabbed:
			print("[INFO] no frame read from stream - exiting")
			sys.exit(0)

frame = imutils.resize(frame, width=400)
frames.append(frame)

        # now that our frames array is filled we can construct our blob(Binary Large Object)
blob = cv2.dnn.blobFromImages(frames, 1.0,
		(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
		swapRB=True, crop=True)

blob = np.transpose(blob, (1, 0, 2, 3))
blob = np.expand_dims(blob, axis=0)