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

