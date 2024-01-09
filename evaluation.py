import matplotlib.pyplot as plt 
import dv_processing as dv
import cv2 as cv
import numpy as np
import glob
import json
## Calculate OpenCV based method 

def counter(filename):
    reader = dv.io.MonoCameraRecording(filename)
    total_count = 0
    actual_frame_count = 0
    while reader.isRunning():
        frame = reader.getNextFrame()
        actual_frame_count+=1
        if frame is not None:
            img = frame.image
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            original_img = img.copy()
            _, mask = cv.threshold(img, 10, 255, cv.THRESH_BINARY)
            # Show a preview of the image
            mask = cv.bitwise_not(mask)
            result = cv.bitwise_and(img, img, mask=mask)
            kernel = np.ones((5, 5), np.uint8)
            result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
            cnts, hiers = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
            # print(len(cnts))
            for cnt in cnts:
                try:
                    ellipse = cv.fitEllipse(cnt)
                    total_count += 1
                    img = original_img.copy()
                except:
                    continue
    return total_count, actual_frame_count


count = 0
global_count_infer = 0
all_frames = 0
files = glob.glob('eye_dataset/dvS*.aedat4')
for file in files:
    count, frames = counter(file)
    global_count_infer += count
    all_frames += frames

global_count_real = 0

files = glob.glob("eye_dataset/gt_data/ellipse*.json")
for f in files:
    with open(f, 'r') as json_file:
        d = json.load(json_file)
        global_count_real += len(d)
print(f"Correct guess: {(global_count_real/all_frames)*100:.2f}%")