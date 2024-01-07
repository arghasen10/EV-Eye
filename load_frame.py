import dv_processing as dv
import cv2 as cv
import numpy as np
import json

def meets_criteria(contour, area_threshold_min, area_threshold_max, axis_ratio_threshold_min, axis_ratio_threshold_max):
    try:
        x_min, x_max, y_min, y_max, width_min, width_max, height_min, height_max = 111, 200, 59, 117, 6, 38, 7, 47
        ellipse = cv.fitEllipse(contour)
        (a, b) = ellipse[1]
        x, y, w, h = cv.boundingRect(contour)
        axis_ratio = max(a, b) / min(a, b)
        area = cv.contourArea(contour)
        if (x_min < x < x_max) and (y_min < y < y_max) and (width_min < w < width_max) and (height_min < h < height_max) and (area_threshold_min < area < area_threshold_max) and (axis_ratio_threshold_min < axis_ratio < axis_ratio_threshold_max):
            return True
        else:
            return False
        # return (area_threshold_min < area < area_threshold_max) and (axis_ratio_threshold_min < axis_ratio < axis_ratio_threshold_max)
    except:
        return False


# Open a file
reader = dv.io.MonoCameraRecording("eye_dataset/dvSave-2024_01_07_21_33_55.aedat4")

flag = False


# Variable to store the previous frame timestamp for correct playback
lastTimestamp = None
frame_no = 0

# Initialize variables for Kalman filter
last_measurement = None
last_prediction = None



# Run the loop while camera is still connected
while reader.isRunning():
    # if flag == False:
    #     flag = True
    #     fourcc = cv.VideoWriter_fourcc(*'h264')
    #     output_video = cv.VideoWriter('output_video.mp4', fourcc, 30.0, (reader.getEventResolution()[0], reader.getEventResolution()[1]))  # Replace width and height with appropriate values

    # Read a frame from the camera
    frame = reader.getNextFrame()
    
    if frame is not None:
        # Print the timestamp of the received frame
        # print(f"Received a frame at time [{frame.timestamp}]")
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
            if meets_criteria(cnt, 50, 821.5, 1.045, 2.131):
                ellipse = cv.fitEllipse(cnt)
                cv.ellipse(img, ellipse, (255,0, 255), 1, cv.LINE_AA)
                (x, y), (w, h), _ = ellipse
                # cv.imshow("Frame", img)
                # cv.waitKey(0)
                # img = original_img.copy()
        
        delay = (2 if lastTimestamp is None else (frame.timestamp - lastTimestamp) / 1000)
        # output_video.write(img)
        cv.imshow("Frame", img)
        # Perform the sleep
        cv.waitKey(int(delay))

        # Store timestamp for the next frame
        lastTimestamp = frame.timestamp

# output_video.release()