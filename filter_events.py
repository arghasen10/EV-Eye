import dv_processing as dv
import cv2 as cv
from datetime import datetime, timedelta
import statistics
import numpy as np

def slicing_callback(events: dv.EventStore):
    # Generate a preview frame
    frame = visualizer.generateImage(events)

    # Show the accumulated image
    cv.imshow("Preview", frame)
    cv.waitKey(0)

filename = "eye_dataset/dvSave-2024_01_08_20_18_27.aedat4"
# Open any camera
reader = dv.io.MonoCameraRecording(filename)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
visualizer.setNegativeColor(dv.visualization.colors.darkGrey())

# Initialize a preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Initialize a slicer
slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(timedelta(milliseconds=204.7), slicing_callback)

# Run the loop while the camera is still connected
frame_count = 0
while reader.isRunning():
    # Read batch of events
    events = reader.getNextEventBatch()
    kernel = np.ones((3, 3), np.uint8) 

    if events is not None:
        event_count = len(events)
        if 17000 > event_count > 4000:
            # Print received packet time range
            frame = visualizer.generateImage(events)
            # Show the accumulated image
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.dilate(frame, kernel, iterations=1)
            edged = cv.Canny(frame, 10, 20)
            edged = cv.dilate(edged, kernel, iterations=1)
            original_img = frame.copy()
            contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
            for cnt in contours:
                if len(cnt) > 150 and cv.contourArea(cnt) > 500:
                    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                    circle_frame = frame.copy() 
                    circles = cv.HoughCircles(circle_frame, cv.HOUGH_GRADIENT,3,100, param1=20,param2=30,minRadius=0,maxRadius=0)
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0,:]:
                            if (i[2] > 80) or (i[2] < 35):
                                continue 
                            if i[0] > i[1]:
                                if i[0]/i[1] > 6:
                                    continue
                            elif i[1] > i[0]:
                                if i[1]/i[0] > 6:
                                    continue
                            # draw the outer circle
                            cv.circle(circle_frame,(i[0],i[1]),i[2],(0,255,0),2)
                            # draw the center of the circle
                            cv.circle(circle_frame,(i[0],i[1]),2,(0,0,255),3)
                            print("################### ", i)
                            
                            cv.imshow("Preview", circle_frame)
                            key = cv.waitKey(0)

                            
                            # cv.imshow("Preview", circle_frame)
                            key = cv.waitKey(0)
                            
                            if key == ord('c'):
                                with open("event_circles.txt", "a") as file:
                                    file.write("c:"+str(i)+",")
                            if key == ord('i'):
                                with open("event_circles.txt", "a") as file:
                                    file.write("i:"+str(i)+",")
                            circle_frame = frame.copy() 
                    frame = original_img.copy()
            # result = cv.bitwise_and(frame, frame, mask=mask)
            # kernel = np.ones((5, 5), np.uint8)
            # result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)

            # cv.imshow("Preview", frame)
            # cv.waitKey(1)

