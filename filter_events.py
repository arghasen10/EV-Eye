import dv_processing as dv
import cv2 as cv
import numpy as np
import datetime

def euclidean_distance(point1, point2):
    return np.sqrt(np.square(point2[0] - point1[0]) + np.square(point2[1] - point1[1]))

filename = "eye_dataset/dvSave-2024_01_08_20_18_27.aedat4"
# Open any camera
reader = dv.io.MonoCameraRecording(filename)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
visualizer.setNegativeColor(dv.visualization.colors.darkGrey())

# Initialize Kalman filter parameters
kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03


# Initialize a preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
avg_coordinate = np.array([0, 0])

# Run the loop while the camera is still connected
frame_count = 0
start_time = None
flag = 0
kernel = np.ones((3, 3), np.uint8) 
total_detection = 0
while reader.isRunning():
    # Read batch of events
    events = reader.getNextEventBatch()

    if events is not None:
        if start_time is None:
            start_time = events[0].timestamp()
        event_count = len(events)
        if 17000 > event_count > 4000:
            # Print received packet time range
            frame = visualizer.generateImage(events)
            # Show the accumulated image
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            og_frame = frame.copy()
            og_frame2 = frame.copy()
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
                            frame_count+=1
                            if (i[2] > 80) or (i[2] < 35):
                                continue 
                            if i[0] > i[1]:
                                if i[0]/i[1] > 6:
                                    continue
                            elif i[1] > i[0]:
                                if i[1]/i[0] > 6:
                                    continue
                            if flag == 0:
                                avg_coordinate = i[:2]
                                kalman.statePre = np.array([i[0], i[1], 0, 0], np.float32)
                                kalman.statePost = np.array([i[0], i[1], 0, 0], np.float32)
                                flag=1
                            else:
                                # Calculate euclidean distance
                                distance = euclidean_distance(avg_coordinate, i[:2])
                                if distance < 20:
                                    # Apply Kalman filter prediction and correction
                                    prediction = kalman.predict()
                                    kalman.correct(np.array([[np.float32(i[0])], [np.float32(i[1])]]))

                                    # Get corrected coordinates from Kalman filter
                                    corrected_coordinates = np.array([kalman.statePost[0], kalman.statePost[1]])
                                    # Update average coordinate
                                    avg_coordinate = (avg_coordinate + i[:2]) / 2
                                    # draw the outer circle
                                    cv.circle(og_frame,(i[0],i[1]),i[2],(0,255,0),2)
                                    cv.circle(og_frame,(int(corrected_coordinates[0]), int(corrected_coordinates[1])),i[2],(255,0,0),2)
                                    # draw the center of the circle
                                    cv.circle(og_frame,(i[0],i[1]),2,(0,0,255),3)
                                    cv.imshow("Preview", og_frame)
                                    key = cv.waitKey(1)
                                    total_detection+=1
                            og_frame = og_frame2.copy() 
                    frame = original_img.copy()
        end_time = events[event_count-1].timestamp()
cv.destroyAllWindows()
time_taken = (end_time-start_time)/1000000
print("total_detection", total_detection)
print("Total Time", time_taken)
print("Start time", start_time)
print("End Time", end_time)
print("Throughput", (total_detection/time_taken))