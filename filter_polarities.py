import dv_processing as dv
import cv2 as cv
import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.square(point2[0] - point1[0]) + np.square(point2[1] - point1[1]))

def are_circles_close(circle1, circle2, distance_threshold=10):
    circle2, circle1 = np.array(circle2), np.array(circle1)
    distance = euclidean_distance(circle1[:2], circle2[:2])
    return distance < distance_threshold


def show_circle(frame):
    kernel = np.ones((3, 3), np.uint8) 
    frame = cv.dilate(frame, kernel, iterations=1)
    edged = cv.Canny(frame, 10, 20)
    edged = cv.dilate(edged, kernel, iterations=1)
    circles_list = []
    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    for cnt in contours:
        if len(cnt) > 150 and cv.contourArea(cnt) > 500:
            cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT,3,100, param1=20,param2=30,minRadius=0,maxRadius=0)
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
                    circles_list.append(i)
    return circles_list

filename = "eye_dataset/dvSave-2024_01_08_20_18_27.aedat4"
# Open any camera
reader = dv.io.MonoCameraRecording(filename)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
visualizer.setNegativeColor(dv.visualization.colors.darkGrey())

# Initialize a preview window
# cv.namedWindow("Preview", cv.WINDOW_NORMAL)
result = cv.VideoWriter('filename2.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (260,346)) 
frame_count = 0
# Run the loop while the camera is still connected
while reader.isRunning():
    # Read batch of events
    events = reader.getNextEventBatch()
    
    if events is not None:
        frame = visualizer.generateImage(events) 
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        positive_events = [event for event in events if event.polarity() == True]
        img = np.ones(reader.getEventResolution(), dtype=np.uint8)*255
        for p in positive_events:
            img[p.x(),p.y()] = 0
        img = cv.rotate(img, cv.ROTATE_180)
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.flip(img, 1)
        p_circles_list = show_circle(img)

        negative_events = [event for event in events if event.polarity() == False]
        img = np.ones(reader.getEventResolution(), dtype=np.uint8)*255
        for p in negative_events:
            img[p.x(),p.y()] = 0
        img = cv.rotate(img, cv.ROTATE_180)
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.flip(img, 1)
        og_img = frame.copy()
        n_circles_list = show_circle(img)
        if len(p_circles_list) != 0 and len(n_circles_list) != 0:
            for p_circle in p_circles_list:
                for n_circle in n_circles_list:
                    if are_circles_close(p_circle, n_circle, distance_threshold=50):
                        frame_count+=1
                        print("Circles are close:", p_circle, n_circle)
                        cv.circle(frame, p_circle[:2], p_circle[2], (0,255,0),2)
                        cv.circle(frame, n_circle[:2], n_circle[2], (0,0,255),2)
                        cv.circle(frame, (int((p_circle[0]+n_circle[0])/2), int((p_circle[1]+n_circle[1])/2)), 4, (255,0,0), 4)
                        result.write(frame) 
                        cv.imshow("circles", frame)
                        filename = f"images/{frame_count}.png"
                        cv.imwrite(filename, frame)
                        cv.waitKey(1)
                        frame = og_img.copy()

result.release() 
cv.destroyAllWindows() 
