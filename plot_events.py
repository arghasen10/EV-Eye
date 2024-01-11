import dv_processing as dv
import cv2 as cv
from datetime import datetime, timedelta
import statistics


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
    
    if events is not None:
        # Print received packet time range
        frame = visualizer.generateImage(events)
        # Show the accumulated image
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        cv.imshow("Preview", frame)
        cv.waitKey(1)

