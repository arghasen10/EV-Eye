import json
import dv_processing as dv
import cv2 as cv

def get_ellipse_data_by_frame(frame_number, ellipse_data):
    for frame_ellipses in ellipse_data:
        for key, value in frame_ellipses.items():
            if int(key.split('_')[1]) == frame_number:
                return value
    return None


with open("eye_dataset/gt_data/ellipse_data_2024_01_07_21_35_28.json", 'r') as json_file:
    ellipse_data = json.load(json_file)

filename = "eye_dataset/dvSave-2024_01_07_21_35_28.aedat4"
hash = filename.split("/")[-1].split(".")[0].split("-")[-1]
reader = dv.io.MonoCameraRecording(filename)
frame_no = 0
while reader.isRunning():
    frame = reader.getNextFrame()
    frame_no+=1
    if frame is not None:
        img = frame.image
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        original_img = img.copy()
        ellipse = get_ellipse_data_by_frame(frame_no, ellipse_data)
        if ellipse is not None:
            cv.ellipse(img, get_ellipse_data_by_frame(frame_no, ellipse_data), (255,0, 255), 1, cv.LINE_AA)
            cv.imshow("Frame", img)
        else:
            print("No ellipse found for this frame")
            cv.imshow("Frame", img)
        key = cv.waitKey(0)
        img = original_img.copy()
        
cv.destroyAllWindows()
