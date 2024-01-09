import json
import cv2 as cv
import glob

def get_ellipse_data_by_frame(frame_number, ellipse_data):
    for frame_ellipses in ellipse_data:
        for key, value in frame_ellipses.items():
            if int(key.split('_')[1]) == frame_number:
                return value
    return None

filename = "eye_dataset/dvSave-2024_01_08_20_18_27.aedat4"
hash = filename.split("/")[-1].split(".")[0].split("-")[-1]
with open(f"eye_dataset/gt_data/ellipse_data_{hash}.json", 'r') as json_file:
    ellipse_data = json.load(json_file)

frames = glob.glob(f"eye_dataset/gt_data/saved_frame_{hash}_*.png")
for frame in frames:
    img = cv.imread(frame)
    ellipse = get_ellipse_data_by_frame(frame_number=int(frame.split("_")[-1].split(".")[0]), ellipse_data=ellipse_data)
    cv.ellipse(img, ellipse, (255,0, 255), 1, cv.LINE_AA)
    cv.imshow("Frame", img)
    key = cv.waitKey(0)
        
cv.destroyAllWindows()
