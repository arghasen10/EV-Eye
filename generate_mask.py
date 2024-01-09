import json
import cv2 as cv
import glob
import numpy as np
import h5py
# import matplotlib.pyplot as plt

def get_ellipse_data_by_frame(frame_number, ellipse_data):
    for frame_ellipses in ellipse_data:
        for key, value in frame_ellipses.items():
            if int(key.split('_')[1]) == frame_number:
                return value
    return None


def generate_mask(filename): 
    hash = filename.split("/")[-1].split(".")[0].split("-")[-1]
    with open(f"eye_dataset/gt_data/ellipse_data_{hash}.json", 'r') as json_file:
        ellipse_data = json.load(json_file)

    frames = glob.glob(f"eye_dataset/gt_data/saved_frame_{hash}_*.png")
    with h5py.File(f"eye_dataset/gt_data/ellipse_masks_{hash}.h5", "w") as hf:
        data_grp = hf.create_group("data")
        label_grp = hf.create_group("label")
        for frame in frames:
            img = cv.imread(frame)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            mask = np.zeros((260,346), dtype=np.uint8)
            frame_number=int(frame.split("_")[-1].split(".")[0])
            ellipse = get_ellipse_data_by_frame(frame_number, ellipse_data=ellipse_data)
            cv.ellipse(mask, ellipse, color=1, thickness=-1)
            data_grp.create_dataset(f"{frame_number}", data=img)
            label_grp.create_dataset(f"{frame_number}",data=mask)


files = glob.glob('eye_dataset/dvS*.aedat4')
for file in files:
    generate_mask(file)





# with h5py.File(f"eye_dataset/gt_data/ellipse_masks_{hash}.h5", "r") as hf:
#     data_grp = hf["data"]
#     label_grp = hf["label"]
#     for frame_name in data_grp:
#         img_data = data_grp[frame_name][()]
#         mask_data = label_grp[frame_name][()]

#         # Display the image and mask
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv.cvtColor(img_data, cv.COLOR_BGR2RGB))
#         plt.title("Image")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.imshow(mask_data, cmap='gray')
#         plt.title("Mask")
#         plt.axis("off")

#         plt.show()