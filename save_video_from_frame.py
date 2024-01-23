import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
video = cv2.VideoWriter('video.avi', 0, 100, (346,260))

for j in range(1,2180):
    filename = f"images/frames_{str(j)}.png"
    img = cv2.imread(filename)
    if img is None:
        continue
    video.write(img)

cv2.destroyAllWindows()
video.release()