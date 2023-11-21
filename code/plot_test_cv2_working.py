import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 255, 255)
thickness = 1

frame_width = 960
frame_height = 540

number_of_frames = 5
number_of_small_frames = 4
aspect_ratio = frame_width / frame_height

small_frame_width = int(frame_width / number_of_small_frames)
small_frame_height = int(small_frame_width / aspect_ratio)

result = cv2.VideoWriter('test_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width,frame_height+small_frame_height))

for i in range(100):
    frame0 = np.ones((frame_height, frame_width, 3),
                     dtype=np.uint8)  # Create a white frame

    frame0[:, :, :] = [random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)]
    
    org1= (int(frame_width-.95*frame_width),int(frame_height-.05*frame_height))
    frame0 = cv2.putText(frame0, 'OpenCV', org1, font, fontScale, color, thickness, cv2.LINE_AA)

    small_frames = []
    for i in range(number_of_frames-1):
        temp_frame = np.ones(
            (small_frame_height, small_frame_width, 3), dtype=np.uint8)
        temp_frame[:, :, :] = [random.randint(
            0, 255), random.randint(0, 255), random.randint(0, 255)]
        org2= (int(small_frame_width-.95*small_frame_width),int(small_frame_height-.1*small_frame_height))
        temp_frame = cv2.putText(temp_frame, 'OpenCV', org2, font, fontScale, color, thickness, cv2.LINE_AA)
        small_frames.append(temp_frame)

    bottom_frames = np.concatenate(small_frames, axis=1)

    output = np.concatenate((frame0, bottom_frames), axis=0)
    result.write(output)
    cv2.imshow("preview", output)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
