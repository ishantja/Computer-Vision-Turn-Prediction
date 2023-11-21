import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
 
frame_width = 960
frame_height = 540

number_of_frames = 5
number_of_small_frames = 4
aspect_ratio = frame_width / frame_height

small_frame_width = int(frame_width / number_of_small_frames)
small_frame_height = int(small_frame_width / aspect_ratio)

small_frames = []
for i in range(number_of_frames-1):
    temp_frame = np.ones((small_frame_height, small_frame_width, 3), dtype=np.uint8)
    temp_frame[:, :, :] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)] 
    small_frames.append(temp_frame)

frame0 = np.ones((frame_height, frame_width, 3), dtype=np.uint8)  # Create a white frame
frame0[:, :, :] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)] 
 
bottom_frames = np.concatenate(small_frames, axis=1)
bottom_frames = cv2.resize(bottom_frames, (frame_width, small_frame_height))
# print(bottom_frames.shape)

output = np.concatenate((frame0, bottom_frames), axis = 0)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)
# ax3 = plt.subplot2grid(shape=(2, 5), loc=(1, 1), colspan=1)
# ax4 = plt.subplot2grid(shape=(2, 5), loc=(1, 2), colspan=1)
# ax5 = plt.subplot2grid(shape=(2, 5), loc=(1, 3), colspan=1)

# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1)
 
# creating window size to display images
fig = plt.gcf()
fig_width, fig_height = fig.get_size_inches()
dpi = fig.get_dpi()

screen_width_in_pixels = fig_width * dpi
screen_height_in_pixels = fig_height * dpi
screen_coverage = 1

inch_width = fig_width * 2
inch_height = inch_width/ aspect_ratio

fig.set_size_inches(inch_width,inch_height)
# fig.set_figheight()


# print(f"screen w: {screen_width_in_pixels}, screen h: {screen_height_in_pixels}")
 
# plotting subplots
ax1.imshow(frame0)
ax1.set_title('Final Frame', y=-0.01)
ax1.set_xticks([])
ax1.set_yticks([])

ax2.imshow(bottom_frames)
ax2.set_title('ax2', y=-0.01)
ax2.set_xticks([])
ax2.set_yticks([])

# ax3.imshow(small_frames[1])
# ax3.set_title('ax3', y=-0.01)
# ax3.set_xticks([])
# ax3.set_yticks([])

# ax4.imshow(small_frames[2])
# ax4.set_title('ax4', y=-0.01)
# ax4.set_xticks([])
# ax4.set_yticks([])

# ax5.imshow(small_frames[3])
# ax5.set_title('ax5', y=-0.01)
# ax5.set_xticks([])
# ax5.set_yticks([])
 
# automatically adjust padding horizontally 
# as well as vertically.
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
 
# display plot
plt.show(block=True)
plt.pause(1)
plt.close()
