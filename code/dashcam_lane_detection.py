# this program detects straight lanes, classifies it into left line and right line,
# and finally colors them according to the type of line (dashed = red, solid = green)

import cv2
import numpy as np
import os


class LaneDetectorPipeline():
    def __init__(self, image, scale):
        self.image = image
        self.rows = image.shape[0]  # 540
        self.cols = image.shape[1]  # 960

        self.lane_image = self.image

        # test for flipped video
        # set to True to flip the video and see the results. (It works)
        self.flip = False

        self.scale = scale
        self.minLineLength = 60
        self.maxLineGap = 50

    def pre_processing(self, resize, gray, blur, edge):
        if resize:
            new_width = int(self.cols * self.scale)
            new_height = int(self.rows * self.scale)
            self.image = cv2.resize(self.image, (new_width, new_height))
            self.rows = self.image.shape[0]
            self.cols = self.image.shape[1]
            image_scaled = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        if self.flip:
            self.image = cv2.flip(self.image, 1)
            image_scaled = cv2.flip(image_scaled, 1)

        if gray:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if blur:
            self.image = cv2.GaussianBlur(self.image, (7, 7), 0)

        if edge:
            self.image = cv2.Canny(self.image, 50, 150)

        return self.image, image_scaled

    def crop_lane(self):
        # defining a region of interest
        # TODO: This should be parameterized to the image size
        region = np.array(
            [(10, self.rows), (370, self.rows), (240, 135), (150, 135)])

        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, pts=[region], color=(255, 255))
        self.lane_image = cv2.bitwise_and(self.image, mask)
        return self.lane_image

    def hough_lines(self):
        lines = cv2.HoughLinesP(self.lane_image, 2, np.pi/180, 40, np.array(
            []), minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        return lines

    def line_classifier(self, lines, image):
        left = []
        right = []
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                x1, y1, x2, y2 = l
                slope = (y2-y1)/(x2-x1)
                if slope < 0:
                    left.append(l)
                else:
                    right.append(l)
        left_sum = 0
        right_sum = 0
        for i in left:
            l = i
            x1, y1, x2, y2 = l
            distance = pow((pow((x2 - x1), 2) + pow((y2 - y1), 2)), 1 / 2)
            left_sum = left_sum+distance
        for i in right:
            l = i
            x1, y1, x2, y2 = l
            distance = pow((pow((x2 - x1), 2) + pow((y2 - y1), 2)), 1 / 2)
            right_sum = right_sum + distance

        # the sum of distance of lines detected in the left side will be less if it contains broken, i.e. dashed lines. So, we will color the dashed lines as red and the other line as green
        if left_sum < right_sum:
            for i in left:
                l = i
                x1, y1, x2, y2 = l
                cv2.line(image, (l[0], l[1]), (l[2], l[3]),
                         (255, 0, 0), 2, cv2.LINE_AA)
            for i in right:
                l = i
                x1, y1, x2, y2 = l
                cv2.line(image, (l[0], l[1]), (l[2], l[3]),
                         (0, 255, 0), 2, cv2.LINE_AA)
        elif left_sum > right_sum:
            for i in left:
                l = i
                x1, y1, x2, y2 = l
                cv2.line(image, (l[0], l[1]), (l[2], l[3]),
                         (0, 255, 0), 2, cv2.LINE_AA)
            for i in right:
                l = i
                x1, y1, x2, y2 = l
                cv2.line(image, (l[0], l[1]), (l[2], l[3]),
                         (255, 0, 0), 2, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# visualizer class is capable of productin 1-5 output windows scaled proportionally to the input sample image
# it also has the capability to save the generated frames asa a video if used inside a loop
class Visualizer():
    def __init__(self, image, scale, frames=[None, None, None, None, 'Final Frame'], save=False):
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.frame4 = None
        self.frame0 = None

        self.save = save
        self.frame_dict = {frames[0]: self.frame1, frames[1]: self.frame2,
                           frames[2]: self.frame3, frames[3]: self.frame4, frames[4]: self.frame0}
        self.scale = scale

        self.image = self.resize(image, self.scale)
        self.frame_height = self.image.shape[0]
        self.frame_width = self.image.shape[1]

        self.number_of_frames = 1+(4-frames.count(None))
        self.number_of_small_frames = self.number_of_frames - 1
        self.aspect_ratio = self.frame_width/self.frame_height

        self.small_frame_width = int(
            self.frame_width/self.number_of_small_frames)
        self.small_frame_height = int(
            (self.number_of_small_frames * self.small_frame_width)/self.aspect_ratio)

        self.capture_width = self.frame_width * self.number_of_small_frames
        self.capture_height = self.frame_height * \
            self.number_of_small_frames + self.frame_height

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 255, 255)
        self.thickness = 2

        self.fps = 25
        if self.save:
            save_path = Visualizer.get_absolute_path(
                "output") + "/lane_detection_output.mp4"
            self.result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
                *'mp4v'), self.fps, (self.capture_width, self.capture_height))

    @staticmethod
    def get_absolute_path(dir):
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        absolute_path = os.path.join(parent_directory, dir)
        return absolute_path

    def resize(self, image, scale):
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))
        return image

    def update_frame(self, image, frame_name):
        if frame_name in self.frame_dict.keys():
            self.frame_dict[frame_name] = image
            self.frame_dict[frame_name] = cv2.cvtColor(
                self.frame_dict[frame_name], cv2.COLOR_BGR2RGB)
            # tune these two values for text alignment
            vertical_placement = 0.95
            horizontal_placement = 0.8
            org2 = (int(self.frame_width-vertical_placement*self.frame_width),
                    int(self.frame_height-horizontal_placement*self.frame_height))
            self.frame_dict[frame_name] = cv2.putText(
                self.frame_dict[frame_name], frame_name, org2, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

    def update_final_frame(self, image, frame_name):
        image = cv2.resize(image, (self.frame_width * self.number_of_small_frames,
                           self.frame_height * self.number_of_small_frames))

        self.frame_dict[frame_name] = image

        vertical_placement = 0.95
        horizontal_placement = 0.95
        org1 = (int(self.frame_width-vertical_placement*self.frame_width),
                int(self.capture_height-horizontal_placement * self.capture_height))
        self.frame_dict[frame_name] = cv2.putText(
            self.frame_dict[frame_name], frame_name, org1, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

        small_frames = []
        for small_frame_name in list(self.frame_dict.keys())[:-1]:
            if small_frame_name != None:
                small_frames.append(self.frame_dict[small_frame_name])

        bottom_frames = np.concatenate(small_frames, axis=1)

        output = np.concatenate(
            (self.frame_dict[frame_name], bottom_frames), axis=0)
        if self.save:
            self.result.write(output)

        output = cv2.resize(
            output, (int(output.shape[1]*0.6), int(output.shape[0]*0.6)))
        cv2.imshow(frame_name + " Pipeline", output)

    def __del__(self):
        print("Closing output feed...")


def main():
    scale = 0.4  # scaling down all frames for faster processing
    video_path = Visualizer.get_absolute_path("data") + "/whiteline.mp4"
    print(video_path)
    save = True
    frame = cv2.VideoCapture(video_path)
    _, sample_frame = frame.read()
    frames = ['Input', 'Edges', 'Lane Crop',
              'Hough Lines', 'Lane Segmentation']
    visuals = Visualizer(image=sample_frame, scale=scale,
                         frames=frames, save=save)

    while (frame.isOpened()):
        success, image = frame.read()
        if not success:
            print("\nEnd of frames\n")
            break

        current_frame_object = LaneDetectorPipeline(image, scale)

        image_edge, image_scaled = current_frame_object.pre_processing(
            resize=True, gray=True, blur=True, edge=True)
        visuals.update_frame(image_scaled, 'Input')
        visuals.update_frame(image_edge, 'Edges')

        image_cropped = current_frame_object.crop_lane()
        visuals.update_frame(image_cropped, 'Lane Crop')

        # using hough transform to detect lines
        hough_lines = current_frame_object.hough_lines()
        image_hough_lines = image_scaled
        for line in hough_lines:
            line = line[0]
            cv2.line(image_hough_lines, (line[0], line[1]),
                     (line[2], line[3]), (255, 100, 255), 1, cv2.LINE_AA)
        visuals.update_frame(image_hough_lines, 'Hough Lines')

        # classifying lines as left and right and coloring them acconding to the given convention
        image_lines_classified = current_frame_object.line_classifier(
            hough_lines, image_scaled)
        visuals.update_final_frame(image_lines_classified, 'Lane Segmentation')

        del current_frame_object

        if cv2.waitKey(25) & 0xFF == ord('q'):
            # del visuals
            break

    frame.release()
    cv2.destroyAllWindows()
    del visuals


if __name__ == "__main__":
    main()