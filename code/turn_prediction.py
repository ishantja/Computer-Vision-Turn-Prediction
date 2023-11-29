# this script detects curved lanes, calculates its radius of curvature, and tells whether it is a right or a left turn
# also has provisions to handle failure cases

import cv2
import numpy as np
import os

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

        self.number_of_small_frames = 4
        self.aspect_ratio = self.frame_width/self.frame_height

        self.small_frame_width = int(
            self.frame_width/self.number_of_small_frames)
        self.small_frame_height = 350

        self.capture_width = self.frame_width
        self.capture_height = self.frame_height + self.small_frame_height

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 255, 255)
        self.thickness = 2

        self.fps = 25
        if self.save:
            save_path = Visualizer.get_absolute_path(
                "output") + "/turn_prediction_output.mp4"
            self.result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
                *'mp4v'), self.fps, (self.capture_width, self.capture_height))

    @staticmethod
    def show_image(image):
        cv2.imshow("preview", image)

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
            image = cv2.resize(image, (self.small_frame_width, self.small_frame_height))
            self.frame_dict[frame_name] = image
            self.frame_dict[frame_name] = cv2.cvtColor(
                self.frame_dict[frame_name], cv2.COLOR_BGR2RGB)
            # tune these two values for text alignment
            vertical_placement = 0.93
            horizontal_placement = 0.55
            org2 = (int(self.frame_width-vertical_placement*self.frame_width),
                    int(self.frame_height-horizontal_placement*self.frame_height))
            self.frame_dict[frame_name] = cv2.putText(
                self.frame_dict[frame_name], frame_name, org2, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

    def update_final_frame(self, image, frame_name):
        self.frame_dict[frame_name] = image

        vertical_placement = 0.92
        horizontal_placement = 0.88
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

class TurnPrediciton():
    def __init__(self, image, scale):
        self.image = image
        self.rows = image.shape[0]
        self.cols = image.shape[1]

        self.lane_image = self.image
        self.flip = False

        self.scale = scale

        self.left_poly = None
        self.right_poly = None

    def pre_processing(self, resize, gray, blur):
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
        image_scaled = self.image

        return self.image
    
    def crop_lane(self,gray):
        # defining region of interest so that maximum length of the lane can be detected
        # TODO: come up with better way to crop using image shape parameters
        region = np.array([(230, self.rows - 60), (1120, self.rows - 60), (730, 440), (610, 440)])

        # cropping only the region of interest for lane detection
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, pts=[region], color=(255, 255))
        self.lane_image = cv2.bitwise_and(gray, mask)
        return self.lane_image, region

    def warp_lane(self, cropped, region):
        # warping the cropped lane into a flat plane using homography
        # TODO: flat plane size should be proportional to the lane cropped 
        flat_plane = np.array([(0, 0), (200, 0), (200, 500), (0, 500)], dtype=float)
        h, status = cv2.findHomography(region, flat_plane)
        h_inv, status_inv = cv2.findHomography(flat_plane,region)
        warped = cv2.warpPerspective(cropped, h, (200,500))
        warped = cv2.flip(warped,0)
        dst = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        return warped, dst, h_inv

    def detect_edges(self, warped):
        # detecting edges on the warped lane using canny edge detection
        edge = cv2.Canny(warped, 50, 450)  # 50,350
        return edge

    def hough_lines(self,edge):
        # finding lines on the detected edges using hough transform. parameters have been tuned such that atleast 4 lines will be generated on both the left and the right curves of the lane
        linesP = cv2.HoughLinesP(edge, 2, np.pi / 180, 25, np.array([]), minLineLength=10, maxLineGap=150) # image, rho, theta, threshold = min # of intersections for line, lines, min, max
        return linesP

    # TODO: only find out which one is the left line and which one is the rigth line
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

    def lines_to_points(self,linesP, points):
        # segregating left lines and right lines based on their location in the warped image
        # TODO: coloring left lane = red and right lane = green 
        left = []
        right = []

        points = cv2.cvtColor(points, cv2.COLOR_GRAY2BGR)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                if l[0] < 100 and l[2] < 100:
                    left.append(l)
                    # cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    right.append(l)
                    # cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)

        # segregating into points falling on the left or right curves and visualizing the points
        all_points_left = []
        for i in left:
            all_points_left.append((i[0],i[1]))
            all_points_left.append((i[2], i[3]))

        for i in all_points_left:
            cv2.circle(points, i, 5, (255,255,0), 2)

        all_points_right = []
        for i in right:
            all_points_right.append((i[0],i[1]))
            all_points_right.append((i[2], i[3]))

        for i in all_points_right:
            cv2.circle(points, i, 5, (255,255,0), 2)

        return all_points_left, all_points_right, points

    def fit_poly(self, all_points_left, all_points_right,dst,h_inv, image, left_fit_coefs, right_fit_coefs, x_r):
        # checking if detection can be satisfactorily performed
        # TODO: satisfactory condition should be to check for shadows or unreliable points detection/ breaks in point detection
        count = 0
        for i in all_points_left:
            x = i[0]
            y = i[1]

            if x > 0 and x < 50 and y > 50 and y < 200:
                count = count + 1

        # start polynomial fitting only if detection is possible
        # TODO: make code robust to flipped frames/ lanes
        if count > 0:
            # fit a polynomial
            print("Lane detected.")
            left = np.array(all_points_left)
            for i in range(len(left)):
                x_l = left[:, 0]
                y_l = left[:, 1]
            right = np.array(all_points_right)
            for i in range(len(right)):
                x_r = right[:, 0]
                y_r = right[:, 1]

            left_fit_coefs = np.polyfit(y_l, x_l, 2)
            right_fit_coefs = np.polyfit(y_r, x_r, 2)

            y_values = np.linspace(0, dst.shape[0] - 1, dst.shape[0])
            x_values_left = left_fit_coefs[0] * y_values ** 2 + left_fit_coefs[1] * y_values + left_fit_coefs[2]
            x_values_right = right_fit_coefs[0] * y_values ** 2 + right_fit_coefs[1] * y_values + right_fit_coefs[2]

            left_polynomial = np.int_(np.vstack((x_values_left, y_values)).T)
            right_polynomial = np.int_(np.vstack((x_values_right, y_values)).T)

            cv2.polylines(dst, [left_polynomial], False, (0, 255, 0),5)
            cv2.polylines(dst, [right_polynomial], False, (0, 0, 255),5)
        else:
            # else, using previously calculated coefficients
            print("Lane detection failed! Resorting to previous best lane.")
            y_values = np.linspace(0, dst.shape[0] - 1, dst.shape[0])
            x_values_left = left_fit_coefs[0] * y_values ** 2 + left_fit_coefs[1] * y_values + left_fit_coefs[2]
            x_values_right = right_fit_coefs[0] * y_values ** 2 + right_fit_coefs[1] * y_values + right_fit_coefs[2]

            left_polynomial = np.int_(np.vstack((x_values_left, y_values)).T)
            right_polynomial = np.int_(np.vstack((x_values_right, y_values)).T)

            cv2.polylines(dst, [left_polynomial], False, (0, 255, 0),5)
            cv2.polylines(dst, [right_polynomial], False, (0, 0, 255),5)

        # finding radius of curvature
        left_curvature = ((1+(2*left_fit_coefs[0]*x_r[0]+left_fit_coefs[1])**2)**1.5)/np.absolute(2*left_fit_coefs[0])
        right_curvature = ((1+(2*right_fit_coefs[0]*x_r[0]+right_fit_coefs[1])**2)**1.5)/np.absolute(2*right_fit_coefs[0])
        road_curvature = (left_curvature + right_curvature) / 2
        if road_curvature > 3000:
            road_curvature = 2697.6605746337345
        info1 = ("Radius of Curvature: "+str(round(road_curvature, 2))+" m ")
        print(info1)

        # detecting left/right turn
        # if derivative is negative then it is a right turn
        slope = 2 * right_fit_coefs[0]*x_r[0]+right_fit_coefs[1]
        if slope < 0:
            info2 = ("Turn: Right")
            print(info2)
            print("\n")
        else:
            info2 = ("Turn: Left")
            print(info2)
            print("\n")

        # unwarping the dst image to get the final image
        # TODO: use visualizer class to show all intermediate frames
        final = cv2.flip(dst,0)
        final = cv2.warpPerspective(final, h_inv, (1280, 720))

        # superimposing detected lane onto the original image
        final = cv2.add(image, final)

        # overlaying textual information
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (200, 255, 50)
        thickness = 2
        final = cv2.putText(final, info1 , (100,50), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        final = cv2.putText(final, info2 , (100,80), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        
        return final, left_fit_coefs, right_fit_coefs, x_r
    
def main():
    # TODO: write turn prediction class with all the preprocessing 
    video_path = Visualizer.get_absolute_path(
                "data") + "/challenge.mp4"
    frame = cv2.VideoCapture(video_path)
    left_fit_coefs = None
    right_fit_coefs = None
    x_r = None
    _, sample_image = frame.read()
    visuals = Visualizer(sample_image, scale=1, frames=['Top View','Edges', 'Points', 'Polynomials', 'Turn Prediction'], save=True)
    while (frame.isOpened()):
        success, image = frame.read()
        if not success:
            print("\nEnd of frames\n")
            break
        
        scale = 1
        currentFrame = TurnPrediciton(image, scale)
        rows, cols, channels = image.shape

        gray = currentFrame.pre_processing(resize=False, gray=True, blur=False)
        cropped, region = currentFrame.crop_lane(gray)

        warped, dst, h_inv = currentFrame.warp_lane(cropped,region)
        edge = currentFrame.detect_edges(warped)
        visuals.update_frame(warped, "Top View")
        visuals.update_frame(edge, "Edges")

        linesP = currentFrame.hough_lines(edge)

        all_points_left, all_points_right, points = currentFrame.lines_to_points(linesP, warped)
        visuals.update_frame(points, "Points")

        final, left_fit_coefs, right_fit_coefs, x_r = currentFrame.fit_poly(all_points_left, all_points_right, dst, h_inv, image, left_fit_coefs, right_fit_coefs, x_r)
        dst = cv2.cvtColor(
                dst, cv2.COLOR_BGR2RGB)
        visuals.update_frame(dst, "Polynomials")
        
        visuals.update_final_frame(final, "Turn Prediction")

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    frame.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()