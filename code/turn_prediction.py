# this script detects curved lanes, calculates its radius of curvature, and tells us whether it is a right or a left turn

import cv2
import numpy as np

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

    def warp_lane(self):
        pass

    def detect_edges(self):
        pass

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

    def lines_to_points(self):
        pass

    def fit_poly(self):
        pass

    
def main():
    # TODO: write turn prediction class with all the preprocessing 
    frame = cv2.VideoCapture('/home/ishan/Documents/UMD/portfolio/projects/turn_prediction/data/challenge.mp4')

    while (frame.isOpened()):
        success, image = frame.read()
        if not success:
            print("\nEnd of frames\n")
            break
        
        scale = 1
        currentFrame = TurnPrediciton(image, scale)
        rows, cols, channels = image.shape

        gray_image = currentFrame.pre_processing(resize=False, gray=True, blur=False)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # defining region of interest so that maximum length of the lane can be detected
        # TODO: come up with better way to crop using image shape parameters
        region = np.array([(230, rows - 60), (1120, rows - 60), (730, 440), (610, 440)])

        # cropping only the region of interest for lane detection
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, pts=[region], color=(255, 255))
        cropped = cv2.bitwise_and(gray, mask)

        # warping the cropped lane into a flat plane using homography
        # TODO: flat plane size should be proportional to the lane cropped 
        flat_plane = np.array([(0, 0), (200, 0), (200, 500), (0, 500)], dtype=float)
        h, status = cv2.findHomography(region, flat_plane)
        h_inv, status_inv = cv2.findHomography(flat_plane,region)
        warped = cv2.warpPerspective(cropped, h, (200,500))
        warped = cv2.flip(warped,0)
        dst = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        # detecting edges on the warped lane using canny edge detection
        edge = cv2.Canny(warped, 50, 450)  # 50,350

        # finding lines on the detected edges using hough transform. parameters have been tuned such that atleast 4 lines will be generated on both the left and the right curves of the lane
        linesP = cv2.HoughLinesP(edge, 2, np.pi / 180, 25, np.array([]), minLineLength=10, maxLineGap=150) # image, rho, theta, threshold = min # of intersections for line, lines, min, max

        # segregating left lines and right lines based on their location in the warped image
        # TODO: coloring left lane = red and right lane = green 
        left = []
        right = []
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

        # for i in all_points_left:
        #     cv2.circle(dst, i, 5, (255,0,0), 1)

        all_points_right = []
        for i in right:
            all_points_right.append((i[0],i[1]))
            all_points_right.append((i[2], i[3]))

        # for i in all_points_right:
        #     cv2.circle(dst, i, 5, (255,0,0), 1)

        # # code snippet to test points for detection window to check whether to continue with lane detection or use past values
        # p1 = (0,50)
        # p2 = (0,200)
        # p3 = (50,200)
        # p4 = (50,50)
        #
        # dst = cv2.line(dst, p1, p2, (255,0,50), 3)
        # dst = cv2.line(dst, p2, p3, (255, 0, 50), 3)
        # dst = cv2.line(dst, p3, p4, (255, 0, 50), 3)
        # dst = cv2.line(dst, p4, p1, (255, 0, 50), 3)

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

            # print("New Frame")
            # print("Left curve coefficients:" + str(left_fit_coefs))
            # print("Right curve coefficients:" + str(right_fit_coefs) + "\n")

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

        # displaying the final detection result
        # you can uncomment one of them and comment all others to properly visualize the various stages
        cv2.imshow("lane", final)
        # cv2.imshow("lane",dst)
        # cv2.imshow("preview", edge)
        # cv2.imshow("original", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    frame.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()