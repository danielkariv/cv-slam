import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


# Uses checkerboard to calibrate the camera. (TODO: learn how to do it without checkerboard)
class CameraCalibartion:
    def __init__(self, checkboard_pattern=(6, 9), ):
        self.CHECKERBOARD = checkboard_pattern
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = []

        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.prev_img_shape = None

    """ Update new points with checkboard images. Finish by getCameraCalibration()"""

    def processCamera(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD)
        if not retval:
            print("failed to find checkerboard.")
            return gray
        winSize = (11, 11)
        zeroZone = (-1, -1)
        self.objpoints.append(self.objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, winSize, zeroZone, self.criteria)
        self.imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray, self.CHECKERBOARD, corners2, retval)
        self.prev_img_shape = gray.shape[::-1]
        return img
        pass

    """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """

    def getCameraCalibration(self, debug=False, show=False):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.prev_img_shape,
                                                           None, None)
        # Debug info when getting camera calibration.
        if debug:
            print("Camera Matrix: ", mtx,
                  "\ndistortion: ", dist,
                  "\nrotation vector: ", rvecs,
                  "\ntranslation vector: ", tvecs)
        # display it?
        if show:
            vecs = np.array(tvecs)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vecs[:, [0]], vecs[:, [1]], vecs[:, [2]], c='r')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        # suppose to save it to a file, didn't test.
        np.savez("camera_calibration.npz", mtx, dist)
        return ret, mtx, dist, rvecs, tvecs


class Extractor:
    def __init__(self, k, dist):
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        # calibrated camera info
        self.k = k
        self.dist = dist
        # local usage
        self.lastFrame = None
        self.lastKeypoints = []
        self.lastDescriptors = []
        #
        self.lm_xyz = []
        self.cam_xyz = []
        self.scale = 1.0
        self.camPos = [0, 0, 0]
        pass

    def extract(self, gray_frame):
        kps, des = self.orb.detectAndCompute(gray_frame, None)

        return kps, des

    # Not working anymore
    def undistort_frame(self, frame):
        # undistort image
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.k, self.dist, (w, h), 0.0, )
        frame = cv2.undistort(frame, self.k, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
        return frame, newcameramtx

    def process_frame(self, frame, width, height):
        #current_frame, newcameramtx = self.undistort_frame(frame)  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        final_img = frame #current_frame
        currentKeypoints, currentDescriptors = self.extract(frame)
        if len(self.lastKeypoints) > 0 and len(self.lastDescriptors) > 0:
            matches = self.matcher.knnMatch(self.lastDescriptors, currentDescriptors, k=2)
            # draw matches.
            final_img = cv2.drawMatchesKnn(self.lastFrame, self.lastKeypoints,frame, currentKeypoints,
                                           matches[:10], None)

            points1 = []
            points2 = []
            # ratio test as in Lowe's paper
            for m in matches[:]:
                # make sure the matcher found pair
                if len(m) == 2:
                    (m1, m2) = m
                    if m1.distance < 0.75 * m2.distance:
                        (x1, y1) = self.lastKeypoints[m1.queryIdx].pt
                        (x2, y2) = currentKeypoints[m1.trainIdx].pt
                        # be within orb distance 32
                        if m1.distance < 32:
                            points1.append((x1, y1))
                            points2.append((x2, y2))
                else:
                    print("match isn't a pair, it got ", len(m), " values.")

            points1 = np.array(points1)
            points2 = np.array(points2)
            # hacking way to check if we got enough points.
            if len(points1) < 8 or len(points2) < 8:
                print("not enough points to run findEssentialMat")
                self.lastKeypoints = currentKeypoints
                self.lastDescriptors = currentDescriptors
                self.lastFrame = current_frame
                return final_img

            # works with known K (camera parameters)
            essentialMatrix, mask = cv2.findEssentialMat(points1, points2, self.k, threshold=1.0)

            if essentialMatrix is None:
                print("not enough points to create essentialMatrix")
                self.lastKeypoints = currentKeypoints
                self.lastDescriptors = currentDescriptors
                self.lastFrame = current_frame
                return final_img

            # calculate the Rotation matrix and Translation vector
            points, R, t, mask = cv2.recoverPose(essentialMatrix, np.float32(points1), np.float32(points2), self.k, 4)

            R = np.asmatrix(R).I

            # find the new camera position
            self.cam_xyz.append([self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]])

            # calculate the camera matrix
            C = np.hstack((R, t))
            P = np.asmatrix(self.k) * np.asmatrix(C)

            # turn the 2d landmarks into 3d points
            for i in range(min(10, len(points2))):
                # calculate the 3x1 matrix of the point
                x_i = np.asmatrix([points2[i][0], points2[i][1], 1]).T

                # calculate the 3d equivalent
                X_i = np.asmatrix(P).I * x_i
                self.lm_xyz.append([X_i[0][0] * self.scale + self.camPos[0],
                                    X_i[1][0] * self.scale + self.camPos[1],
                                    X_i[2][0] * self.scale + self.camPos[2]])

            # update the camera position
            self.camPos = [self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]]
            print("camPos: ", self.camPos)

        self.lastKeypoints = currentKeypoints
        self.lastDescriptors = currentDescriptors
        self.lastFrame = frame
        return final_img
        pass

    def show_data(self):
        self.lm_xyz = np.array(self.lm_xyz)
        self.cam_xyz = np.array(self.cam_xyz)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.lm_xyz[:, [0]], self.lm_xyz[:, [1]], self.lm_xyz[:, [2]], c='black', alpha=0.005)
        ax.scatter(self.cam_xyz[:, [0]], self.cam_xyz[:, [1]], self.cam_xyz[:, [2]], c=np.arange(len(self.cam_xyz)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig2, (ax3) = plt.subplots()
        map1 = ax3.imshow(np.stack([np.arange(len(self.cam_xyz)), np.arange(len(self.cam_xyz))]), cmap='viridis')
        fig.colorbar(map1, ax=ax)
        plt.show()


if __name__ == "__main__":

    def calibrate():
        cap = cv2.VideoCapture("IMG_6438.MOV") # Local video from phone ( not uploaded)
        W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cc = CameraCalibartion()

        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:

                frame = cv2.resize(frame, (int(W // 2), int(H // 2)))
                frame = cc.processCamera(frame)
                print((int(W // 2), int(H // 2)))
                cv2.imshow("Video", frame)
                cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
                # Press Q on keyboard to  exit, or wait 1ms.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                for i in range(20):
                    ret, frame = cap.read()
            else:
                break
        # release and finish.
        cap.release()
        cv2.destroyAllWindows()

        cc.getCameraCalibration(True, True)


    def load_and_track():
        mtx, dist = None, None
        with np.load('camera_calibration.npz', allow_pickle=True) as X:
            # Show how the npz file stored.
            for name in X:
                print("name:", name, "| array:", X[name])
            # extract the data needed
            mtx, dist = X['arr_0'], X['arr_1']
        # Load video to capture
        # TODO: change to args instead of hardcoded, and add webcam support.
        #cap = cv2.VideoCapture('IMG_6439.MOV') # local video (not uploaded)
        cap = cv2.VideoCapture('test_countryroad.mp4')
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        extractor = Extractor(mtx, dist)
        # camera parameters
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 0)
            if ret:
                # frame = process_frame(frame)
                #cut_width, cut_height = int(width // 2), int(height // 2)
                #frame = cv2.resize(frame, (cut_width, cut_height))
                frame = extractor.process_frame(frame, width,height)#cut_width, cut_height)
                cv2.imshow("Video", frame)
                # cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
                # Press Q on keyboard to  exit, or wait 1ms.
                if cv2.waitKey(16) & 0xFF == ord('q'):
                    break
                # skip every X frames.
                for i in range(3):
                    cap.grab()
            else:
                break
        # release and finish.
        cap.release()
        cv2.destroyAllWindows()

        extractor.show_data()


    # ---------- real main! -------------
    # calibrate()
    load_and_track()
