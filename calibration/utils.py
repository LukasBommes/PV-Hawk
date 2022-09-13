import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def to_celsius(image):
    """Convert raw intensity values of radiometric image to Celsius scale."""
    return image*0.04-273.15


def preprocess_radiometric_frame(frame, equalize_hist=False):
    frame = to_celsius(frame)
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    frame = (frame*255.0).astype(np.uint8)
    if equalize_hist:
        frame = cv2.equalizeHist(frame)
    return frame
    
    
def load_images(filepath, mode="ir", max_num=None, preprocess_ir=True):
    images = []
    file_names = sorted(glob.glob(filepath))
    if max_num is not None:
        file_names = file_names[:max_num]
    for file_name in tqdm(file_names):
        if mode == "ir":
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            if preprocess_ir:
                image = preprocess_radiometric_frame(image)
                image = cv2.bitwise_not(image)
        elif mode == "rgb":
            image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        images.append(image)
    return images, file_names


def resize(images, dst_width, dst_height):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (dst_width, dst_height), interpolation=cv2.INTER_CUBIC)
    return images
    
    
def find_chessboard_corners(images, grid_size, mode, draw=False):
    rets = []
    img_points = []
    for image in images:
        if mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image, grid_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            corners = corners.reshape(1, -1, 2)
            img_points.append(corners)
        else:
            img_points.append(None)
        rets.append(ret)

        if draw:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(image_rgb, grid_size, corners, ret)
            plt.imshow(image_rgb)
            plt.show()
            print(ret)    
            
    return rets, img_points


def create_obj_points(num_images, grid_size):
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp = objp.reshape(1, -1, 3)
    obj_points = []
    for i in range(num_images):
        obj_points.append(objp)
    return obj_points


def map_ir_to_rgb(ir_pts, homography_ir_to_rgb):
    ir_pts = cv2.convertPointsToHomogeneous(ir_pts)
    rgb_pts = (homography_ir_to_rgb @ ir_pts.reshape(-1, 3).T).T
    rgb_pts = cv2.convertPointsFromHomogeneous(rgb_pts).reshape(-1, 2)
    return rgb_pts


def map_rgb_to_ir(rgb_pts, homography_rgb_to_ir):
    rgb_pts = cv2.convertPointsToHomogeneous(rgb_pts)
    ir_pts = (homography_rgb_to_ir @ rgb_pts.reshape(-1, 3).T).T
    ir_pts = cv2.convertPointsFromHomogeneous(ir_pts).reshape(-1, 2)
    return ir_pts
