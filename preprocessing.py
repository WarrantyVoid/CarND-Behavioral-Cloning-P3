import cv2
import numpy as np
import skimage.transform as imgtf
import re
import datetime

############ Constants ############


# Factor in between steering unit and degree
STEERING_FACTOR = 25.0

# Minimal steering distance (curve ahead)
MIN_STEERING_DISTANCE = 1.0

# Maximal steering distance (straight road)
MAX_STEERING_DISTANCE = 5.0

# Regular expression pattern to extract time from image file name
TIME_PATTERN = re.compile('_(\d*)')

############ Functions ############

def get_time(filename):
    result = TIME_PATTERN.findall(filename)
    date_str = ' '.join(result)
    return datetime.datetime.strptime(date_str, "%Y %m %d %H %M %S %f")


# Adds additional dimension to gray scale images (for neural nets)
def add_gray_dimension(images):
    return images[..., np.newaxis]


# Drops additional dimension in gray scale images (for image display/processing)
def drop_gray_dimension(images):
    return images.reshape(images.shape[:-1])


# Calculates adjusted steering angle base on camera delta and steering distance
def adjust_steering_by_angle(steering, view_angle):
    steering2 = steering + view_angle / STEERING_FACTOR
    return max(-1.0, min(steering2, 1.0))


# Calculates the steering distance given a curvature
def get_steering_distance(curvature, max_curvature=0.25):
    steering_range = MAX_STEERING_DISTANCE - MIN_STEERING_DISTANCE
    curvature = (abs(curvature) - 0.5 * max_curvature) * steering_range / max_curvature * 2.0
    return MAX_STEERING_DISTANCE - 1.0 / (1.0 + np.exp(-curvature)) * steering_range


# Calculates adjusted steering angle base on camera delta and steering distance
def adjust_steering_by_offset(steering, view_offset, curvature=0.0, max_curvature=0.25, steering_distance=None):
    if steering_distance is None:
        steering_distance = get_steering_distance(curvature, max_curvature)
    alpha = steering * STEERING_FACTOR
    b = np.tan(np.radians(alpha)) * steering_distance - view_offset
    alpha2 = np.degrees(np.arctan2(b, steering_distance))
    steering2 = alpha2 / STEERING_FACTOR
    return np.maximum(-1.0, np.minimum(steering2, 1.0))


# Loads image from disk as RGB
def load(image_filename):
    image = cv2.imread("{}".format(image_filename))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Flips image horizontally
def flip_horizontal(image):
    image = cv2.flip(image, 1)
    return image


# Shear
def shear(image, angle):
    a = angle / 180 * np.pi
    transform = imgtf.AffineTransform(matrix=np.array([[1, np.sin(a), -image.shape[0] * np.sin(a)], [0, 1, 0], [0, 0, 1]]))
    return imgtf.warp(image, transform, mode='edge')


# Rotate
def rotate(image, angle):
    return imgtf.rotate(image, -angle/2, center=(image.shape[0], image.shape[1]/2), mode='edge')


# Pre-processes an image by cropping, scaling and YUV conversion
def preprocess(image):
    image = cv2.resize(image, (64, 160))
    image = image[73:image.shape[0] - 23, 0:image.shape[1]]
    image = image.astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    image = (image - image_min) / (image_max - image_min) - 0.5
    return image


############ Main logic ############


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for x in range(-25, 26, 1):
        cs = x / 25.0
        ls = adjust_steering_by_offset(cs, -1.0 / 6.0, steering_distance=1)
        rs = adjust_steering_by_offset(cs, 1.0 / 6.0, steering_distance=1)
        print("cs: {:.2f} ls: {:.2f} rs: {:.2f}".format(cs, ls, rs))

    a = np.arange(-0.25, 0.26, 0.01)
    b = get_steering_distance(a)
    plt.plot(a, b)
    plt.title('Steering based on curvature')
    plt.xlabel('Curvature')
    plt.ylabel('Steering distance')
    plt.show()

    a = np.arange(1, 5.1, 0.1)
    b = adjust_steering_by_offset(0, -1.0 / 6, steering_distance=a)
    c = adjust_steering_by_offset(0, 1.0 / 6, steering_distance=a)
    plt.plot(a, b)
    plt.plot(a, c)
    plt.title('Steering adjustments left and right camera')
    plt.xlabel('Steering distance')
    plt.ylabel('Angle adjust')
    plt.show()

    a = np.arange(-0.25, 0.26, 0.1)
    b = adjust_steering_by_offset(a, -1.0 / 6, steering_distance=2.0)
    c = adjust_steering_by_offset(a, 1.0 / 6, steering_distance=2.0)
    plt.plot(a, a)
    plt.plot(a, b)
    plt.plot(a, c)
    plt.title('Steering adjustments left and right camera')
    plt.xlabel('Center angle')
    plt.ylabel('Adjusted Angle')
    plt.show()
