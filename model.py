
import csv
import numpy as np
import random
import datetime
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import preprocessing as pre
import visualization as vis


############ Constants ############


# Relative position of left cam (in road width ratio)
POS_LEFT_CAM = (-1.0 / 6.0)

# Relative position of right cam (in road width ratio)
POS_RIGHT_CAM = (1.0 / 6.0)

# Time to look ahead for curvature
LOOK_AHEAD_TIME = datetime.timedelta(seconds=1, milliseconds=500)

############ Functions ############


# Read the driving log file
def read_driving_log(filepath):
    with open(filepath) as csvfile:

        # Read data from csv
        reader = csv.reader(csvfile, skipinitialspace=True)
        images = []
        steering = []
        times = []
        for line in reader:
            images.append([line[0], line[1], line[2]])
            steering.append([float(line[3]), float(line[3]), float(line[3])])
            times.append(pre.get_time(line[0]))

        # Calculate curvature value
        curvature = []
        for i in range(len(steering)):
            curvature.append(steering[i][0])
            frame_time = times[i]
            frame_count = 1
            while i + frame_count < len(steering):
                if times[i + frame_count] - frame_time > LOOK_AHEAD_TIME:
                    break
                curvature[i] += steering[i + frame_count][0]
                frame_count += 1
            curvature[i] /= frame_count

        # Calculate adjusted steering for left and right images based on curvature
        steering_distances = []
        for i in range(len(steering)):
            steering_distances.append(pre.get_steering_distance(curvature[i]))
            steering[i][1] = pre.adjust_steering_by_offset(steering[i][1], POS_LEFT_CAM, curvature[i])
            steering[i][2] = pre.adjust_steering_by_offset(steering[i][2], POS_RIGHT_CAM, curvature[i])

        features = np.array(images)
        labels = np.array(steering)
        vis.show_labeled_graph(
            [labels[:1200, 0:1], labels[:1200, 1:2], labels[:1200, 2:3], curvature[:1200]],
            ['steering center', 'steering  left', 'steering right', 'curvature'],
            title='Curvature vs steering for one lap', invert_y=True)

        return {'features': features, 'labels': labels}


# Filters the driving log data, normalizing feature frequency based on histogram
def filter_log_data(log_data):
    # Calculate histogram with 1° angle resolution
    images = log_data['features']
    steering = log_data['labels']
    histogram = np.histogram(steering[:, 0:1], bins=np.arange(-1.0, 1.0, 1.0 / pre.STEERING_FACTOR))
    vis.show_labeled_graph([np.cumsum(steering[:, 0:1])], ['steering'], title='Cumulated steering angles before filtering', invert_y=True)
    vis.show_bar_graph(histogram, title='Steering angle histogram before filtering')

    # Determine keeping probabilities
    bins_count = np.count_nonzero(histogram[0])
    bins_mean = sum(histogram[0]) / bins_count
    bins_max = np.max(histogram[0])
    bins_target_max = bins_mean * 1.414
    bins_factor = (bins_target_max - bins_mean) / (bins_max - bins_mean)
    keep_probabilities = np.ones(len(histogram[0]))
    for i in range(len(histogram[0])):
        if histogram[0][i] > bins_mean:
            keep_probabilities[i] = (bins_mean + (histogram[0][i] - bins_mean) * bins_factor) / histogram[0][i]

    # Perform the filtering
    filtered_images = []
    filtered_steering = []
    for i in range(len(steering)):
        keep_it = keep_probabilities[int(np.floor((1.0 + steering[i][0]) * pre.STEERING_FACTOR))]
        if random.uniform(0.0, 1.0) <= keep_it:
            filtered_images.append(images[i])
            filtered_steering.append(steering[i])

    features = np.array(filtered_images)
    labels = np.array(filtered_steering)
    histogram = np.histogram(labels[:, 0:1], bins=np.arange(-1.0, 1.0, 1.0 / pre.STEERING_FACTOR))
    vis.show_labeled_graph([np.cumsum(labels[:, 0:1])], ['steering'], title='Cumulated steering angles after filtering', invert_y=True)
    vis.show_bar_graph(histogram, title='Steering angle histogram after filtering')
    return {'features': features, 'labels': labels}


# Creates the neural network model in Keras
def create_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, activation='elu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(36, 5, 5, activation='elu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(48, 5, 5, activation='elu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
    model.add(Flatten())
    model.add(Dense(576, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model


############ Classes ############


# Class used for data generating in Keras
class LogDataGenerator:

    # Constructs new generator with given data, split and batch size
    def __init__(self, log_data, validation_split, batch_size):
        self.log_data = log_data
        self.batch_size = batch_size
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            log_data['features'],
            log_data['labels'],
            test_size=validation_split,
            random_state=0)
        self.image_width = 64
        self.image_height = 64
        self.image_depth = 3

    # Returns output shape of data
    def get_output_shape(self):
        return self.image_width, self.image_height, self.image_depth

    # Returns number of batches for one epoch
    def get_training_steps(self):
        return np.ceil(len(self.x_train) * 14 / self.batch_size)

    # Returns number of batches for one epoch
    def get_validation_steps(self):
        return np.ceil(len(self.x_valid) * 14 / self.batch_size)

    # Generate data for training
    def generate_training_data(self):
        while True:
            yield next(self.generate_data(self.x_train, self.y_train))

    # Generate data for testing
    def generate_validation_data(self):
        while True:
            yield next(self.generate_data(self.x_valid, self.y_valid))

    # Actual implementation of data generation
    def generate_data(self, x_data, y_data):
        # Memory allocation
        x_gen = np.zeros([self.batch_size, self.image_height, self.image_width, self.image_depth])
        y_gen = np.zeros([self.batch_size])
        while True:
            # New epoch
            x_data, y_data = shuffle(x_data, y_data)
            count_gen = 0
            for i in range(len(x_data)):
                # Generate center + augmentations
                center = pre.load(x_data[i][0])
                center_steer = y_data[i][0]
                count_gen = self.generate_feature(center, center_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(center, center_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.shear_feature(center, center_steer, random.uniform(-25, -15), 0.25)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.shear_feature(center, center_steer, random.uniform(15, 25), 0.25)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # Generate left + augmentations
                left = pre.load(x_data[i][1])
                left_steer = y_data[i][1]
                count_gen = self.generate_feature(left, left_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(left, left_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.shear_feature(left, left_steer, random.uniform(15, 25), 0.5)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # Generate right + augmentations
                right = pre.load(x_data[i][2])
                right_steer = y_data[i][2]
                count_gen = self.generate_feature(right, right_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(right, right_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.shear_feature(right, right_steer, random.uniform(-25, -15), 0.5)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

            # Yield remainder
            if count_gen > 0:
                yield x_gen, y_gen

    # Generates a new feature sample and increases count
    def generate_feature(self, image, steering, x_gen, y_gen, count_gen):
        x_gen[count_gen] = pre.preprocess(image)
        y_gen[count_gen] = steering
        count_gen += 1
        if count_gen >= self.batch_size:
            count_gen = 0
        return count_gen

    # Applies shearing to image and steering
    def shear_feature(self, image, steering, angle, steering_adjust):
        #import matplotlib.pyplot as plt
        #fig = plt.figure(figsize=(12, 4))
        #plt.title('Augmentation Test')
        #fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.axis('off')
        #axis = fig.add_subplot(1, 3, 1)
        #axis.imshow(image)
        #axis.set_title('original')
        image1 = pre.shear(image, angle)
        #axis = fig.add_subplot(1, 3, 2)
        #axis.imshow(image1)
        #axis.set_title('sheared')
        #image2 = pre.rotate(image, angle)
        #axis = fig.add_subplot(1, 3, 3)
        #axis.imshow(image2)
        #axis.set_title('rotated')
        #plt.show()
        return image1, pre.adjust_steering_by_angle(steering, angle * steering_adjust)

    # Applies flipping to image and steering
    def flip_feature(self, image, steering):
        image = pre.flip_horizontal(image)
        return image, -steering


############ Main logic ############


if __name__ == '__main__':
    print("Loading driving log..")
    log_data = read_driving_log('data/driving_log.csv')
    print("{} Entries".format(len(log_data['features'])))

    print("Filtering driving log..")
    log_data = filter_log_data(log_data)
    print("{} Entries".format(len(log_data)))

    print("Training..")
    data_gen = LogDataGenerator(log_data, 0.2, 512)
    model = create_model(data_gen.get_output_shape())
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(
        generator=data_gen.generate_training_data(),
        steps_per_epoch=data_gen.get_training_steps(),
        validation_data=data_gen.generate_validation_data(),
        validation_steps=data_gen.get_validation_steps(),
        nb_epoch=5)

    print("Saving..")
    model.save('model.h5')

    print(history.history.keys())
    vis.show_learn_history(history)
