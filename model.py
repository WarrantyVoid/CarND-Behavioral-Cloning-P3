
import csv
import numpy as np
import random
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
POS_RIGHT_CAM = (+1.0 / 6.0)

# Steering horizon (in road width ratio)
steering_distance = 2.0

# Factor in between steering unit and degree
STEERING_FACTOR = 25


############ Functions ############


# Read the driving log file
def read_driving_log(filepath):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        images = []
        steering = []
        for line in reader:
            images.append([line[0], line[1], line[2]])
            steering.append([float(line[3])])
        return {'features': np.array(images), 'labels': np.array(steering)}


# Filters the driving log data, normalizing feature frequency based on histogram
def filter_log_data(log_data):
    # Calculate histogram with angle resolution
    images = log_data['features']
    steering = log_data['labels']
    histogram = np.histogram(steering, bins=np.arange(-1.0, 1.0, 1.0 / STEERING_FACTOR))
    vis.show_cumulated_graph([steering], ['steering'], title='Cumulated steering angles before filtering')
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
        keep_it = keep_probabilities[int(np.floor((1.0 + steering[i]) * STEERING_FACTOR))]
        if random.uniform(0.0, 1.0) <= keep_it:
            filtered_images.append(images[i])
            filtered_steering.append(steering[i])
    histogram = np.histogram(filtered_steering, bins=np.arange(-1.0, 1.0, 1.0 / STEERING_FACTOR))
    vis.show_cumulated_graph([filtered_steering], ['steering'], title='Cumulated steering angles after filtering')
    vis.show_bar_graph(histogram, title='Steering angle histogram after filtering')
    return {'features': np.array(filtered_images), 'labels': np.array(filtered_steering) }


# Creates the neural network model in Keras
def create_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, activation='elu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(32, 5, 5, activation='elu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(48, 5, 5, activation='elu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.8))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(500, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
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
        return (self.image_width, self.image_height, self.image_depth)

    # Returns number of samples for one epoch
    def get_training_size(self):
        return len(self.x_train) * 14

    # Returns number of samples for one epoch
    def get_validation_size(self):
        return len(self.x_valid) * 14

    # Generate data for training
    def generate_training_data(self):
        while True:
            yield self.generate_data(self.x_train, self.y_train).next()

    # Generate data for testing
    def generate_validation_data(self):
        while True:
            yield self.generate_data(self.x_valid, self.y_valid).next()

    # Actual implementation of data generation
    def generate_data(self, x_data, y_data):
        # Memory allocation
        x_gen = np.zeroes([self.batch_size, self.image_height, self.image_width, self.image_depth])
        y_gen = np.zeroes([self.batch_size])
        while True:
            # New epoch
            shuffle(x_data, y_data)
            count_gen = 0
            for i in range(len(x_data)):
                # Generate center + augmentations
                center = pre.load(x_data[i][0])
                center_steer = y_data[i]
                count_gen = self.generate_feature(center, center_steer, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(center, center_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.shear_feature(center, center_steer, random.uniform(-25, -15))
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.shear_feature(center, center_steer, random.uniform(+15, +25))
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                # Generate left + augmentations
                left = pre.load(x_data[i][1])
                left_steer = pre.adjust_steering_by_offset(y_data[i], POS_LEFT_CAM, steering_distance, STEERING_FACTOR)
                count_gen = self.generate_feature(left, left_steer, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(left, left_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.shear_feature(left, left_steer, random.uniform(-25, -15))
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                # Generate right + augmentations
                right = pre.load(x_data[i][2])
                right_steer = pre.adjust_steering_by_offset(y_data[i], POS_RIGHT_CAM, steering_distance, STEERING_FACTOR)
                count_gen = self.generate_feature(right, right_steer, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(right, right_steer)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.shear_feature(right, right_steer, random.uniform(+15, +25))
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

                image, steering = self.flip_feature(image, steering)
                count_gen = self.generate_feature(image, steering, x_gen, y_gen, count_gen)

            if count_gen > 0:
                yield x_gen, y_gen
                count_gen = 0

    # Generates a new feature and increases count
    def generate_feature(self, image, steering, x_gen, y_gen, count_gen):
        x_gen[count_gen] = pre.preprocess(image)
        y_gen[count_gen] = steering
        count_gen += 1
        if count_gen >= self.batch_size:
            yield x_gen, y_gen
            count_gen = 0
        return count_gen

    # Applies shearing to an image and steering
    def shear_feature(self, image, steering, angle):
        #fig = plt.figure(figsize=(12, 4))
        #plt.title('Augmentation Test')
        #fig.subplots_adjust(hspace=0.1, wspace=0.1)
        #plt.axis('off')
        #axis = fig.add_subplot(1, 3, 1)
        #axis.imshow(image)#((pre.preprocess(image) + 0.5) * 255).astype(np.uint8))
        #axis.set_title('original')
        image1 = pre.shear(image, angle)
        #axis = fig.add_subplot(1, 3, 2)
        #axis.imshow(image1)#((image1 + 0.5) * 255).astype(np.uint8))
        #axis.set_title('sheared')
        #image2 = pre.rotate(image, angle)
        #image2 = pre.preprocess(image2)
        #axis = fig.add_subplot(1, 3, 3)
        #axis.imshow(image2)#((image2 + 0.5) * 255).astype(np.uint8))
        #axis.set_title('rotated')
        #plt.show()
        return image1, pre.adjust_steering_by_angle(steering, angle / 2)

    # Applies flipping to an image and steering
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
    data_gen = LogDataGenerator(log_data, 0.2, 256)
    model = create_model(data_gen.get_output_shape())
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(
        generator=data_gen.generate_training_data(),
        steps_per_epoch=data_gen.get_training_size(),
        validation_data=data_gen.generate_validation_data(),
        validation_steps=data_gen.get_validation_size(),
        nb_epoch=4)

    print("Saving..")
    model.save('model.h5')

    print(history.history.keys())
    vis.show_learn_history(history)
