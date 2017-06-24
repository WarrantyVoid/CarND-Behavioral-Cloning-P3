import numpy as np
import matplotlib.pyplot as plt


############ Functions ############


# Shows data as bar graphs
def show_bar_graph(data, title=None, log_scale=False):
    x = data[1]
    y = data[0]
    fig, axis = plt.subplots(figsize=(15, 5))
    axis.set_axisbelow(True)
    axis.grid(True, linestyle=":")
    x_bar = np.zeros(len(y))
    for i in range(len(y)):
        x_bar[i] = (x[i] + x[i + 1]) / 2.0
    axis.bar(x_bar, y, width=x[1]-x[0], edgecolor="#000077")
    #axis.set_xticks(x)
    #axis.set_xticklabels(x)
    if title is not None:
        plt.title(title)
    if log_scale:
        axis.set_yscale('log')
    plt.show()


# Shows labeled data values
def show_labeled_graph(data, labels, title=None, invert_y=False):
    fig = plt.figure(figsize=(15, 5))
    handles = []
    for i in range(len(data)):
        h, = plt.plot(data[i], label=labels[i])
        handles.append(h)
    fig.get_axes()[0].set_axisbelow(True)
    fig.get_axes()[0].grid(True, linestyle=":")
    plt.legend(handles=handles)
    if invert_y:
        fig.get_axes()[0].invert_yaxis()
    if title is not None:
        plt.title(title)
    plt.show()


# Shows a Keras learning history
def show_learn_history(data, title=None):
    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    if title != None:
        plt.title(title)
    plt.show()
