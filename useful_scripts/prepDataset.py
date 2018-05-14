import os
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
# uncomment for inline for the notebook:
# %matplotlib inline
import pickle

# enter the directory where the training images are:
TRAIN_DIR = 'train/'
IMAGE_SIZE = 512

train_image_file_names = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]

# to decode a single png img:
# graph = tf.Graph()
# with graph.as_default():
#     file_name = tf.placeholder(dtype=tf.string)
#     file1 = tf.read_file(file_name)
#     image = tf.image.decode_png(file1)

# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     image_vector = session.run(image, feed_dict={
#         file_name: train_image_file_names[1]})
#     print(image_vector)
#     session.close()

# method to decode many png images:
def decode_image(image_file_names, resize_func=None):

    images = []

    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file1 = tf.read_file(file_name)
        image = tf.image.decode_png(file1)
        # , channels=3) <-- use three channels for rgb pictures

        k = tf.placeholder(tf.int32)
        tf_rot_img = tf.image.rot90(image, k=k)

        # im_rot = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        tf_flip_img = tf.image.flip_left_right(tf_rot_img)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for i in range(len(image_file_names)):

            for j in range(4):  # rotation at 0, 90, 180, 270 degrees
                rotated_img = session.run(tf_rot_img, feed_dict={
                                          file_name: image_file_names[i], k: j})
                images.append(rotated_img)

                flipped_img = session.run(
                    tf_flip_img, feed_dict={
                        file_name: image_file_names[i], k: j})
                images.append(flipped_img)

            if (i+1) % 1000 == 0:
                print('Images processed: ', i+1)

        session.close()
    return images


train_images = decode_image(train_image_file_names)

print('shape train: ', np.shape(train_images))

# Let's see some of the images
# for i in range(10,14):
#     plt.imshow(train_images[i].reshape([IMAGE_SIZE,IMAGE_SIZE]), cmap=plt.get_cmap('gray'))
#     plt.show()

# for rgb images:
# for i in range(10,20):
#     plt.imshow(train_images[i])
#     plt.show()

def create_batch(data, label, batch_size):
    i = 0
    while i*batch_size <= len(data):
        with open(label + '_' + str(i) + '.pickle', 'wb') as handle:
            content = data[(i * batch_size):((i+1) * batch_size)]
            pickle.dump(content, handle)
            print('Saved', label, 'part #' + str(i),
                  'with', len(content), 'entries.')
        i += 1


# Create one hot encoding for labels
# labels = [[1., 0.] if 'dog' in name else [0., 1.] for name in train_image_file_names]
# these are all real images, so let's encode them all with 1's
labels = [[1., 0.] for name in train_image_file_names]

# TO EXPORT DATA WHEN RUNNING LOCALLY - UNCOMMENT THESE LINES
# a batch with 5000 images has a size of around 3.5 GB
# create_batch(labels, 'pickled/', np.shape(train_images)[0])
create_batch(train_images, 'pickled/', np.shape(train_images)[0])

print('done creating dataset')
