import time
import numpy as np
import tensorflow as tf
import pprint
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage

try:
    xrange
except:
    xrange = range

epoch=15000
learning_rate=1e-4
image_size=33
label_size=21
batch_size=128
c_dim=1
scale=3
stride=14
checkpoint_dir="checkpoint"
sample_dir="sample"
is_train=True
sess = tf.Session()
is_grayscale = (c_dim == 1)

# ---------------------------------- Building model ----------------------------------
images = tf.placeholder(tf.float32, [None, image_size, image_size, c_dim], name='images')
labels = tf.placeholder(tf.float32, [None, label_size, label_size, c_dim], name='labels')

weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
}
biases = {
    'b1': tf.Variable(tf.zeros([64]), name='b1'),
    'b2': tf.Variable(tf.zeros([32]), name='b2'),
    'b3': tf.Variable(tf.zeros([1]), name='b3')
}

conv1 = tf.nn.relu(tf.nn.conv2d(images, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
pred = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3']

# Loss function (MSE)
loss = tf.reduce_mean(tf.square(labels - pred))
saver = tf.train.Saver()

def modcrop(image, scale=3):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def preprocess(path, scale=3):
  image = scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  label_ = modcrop(image, scale)
  image = image / 255.
  label_ = label_ / 255.
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def train_input_setup():
    # Preparing data
    dataset="Train"
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) / 2 # 6

    for i in xrange(len(data)):
        # Preprocessing
        input_, label_ = preprocess(data[i], scale)

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        for x in range(0, h-image_size+1, stride):
            for y in range(0, w-image_size+1, stride):
                sub_input = input_[x:x+image_size, y:y+image_size] # [33 x 33]
                sub_label = label_[x+int(padding):x+int(padding)+label_size, y+int(padding):y+int(padding)+label_size] # [21 x 21]
                sub_input = sub_input.reshape([image_size, image_size, 1])  
                sub_label = sub_label.reshape([label_size, label_size, 1])
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    # Making data
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=np.asarray(sub_input_sequence))
        hf.create_dataset('label', data=np.asarray(sub_label_sequence))

def test_input_setup():
  # Preparing data
    dataset="Test"
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) / 2 # 6

    # Preprocessing
    input_, label_ = preprocess(data[2], scale)

    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    for x in range(0, h-image_size+1, stride):
        nx += 1; ny = 0
        for y in range(0, w-image_size+1, stride):
            ny += 1
            sub_input = input_[x:x+image_size, y:y+image_size] # [33 x 33]
            sub_label = label_[x+int(padding):x+int(padding)+label_size, y+int(padding):y+int(padding)+label_size] # [21 x 21]
            sub_input = sub_input.reshape([image_size, image_size, 1])  
            sub_label = sub_label.reshape([label_size, label_size, 1])
            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)

    # Making data
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=np.asarray(sub_input_sequence))
        hf.create_dataset('label', data=np.asarray(sub_label_sequence))

    return nx, ny

def train(sess, checkpoint_dir):
    train_input_setup()
    data_dir = os.path.join('./{}'.format(checkpoint_dir), "train.h5")
    
    # Reading Data
    with h5py.File(data_dir, 'r') as hf:
        train_data = np.array(hf.get('data'))
        train_label = np.array(hf.get('label'))

    # Stochastic gradient descent with the standard backpropagation
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if load(checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    print("Training...")

    for ep in xrange(1500):
        # Run by batch images
        batch_idxs = len(train_data) // batch_size
        for idx in xrange(0, batch_idxs):        
            batch_images = train_data[idx*batch_size : (idx+1)*batch_size]
            batch_labels = train_label[idx*batch_size : (idx+1)*batch_size]

            counter += 1
            _, err = sess.run([train_op, loss], feed_dict={images: batch_images, labels: batch_labels})

            if counter % 10 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-start_time, err))

            if counter % 500 == 0:
                model_name = "SRCNN.model"
                model_dir = "%s_%s" % ("srcnn", label_size)
                checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)

def test(sess, checkpoint_dir) :
    nx, ny = test_input_setup()
    data_dir = os.path.join('./{}'.format(checkpoint_dir), "test.h5")
    
    # Reading Data
    with h5py.File(data_dir, 'r') as hf:
        train_data = np.array(hf.get('data'))
        train_label = np.array(hf.get('label'))

    # Stochastic gradient descent with the standard backpropagation
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    counter = 0
    start_time = time.time()

    if load(checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    print("Testing...")
    result = pred.eval({images: train_data, labels: train_label})
    # Merging
    size = [nx, ny]
    h, w = result.shape[1], result.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(result):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    result = img
    result = result.squeeze()
    image_path = os.path.join(os.getcwd(), sample_dir)
    image_path = os.path.join(image_path, "test_image.png")
    scipy.misc.imsave(image_path, result)

def load(checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
 

# -------------------------------------- Main --------------------------------------
def main(_):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    with tf.Session() as sess:
        train(sess, checkpoint_dir)
        test(sess, checkpoint_dir)

if __name__ == '__main__':
    tf.app.run()
