import os.path
import warnings
from distutils.version import LooseVersion

import matplotlib.pyplot as plt
import scipy.misc
import tensorflow as tf

import helper
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Extract tensors
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def conv_1x1(x, num_outputs):
    kernel_size = 1
    stride = 1
    return tf.layers.conv2d(
        x, num_outputs, kernel_size, stride,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # 1x1 convolution of vgg_layer7_out
    vgg_layer7_out_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, 1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='vgg_layer7_out_1x1')

    # Upsample vgg_layer7_out_1x1 and pass to next layer
    vgg_layer7_out_1x1_upsampled = tf.layers.conv2d_transpose(
        vgg_layer7_out_1x1, num_classes, 4,
        strides=(2, 2),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='vgg_layer7_out_1x1_upsampled')

    # Scale vgg_layer4_out
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')

    # 1x1 convolution of vgg_layer4_out out and then use as skip layer
    vgg_layer4_out_1x1 = tf.layers.conv2d(
        vgg_layer4_out_scaled, num_classes, 1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='vgg_layer4_out_1x1')

    # New layer with feed from vgg_layer7_out_1x1_upsampled and skip connection vgg_layer4_out_1x1
    # new_layer1_out is symmetric to vgg layer 4
    new_layer1_out = tf.add(vgg_layer7_out_1x1_upsampled, vgg_layer4_out_1x1, name='new_layer1_out')

    # Upsample new_layer1_out for next layer
    new_layer1_out_upsampled = tf.layers.conv2d_transpose(
        new_layer1_out, num_classes, 4,
        strides=(2, 2),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='new_layer1_out_upsampled')

    # Scale vgg_layer3_out
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')

    # 1x1 convolution of vgg_layer3_out and then use as skip layer
    vgg_layer3_out_1x1 = tf.layers.conv2d(
        vgg_layer3_out_scaled, num_classes, 1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='vgg_layer3_out_1x1')

    # New layer with feed from new_layer1_out_upsampled and skip connection vgg_layer3_out_1x1
    # new_layer2_out is symmetric to vgg layer 3
    new_layer2_out = tf.add(new_layer1_out_upsampled, vgg_layer3_out_1x1, name='new_layer2_out')

    # Upsample new_layer2_out and get the final layer
    final_layer_out = tf.layers.conv2d_transpose(
        new_layer2_out, num_classes, 16,
        strides=(8, 8),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='final_layer_out')

    return final_layer_out


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())

    epoch_training_losses = []

    print()
    print("Training...")
    print()
    for i in range(epochs):
        epoch_training_loss = 0
        print("EPOCH {} ...".format(i + 1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.0009})
            print("Loss: = {:.3f}".format(loss))
            epoch_training_loss = loss
        print()

        epoch_training_losses.append(epoch_training_loss)

    return epoch_training_losses


tests.test_train_nn(train_nn)


def print_tensor_sizes(sess, image_file, image_shape, keep_prob, image_input, *argv):
    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

    # keep_prob and image_input tensors are a must, other tensors are given in the end in *argv
    tensors = [image_input] + list(argv)
    ops_to_run = []

    # Add op for image_input tensor first, then add other tensors
    for tensor in tensors:
        ops_to_run.append(tf.shape(tensor))

    tensor_shapes = sess.run(ops_to_run, feed_dict={image_input: [image], keep_prob: 0.5})

    print()
    print("tensor shapes:")
    for t, ts in zip(tensors, tensor_shapes):
        print(t.name, " shape: ", ts)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    final_run_dir = './final_run'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        ###########################################################################################
        # TODO: Build NN using load_vgg, layers, and optimize function
        ###########################################################################################
        epochs = 20
        batch_size = 5

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)

        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        ###########################################################################################
        # Print trained vgg graph tensor shapes
        ###########################################################################################
        image_file = os.path.join(data_dir, 'data_road', 'training', 'image_2', 'um_000000.png')
        print_tensor_sizes(sess, image_file, image_shape, keep_prob,
                           image_input, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out)

        print()
        print("trainable variables:")
        for tv in tf.trainable_variables():
            print(tv)

        ###########################################################################################
        # TODO: Train NN using the train_nn function
        ###########################################################################################
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)
        epoch_training_losses = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                                         image_input, correct_label, keep_prob, learning_rate)

        ###########################################################################################
        # TODO: Save inference data using helper.save_inference_samples
        ###########################################################################################
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        ###########################################################################################
        # Plot training losses
        ###########################################################################################
        plt.plot(epoch_training_losses)
        plt.title("Training Loss vs. Epoch")
        plt.xlabel("Epochs")
        plt.xticks(range(epochs))
        plt.ylabel("Training Loss")
        plt.savefig(os.path.join(final_run_dir, 'training_losses.png'))

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
