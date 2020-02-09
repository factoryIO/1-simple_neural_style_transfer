import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from vgg_model import load_vgg_model
from image_tools import *

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

CONTENT_LAYER = 'conv4_2'

alpha = 1
beta = 10
gamma = 10 ** -3

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--vgg_model', action='store', type=str, required=True)
my_parser.add_argument('--content_image', action='store', type=str, required=True)
my_parser.add_argument('--style_image', action='store', type=str, required=True)
my_parser.add_argument('--output_image', action='store', type=str, default='/app/artifacts/generated_image.jpg')
args = my_parser.parse_args()


def get_content_image_activations(sess, model, content_image):
    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    out = model[CONTENT_LAYER]
    a_C = sess.run(out)

    return a_C, out


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def get_style_image_activations(sess, model, STYLE_LAYERS, style_image):
    sess.run(model['input'].assign(style_image))

    a_S = {}
    for layer_name, _ in STYLE_LAYERS:
        out = model[layer_name]
        a_S[layer_name] = sess.run(out)

    return a_S


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * n_C **2 * (n_W * n_H) ** 2)

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS, a_S):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_G = out
        J_style += coeff * compute_layer_style_cost(a_S[layer_name], a_G)

    return J_style


def compute_total_variation_regularization(image):
    return tf.image.total_variation(image)[0]


def total_cost(J_content, J_style, J_TV, alpha, beta, gamma):
    J = alpha * J_content + beta * J_style + gamma * J_TV
    return J


def define_total_cost_function(model, STYLE_LAYERS, a_C, a_S, a_G):
    J_content = compute_content_cost(a_C, a_G)
    J_style = compute_style_cost(model, STYLE_LAYERS, a_S)
    J_TV = compute_total_variation_regularization(model['input'])
    J = total_cost(J_content, J_style, J_TV, alpha, beta, gamma)
    return J_content, J_style, J_TV, J


def model_nn(sess, model, train_step, input_image, num_iterations=1000):

    sess.run(tf.compat.v1.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

    # save last generated image
    save_image(args.output_image, generated_image)

    return generated_image


if __name__ == "__main__":

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    # Load and prepare content and style images
    content_image = plt.imread(args.content_image)
    content_image = reshape_and_normalize_image(content_image)
    style_image = plt.imread(args.style_image)
    style_image = reshape_and_normalize_image(style_image)

    # Initialize output image
    generated_image = generate_noise_image(content_image)

    # Load pre-trained VGG19 model
    model = load_vgg_model(args.vgg_model)

    # Set a_C to be the hidden layer activation from the layer we have selected with content image as input
    # Set a_G to be the activations from the same layer, but not yet evaluated (still tensors)
    a_C, a_G = get_content_image_activations(sess, model, content_image)

    # Set a_S to be the hidden layers activations from the layers we have selected with style image as input
    a_S = get_style_image_activations(sess, model, STYLE_LAYERS, style_image)

    # define cost function, optimizer and training steps
    J_content, J_style, J_TV, J = define_total_cost_function(model, STYLE_LAYERS, a_C, a_S, a_G)
    optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    model_nn(sess, model, train_step, generated_image, 1000)
