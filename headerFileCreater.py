import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

w1 = np.load('tflite_wb/tensor1_w.npy')
b1 = np.load('tflite_wb/tensor1_b.npy')

w2 = np.load('tflite_wb/tensor2_w.npy')
b2 = np.load('tflite_wb/tensor2_b.npy')

with open('header_wb/weights_biases.h', 'x') as file1:
    file1.write('const char layer1_w[10][784] = {\n')
    for i in range(9):
        file1.write('{' + str(list(w1[i]))[1:-1] + '},\n')
    file1.write('{' + str(list(w1[9]))[1:-1] + '}\n};\n\n')

    file1.write('const int layer1_b[10] = {')
    file1.write(str(list(b1))[1:-1] + '};\n\n')

    file1.write('const char layer2_w[10][10] = {\n')
    for i in range(9):
        file1.write('{' + str(list(w2[i]))[1:-1] + '},\n')
    file1.write('{' + str(list(w2[9]))[1:-1] + '}\n};\n\n')

    file1.write('const int layer2_b[10] = {')
    file1.write(str(list(b2))[1:-1] + '};\n\n')


pic_in = test_images[0].flatten()
print(test_labels[0])

with open('header_wb/imageTesting.h', 'x') as file2:
    file2.write('unsigned char image[100][784] = {\n')
    for i in range(99):
        file2.write('{' + str(list(test_images[i].flatten()))[1:-1] + '},\n')
    file2.write('{' + str(list(test_images[99].flatten()))[1:-1] + '}\n};\n\n')

    file2.write('unsigned char image_labels[100] = {\n')
    file2.write(str(list(test_labels)[:100])[1:-1] + '};\n\n')

