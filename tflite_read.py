import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

w1 = np.load('tflite_wb/tensor1_w.npy')
b1 = np.load('tflite_wb/tensor1_b.npy')

w2 = np.load('tflite_wb/tensor2_w.npy')
b2 = np.load('tflite_wb/tensor2_b.npy')

# print(w1.shape)
# print(b1.shape)

# print(w2.shape)
# print(b2.shape)
pred_success = 0
for r in range(100):

    pic_in = test_images[r].flatten()

    layer1 = [0] * 10
    for i in range(10):
        for j in range(784):
            layer1[i] += pic_in[j] * w1[i][j]
        layer1[i] += b1[i]
    #Relu
    layer1 = list(np.maximum(layer1,0))
    layer1 = [int(x) for x in layer1]
    print("Layer 1: ", layer1)


    layer2 = [0] * 10
    for i in range(10):
        for j in range(10):
            layer2[i] += layer1[j] * w2[i][j]
        layer2[i] += b2[i]

    layer2 = list(np.maximum(layer2,0))
    layer2 = [int(x) for x in layer2]
    print("Layer 2: ", layer2, '\n')
    
    print(layer2.index(max(layer2)))
    if(test_labels[r] == layer2.index(max(layer2))):
        pred_success += 1

print(pred_success,"%")
