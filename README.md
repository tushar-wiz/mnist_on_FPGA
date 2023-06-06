
## Project Structure
All the files are listed below with their use mentioned
```
/VitisFlow - Contains files for VITIS HLS
  nn.cpp      - Code for the MNIST Neural Net
  test_nn.cpp - Testbench code for nn.cpp

/VivadoFlow
  project.xpr - Testing the generated IP (in progress)

/header_wb
  imageTesting.h   - Images and Image Labels in header format (accessed by test_nn.cpp)
  weights_biases.h - Weights and Biases in header format (accessed by nn.cpp)

/tflite_wb - weights and biases of layer 1 and layer 2 in .npy format
  tensor1_b.npy
  tensor1_w.npy
  tensor2_b.npy
  tensor2_w.npy

cnnFromScratch.py    - Tried writing NN without any AI/ML libraries (https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang)
hdf5_to_csv.py       - To extract weights and biases from hdf5 file (Doesn't work)
headerFileCreater.py - Creates the header files under header_wb
tflite_read.py       - same algortihm as /VitisFlow/nn.cpp but in python
tflite_try.py        - Creating the trained model and quantizing it using tflite
vitis_hls.log        - Vitis Logs
```

## Flow
![Flow for this project](https://imgur.com/joPAo7k.png)
> Text in red means that segment hasn't been completed

## Quantized MNIST Neural Network
![Quantized NN Arch](https://imgur.com/vqhjixR.png)

## Vitis Results 
![Directives](https://imgur.com/0I4pX1f.png)
### Synthesis
> NOTE - Synthesis has been performed for Part Basys3(xc7a35tcpg236-1)
1. Performance and Resource Estimates
![Synth Results](https://imgur.com/ZjzGSyM.png)
2. Function Call Graph
![Function Call Graph](https://imgur.com/vQjREIN.png)

### Cosimulation
1. Performance 
![](https://imgur.com/nYS9wxX.png)
2. Waveform
![](https://imgur.com/uZsTOda.png)
> Tested on 100 images with a result of 92% accuracy where total simulation time was 7.948195ms that is an avg of 79.48uS per image.

### Implementation
1. IP
![](https://imgur.com/9lHXlwV.png)