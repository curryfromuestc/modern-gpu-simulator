<h1><center><b>Tango: A Deep Neural Network Benchmark Suite for Various Accelerators</b></center></h1>
				
<b>TEAM</b><br>
[Aajna Karki](https://www.linkedin.com/in/aajna/)<br>
[Chethan Palangotu Keshava](https://www.linkedin.com/in/chethankeshava/)<br>
Spoorthi Mysore Shivakumar<br>
Goutam Madhukeshwar Hegde	<br>
			
<b>ADVISOR</b><br>
[Hyeran Jeon](https://sites.google.com/a/sjsu.edu/hyeran-jeon/home)

Computer Engineering Department<br>
San Jose State University

				
**************************************************************************************************************				

The most widely used five <b>CNNs</b> ([AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [CifarNet](https://www.cs.toronto.edu/~kriz/cifar.html), [ResNet](https://arxiv.org/abs/1512.03385), [SqueezeNet](https://arxiv.org/abs/1602.07360), [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) (will be soon uploaded) ) and two <b>RNNs</b> ([GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit), [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) are written in pure <b>CUDA</b> and <b>OpenCL</b> and tested on <b>GPGPU-Sim</b>, <b>NVIDIA GPUs</b>, and <b>Xilinx FPGAs</b> for computer architecture and NN algorithm research community. 

Predictions were verified against model files for image recognition (CNNs) and stock trend prediction (RNNs) which were trained by Caffe and Keras. The weight files are included in the benchmark suite. 

Chethan Keshava and Aajna Karki wrote CUDA code. <br>
Spoorthi Mysore Shivakumar and Goutam Madhukeshwar Hegde wrote OpenCL code.

The repository is being updated with more networks.
<b>Please cite the following paper when you use the benchmark suite.</b>

<i>
Aajna Karki, Chethan Palangotu Keshava, Spoorthi Mysore Shivakumar, Joshua Skow, Goutam Madhukeshwar Hegde, and Hyeran Jeon <br>
<b>"Detailed Characterization of Deep Neural Networks on GPUs and FPGAs,"</b> <br>
ACM Workshop on General Purpose GPUs (GPGPU), Providence, RI, April 2019
</i>


<h2><b>Directory Structure</b></h2><br>

<b>GPU</b><br>
This folder contains neural network inference models for NVIDIA GPU and GPGPU-Sim developed in CUDA.

<b>FPGA</b><br>
This folder contains neural network inference models for FPGA developed in OpenCL.

	
<h2><b>Build and Deployment</b></h2><br>
<b>GPU</b><br>
See README instructions inside each GPU neural network folder to build and deploy.<br>

<b>FPGA</b><br>
The steps for building and deploying neural network code for FPGA involves three tools:<br>
1. Vivado HLS<br>
2. Vivado<br>
3. Xilinx SDK<br>

Install these tools from the Xilinx website.<br>
https://www.xilinx.com/support/download.html

Follow the steps in the following document to build and deploy code on Xilinx FPGA.<br>
 [Tutorial_Executing_OpenCL_code_on_Xilinx_FPGA.pdf](FPGA/Tutorial_Executing_OpenCL_code_on_Xilinx_FPGA.pdf)

**************************************************************************************************************				

