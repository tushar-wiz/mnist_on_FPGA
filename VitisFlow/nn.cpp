#include "../header_wb/weights_biases.h"

void nn(unsigned char input_data[784], unsigned char *res)
{
	int layer1[10] = {0};
	int layer2[10] = {0};

	A: for(char i=0; i<10; i++)
	{
		B : for(short int j=0; j<784; j++)
		{
			#pragma HLS PIPELINE
			layer1[i] += input_data[j] * layer1_w[i][j];
		}
		layer1[i] += layer1_b[i];
		if(layer1[i] < 0)	layer1[i] = 0;
	}

	C: for(char i=0; i<10; i++)
	{
		#pragma HLS UNROLL
		D: for(char j=0; j<10; j++)
		{
			#pragma HLS UNROLL
			layer2[i] += layer1[j] * layer2_w[i][j];
		}
		layer2[i] += layer2_b[i];
		if(layer2[i] < 0)	layer2[i] = 0;
	}

	unsigned char index = 0;
	int r = 0;

	MAX_find: for(int i=0; i<10; i++)
	{
		#pragma HLS UNROLL
		if(layer2[i] > r)
		{
			index = i;
			r = layer2[i];
		}
	}
	*res = index;
	return ;
}
