#include <stdio.h>
#include "../header_wb/imageTesting.h"

void nn(unsigned char input_data[784], unsigned char *res);

int main()
{
	unsigned char result;
	unsigned char correct_pred = 0;
	for(int i=0;i<100;i++)
	{
		nn(image[i], &result);
		if(result == image_labels[i])
			correct_pred++;
	}
	printf("%d", correct_pred);
	return 0;
}
