#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "utils.h"
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

// add padding for image before convolution step
Mat extendOriginalImage(Mat src, int rowExt, int colExt) {
	// initialize extend rows, cols
	Mat extend;
	extend.create(src.rows + 2 * rowExt, src.cols + 2 * colExt, src.type());
	extend.setTo(cv::Scalar::all(0));

	// extends original image
	src.copyTo(extend(Rect(rowExt, colExt, src.cols, src.rows)));

	// return extended image
	return extend;
}



// create kernel for Gaussian Filtering
double** createGaussianKernel(int radius) {
	int kernelSize = 2 * radius + 1;
	double sigma = max(kernelSize * 1.0 / 2, 1.0);
	// calculate 2 * (sigma ^ 2)
	double TwoSigmaPow2 = 2 * sigma * sigma;

	double sum = 0;
	double** kernel = new double* [kernelSize];

	// |i|, |j| is distance to the origin of kernel
	for (int i = -radius; i <= radius; ++i) {
		kernel[i + radius] = new double[kernelSize];

		for (int j = -radius; j <= radius; ++j) {
			double temp = i * i + j * j; // u^2 + v^2
			double kernelValue = exp((-1) * temp / TwoSigmaPow2) / (pi * TwoSigmaPow2);

			sum += kernelValue;
			kernel[i + radius][j + radius] = kernelValue;
		}
	}

	// calculate average pixel
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			kernel[i + radius][j + radius] /= sum;
		}
	}

	// return kernel
	return kernel;
}

Mat GaussianBlur(Mat src, int radius) {
	// create the kernel
	double** kernel = createGaussianKernel(radius);

	// convolution step
	return KernelConvolution(src, radius, kernel, 1);
}

// show image to screen
void showImage(Mat src, String windowName) {
	namedWindow(windowName);
	imshow(windowName, src);
	waitKey(0);
}


// function to perform matrix convolution
Mat KernelConvolution(Mat& image, float kernel[], int radius)
{
	int half_rad = radius / 2;

	int row = image.rows;
	int col = image.cols;

	Mat output(row, col, CV_8UC1);

	int image_Step = image.step[0];

	vector<int> offset;
	for (int i = -half_rad; i <= half_rad; i++)
	{
		for (int j = -half_rad; j <= half_rad; j++)
		{
			offset.push_back(image_Step * i + j);
		}
	}



	uchar* pixelDst = output.data;
	uchar* pixelSrc = image.data;

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++, pixelSrc++, pixelDst++)
		{
			if (i < half_rad || i >= row - half_rad || j < half_rad || j >= col - half_rad)
			{
				pixelDst[0] = 0;
				continue;
			}
			float sum = 0;

			for (int x = -half_rad; x <= half_rad; x++)
			{
				for (int y = -half_rad; y <= half_rad; y++)
				{
					int index = (x + half_rad) * radius + (y + half_rad);
					sum += pixelSrc[offset[index]] * kernel[index];
				}
			}

			if (sum < 0)
			{
				sum = 0;
			}
			pixelDst[0] = (int)sum;
		}

	}
	return output;
}