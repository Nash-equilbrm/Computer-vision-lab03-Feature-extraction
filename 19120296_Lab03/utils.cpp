#include"Utils.h"

Mat toGrayScale(const Mat& src) {
	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);
	return dst;
}

float getPixel(const Mat& src, int y, int x) {
	int typeMatrix = src.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
		case CV_32F:
			return src.at<float>(y, x);
		default:    
			return (float)src.at<uchar>(y, x);
	}
}

void setPixel(Mat& src, int y, int x, float value) {
	int typeMatrix = src.type();
	uchar depth = typeMatrix & CV_MAT_DEPTH_MASK;

	switch (depth) {
		case CV_32F: 
			src.at<float>(y, x) = value;
			break;
		default:    
			src.at<uchar>(y, x) = (uchar)value;
			break;
	}
}

float matrixMaxValue(const Mat& src) {
	float max = INT_MIN;
	for (int y = 0; y < src.rows; ++y)
		for (int x = 0; x < src.cols; ++x)
			max = (max > getPixel(src, y, x) ? max : getPixel(src, y, x));
	return max;
}


Mat sobelXKernel() {
	return (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
}

Mat sobelYKernel() {
	return (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
}


Mat gaussianKernel(int kernelSize, float sigma, bool divide_by_sum, bool size_with_signma) {
	if (size_with_signma != false)
		kernelSize = (int)2 * ceil(3 * sigma) + 1;


	Mat gaussianKernel = Mat::zeros(kernelSize, kernelSize, CV_32FC1);
	float sum = 0.0;
	float var = 2 * sigma * sigma;
	float r, pi = 2 * acos(0);

	for (int y = -(kernelSize / 2); y <= kernelSize / 2; ++y) {
		for (int x = -(kernelSize / 2); x <= kernelSize / 2; ++x) {
			r = sqrt(x * x + y * y);
			gaussianKernel.at<float>(y + kernelSize / 2, x + kernelSize / 2) = exp(-(r * r) / var) / (pi * var);
			sum += gaussianKernel.at<float>(y + kernelSize / 2, x + kernelSize / 2);
		}
	}
	if (divide_by_sum == true) {
		for (int i = 0; i < kernelSize; ++i)
			for (int j = 0; j < kernelSize; ++j)
				gaussianKernel.at<float>(i, j) = gaussianKernel.at<float>(i, j) * 1.0 / sum;
	}

	return gaussianKernel;
}

Mat matrixMultiply(const Mat& mat1, const Mat& mat2) {
	assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);

	int height = mat1.rows, width = mat2.cols;
	Mat res = mat1.clone();

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float multiply_res = getPixel(mat1, y, x) * getPixel(mat2, y, x);
			setPixel(res, y, x, multiply_res);
		}

	return res;
}

Mat matrixMinus(const Mat& matrix1, const Mat& matrix2) {
	assert(matrix1.rows == matrix2.rows && matrix1.cols == matrix2.cols);

	int height = matrix1.rows, width = matrix2.cols;
	Mat res = matrix1.clone();

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float multiply_res = getPixel(matrix1, y, x) - getPixel(matrix2, y, x);
			setPixel(res, y, x, multiply_res);
		}

	return res;
}

void printKernel(const Mat& kernel)
{
	int h = kernel.rows, w = kernel.cols;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < h; ++x) {
			cout << getPixel(kernel, y, x) << " ";
		}
		cout << "\n";
	}
}

Mat LoGKernel(int gaussianSize, float signma, bool normalized, bool size_with_signma) {
	if (size_with_signma != false)
		gaussianSize = (int)2 * ceil(3 * signma) + 1;

	Mat LoG_kernel = Mat::zeros(gaussianSize, gaussianSize, CV_32FC1);
	float sum = 0.0;
	float two_sigma_squared = 2 * signma * signma;
	float pi = 2 * acos(0);
	float r;

	for (int y = -(gaussianSize / 2); y <= gaussianSize / 2; ++y) {
		for (int x = -(gaussianSize / 2); x <= gaussianSize / 2; ++x) {
			r = sqrt(x * x + y * y);
			float val = (-4.0) * (exp(-(r * r) / two_sigma_squared) * (1.0 - (r * r / two_sigma_squared))) / (pi * two_sigma_squared * two_sigma_squared);
			if (normalized == true)
				val = val * signma * signma;
			LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2) = val;
			sum += LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2);
		}
	}

	return LoG_kernel;
}

