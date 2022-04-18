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
			break;
		default:    
			return (float)src.at<uchar>(y, x);
			break;
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

Mat gaussianKernel(int gaussianSize, float signma, bool divide, bool size_with_signma) {
	if (size_with_signma == false)
		assert(gaussianSize % 2 == 1);
	else
		gaussianSize = (int)2 * ceil(3 * signma) + 1;

	Mat gaussianKernel = Mat::zeros(gaussianSize, gaussianSize, CV_32FC1);
	float sum = 0.0;
	float var = 2 * signma * signma;
	float r, pi = 2 * acos(0);

	for (int y = -(gaussianSize / 2); y <= gaussianSize / 2; ++y) {
		for (int x = -(gaussianSize / 2); x <= gaussianSize / 2; ++x) {
			r = sqrt(x * x + y * y);
			gaussianKernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2) = exp(-(r * r) / var) / (pi * var);
			sum += gaussianKernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2);
		}
	}
	if (divide == true) {
		for (int i = 0; i < gaussianSize; ++i)
			for (int j = 0; j < gaussianSize; ++j)
				gaussianKernel.at<float>(i, j) = gaussianKernel.at<float>(i, j) * 1.0 / sum;
	}

	return gaussianKernel;
}

Mat sobelXKernel() {
	return (Mat_<float>(3, 3) << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
}

Mat sobelYKernel() {
	return (Mat_<float>(3, 3) << -1, -2, -1,
		0, 0, 0,
		1, 2, 1);
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

Mat mimusElementWise(const Mat& mat1, const Mat& mat2) {
	assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);

	int height = mat1.rows, width = mat2.cols;
	Mat res = mat1.clone();

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float multiply_res = getPixel(mat1, y, x) - getPixel(mat2, y, x);
			setPixel(res, y, x, multiply_res);
		}

	return res;
}

Mat createLoG_Kernel(int gaussianSize, float signma, bool normalized, bool size_with_signma) {
	if (size_with_signma == false)
		assert(gaussianSize % 2 == 1);
	else
		gaussianSize = (int)2 * ceil(3 * signma) + 1;

	Mat LoG_kernel = Mat::zeros(gaussianSize, gaussianSize, CV_32FC1);
	float sum = 0.0;
	float var = 2 * signma * signma;
	float r, pi = 2 * acos(0);

	for (int y = -(gaussianSize / 2); y <= gaussianSize / 2; ++y) {
		for (int x = -(gaussianSize / 2); x <= gaussianSize / 2; ++x) {
			r = sqrt(x * x + y * y);
			float val = (-4.0) * (exp(-(r * r) / var) * (1.0 - (r * r / var))) / (pi * var * var);
			if (normalized == true) val = val * signma * signma;
			LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2) = val;
			sum += LoG_kernel.at<float>(y + gaussianSize / 2, x + gaussianSize / 2);
		}
	}

	return LoG_kernel;
}

