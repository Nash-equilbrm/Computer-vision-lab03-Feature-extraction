#pragma once

#ifndef UTILS_H
#define UTILS_H
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include <cmath>
#include <vector>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define EPS 1e-6

Mat toGrayScale(const Mat& source);

// get and set pixel value
float getPixel(const Mat& source, int y, int x);
void setPixel(Mat& source, int y, int x, float value);

// find a max value in matrix
float matrixMaxValue(const Mat& source);

// functions that return kernel filters
Mat sobelXKernel();
Mat sobelYKernel();
Mat gaussianKernel(int gaussianSize = 5, float signma = 1.0, bool divide = true, bool size_with_signma = false);
Mat createLoG_Kernel(int gaussianSize = 5, float signma = 1.0, bool normalized = false, bool size_with_signma = false);


// perform matrix multiply
Mat matrixMultiply(const Mat& mat1, const Mat& mat2);


Mat mimusElementWise(const Mat& mat1, const Mat& mat2);

#endif