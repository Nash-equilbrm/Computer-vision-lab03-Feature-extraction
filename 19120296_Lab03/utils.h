#pragma once

#ifndef UTILS_H
#define UTILS_H
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


Mat toGrayScale(const Mat& source);

// get va set gia tri cua pixel trong ma tran
float getPixel(const Mat& src, int y, int x);
void setPixel(Mat& src, int y, int x, float value);

// tim gia tri lon nhat trong ma tran
float matrixMaxValue(const Mat& src);

// tao ra cac filter de ap dung phep tinh chap voi anh
Mat sobelXKernel();
Mat sobelYKernel();
Mat gaussianKernel(int kernelSize, float signma = 1.0, bool calculate_size_with_sigma = false);
Mat LoGKernel(int gaussianSize = 5, float signma = 1.0, bool normalize = false, bool calculate_size_with_sigma = false);

void printKernel(const Mat& kernel);

// phep nhan ma tran
Mat matrixMultiply(const Mat& mat1, const Mat& mat2);
// phep tru ma tran => phuong phap DOG
Mat matrixMinus(const Mat& mat1, const Mat& mat2);




#endif