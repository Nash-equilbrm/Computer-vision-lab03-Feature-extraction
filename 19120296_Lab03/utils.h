#pragma once
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#define pi ((atan(1))*(4))

using namespace cv;


// These utilities are reused form previous lab02: Edge detection

Mat extendOriginalImage(Mat inputImage, int rowExt, int colExt);
Mat KernelConvolution(Mat inputImage, int radius, double** kernel, int flag = 1);
double** createGaussianKernel(int radius);
Mat GaussianBlur(Mat src, int radius);
void showImage(Mat src, String windowName);
Mat KernelConvolution(Mat& image, float kernel[], int size);
