#include "utils.h"
#include "harris.h"

int main() {

	Mat img = imread("input1.png", IMREAD_COLOR);
	HarrisDetector harrisDetector;
	vector<CornerPoint> cornerPoints = harrisDetector.detectHarris(img);
	harrisDetector.showCorners(img, cornerPoints);
	return 0;
}