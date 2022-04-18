#include "harris.h"

vector<CornerPoint> HarrisDetector::detectHarris(const Mat& src, float k, float alpha, float d) {
	// Convert Image to GrayScale
	Mat grayImg = toGrayScale(src);

	// Blur the image to reduce noise 
	Mat blurImg, gaussianKernel5x5 = gaussianKernel();
	filter2D(grayImg, blurImg, -1, gaussianKernel5x5);

	// Find gradient Gx, Gy
	Mat gradient_x, gradient_y;
	Mat sobel_x = sobelXKernel(), sobel_y = sobelYKernel();

	filter2D(blurImg, gradient_x, CV_32FC1, sobel_x);
	filter2D(blurImg, gradient_y, CV_32FC1, sobel_y);

	// Find (Gx)^2, (Gy)^2, Gx.Gy  
	Mat gradient_x_square = matrixMultiply(gradient_x, gradient_x);
	Mat gradient_y_square = matrixMultiply(gradient_x, gradient_y);
	Mat gradient_x_y = matrixMultiply(gradient_y, gradient_y);

	// Perform convolution on those matrix with gaussian filter
	filter2D(gradient_x_square, gradient_x_square, CV_32FC1, gaussianKernel(3, 1));
	filter2D(gradient_y_square, gradient_y_square, CV_32FC1, gaussianKernel(3, 1));
	filter2D(gradient_x_y, gradient_x_y, CV_32FC1, gaussianKernel(3, 1));

	// Create a matrix 2x2 M like this:
	//
	// [Gx^2	GxGy]
	// [GxGy	Gy^2]
	//
	// for every pixel (y, x) = [[gradient_x_square, gradient_x_y], [gradient_x_y, gradient_y_square]]
	//	Calculate R[y, x] = det(M) - k. (trace(M))^2 and store it to new matrix (image) 
	float r_max = INT_MIN;
	Mat R = Mat::zeros(src.size(), CV_32FC1);

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			// create matrix M 
			
			Mat M = Mat::zeros(2, 2, CV_32FC1);
			setPixel(M, 0, 0, getPixel(gradient_x_square, y, x));
			setPixel(M, 0, 1, getPixel(gradient_x_y, y, x));
			setPixel(M, 1, 0, getPixel(gradient_x_y, y, x));
			setPixel(M, 1, 1, getPixel(gradient_y_square, y, x));

			// det(M) = Gx^2 * Gy^2 - GxGy^2
			float det_M = getPixel(M, 0, 0) * getPixel(M, 1, 1) - getPixel(M, 1, 0) * getPixel(M, 0, 1);

			// trace(M) = Gx^2 + Gy^2
			float trace_M = getPixel(M, 0, 0) + getPixel(M, 1, 1);

			float r_val_of_pixel = det_M - k * trace_M * trace_M;
			r_max = (r_max > r_val_of_pixel) ? r_max : r_val_of_pixel;

			setPixel(R, y, x, r_val_of_pixel);
		}
	}

	// Step 6: 6.1. Compare value of R with threshold which is alpha * r_max (r_max is max pixel value in R), stored those position with
	//				R value larger than threshold
	//		   6.2. Use non-maximum suppression algorithm to suppress some consecutive corner-point within Distance range.
			   
	//6.1
	vector<CornerPoint> corner_points;

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			float r_val = getPixel(R, y, x);
			if (r_val > alpha * r_max)
				corner_points.push_back(CornerPoint(r_val, y, x));
		}
	}
	sort(corner_points.begin(), corner_points.end());
	reverse(corner_points.begin(), corner_points.end());

	//6.2
	vector<CornerPoint> new_corner_points;

	for (CornerPoint p1 : corner_points) {
		if (new_corner_points.size() > 0) {
			bool not_found = true;
			for (CornerPoint p2 : new_corner_points) {
				not_found &= (abs(p1.x - p2.x) >= d) || (abs(p1.y - p2.y) >= d);
				if (!not_found && (p1.r_value > p2.r_value)) {
					p2.r_value = p1.r_value;
					p2.x = p1.x;
					p2.y = p1.y;

				}

			}
				
			if (not_found)
				new_corner_points.push_back(p1);
		}
		else
			new_corner_points.push_back(p1);
	}
	return new_corner_points;

}

void HarrisDetector::showCorners(const Mat& src, const vector<CornerPoint>& cornerPoints) {
	Mat dst = src.clone();
	for (CornerPoint point : cornerPoints) {
		circle(dst, Point(point.x, point.y), 4, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("cornersDetector_Harris", dst);
	waitKey(0);
	
}
