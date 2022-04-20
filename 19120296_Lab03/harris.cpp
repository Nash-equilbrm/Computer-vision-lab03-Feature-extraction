#include "harris.h"

vector<Corner> HarrisCornerDetector::detectHarris(const Mat& src, float k, float alpha) {
	// Chuyen doi ve anh grayscale
	Mat grayImg = toGrayScale(src);

	// Lam mo anh bang gaussian filter 
	Mat blurImg, gaussianFilter5x5 = gaussianKernel();
	filter2D(grayImg, blurImg, -1, gaussianFilter5x5);

	// Tinh dao ham Gx, Gy
	Mat gradient_x, gradient_y;
	Mat sobel_x = sobelXKernel(), sobel_y = sobelYKernel();

	filter2D(blurImg, gradient_x, CV_32FC1, sobel_x);
	filter2D(blurImg, gradient_y, CV_32FC1, sobel_y);

	// Tinh (Gx)^2, (Gy)^2, Gx.Gy  
	Mat gradient_x_square = matrixMultiply(gradient_x, gradient_x);
	Mat gradient_y_square = matrixMultiply(gradient_x, gradient_y);
	Mat gradient_x_y = matrixMultiply(gradient_y, gradient_y);

	// Thuc hien tinh chap cac ma tran tren voi gaussian filter
	filter2D(gradient_x_square, gradient_x_square, CV_32FC1, gaussianKernel(3, 1));
	filter2D(gradient_y_square, gradient_y_square, CV_32FC1, gaussianKernel(3, 1));
	filter2D(gradient_x_y, gradient_x_y, CV_32FC1, gaussianKernel(3, 1));

	// Tao ra mot ma tran hessian nhu sau:
	//
	// [Gx^2	GxGy]
	// [GxGy	Gy^2]
	//
	// cho tung pixel (y, x) = [[gradient_x_square, gradient_x_y], [gradient_x_y, gradient_y_square]]
	//	Tinh toan R[y, x] = det(M) - k. (trace(M))^2 va luu ket qua vao ma tran ket qua R 
	float r_max = INT_MIN;
	Mat R = Mat::zeros(src.size(), CV_32FC1);

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			// Tao ra ma tran M 
			
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

			   
	// So sanh gia tri R cua tung pixel voi gia tri nguong Threshold = alpha * r_max (r_max = gia tri Max trong ma tran R), luu lai vi tri nhung
	// pixel co gia tri R lon hon Threshold
 	vector<Corner> corner_points;

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			float r_val = getPixel(R, y, x);
			if (r_val > alpha * r_max)
				corner_points.push_back(Corner(r_val, y, x));
		}
	}
	sort(corner_points.begin(), corner_points.end());
	reverse(corner_points.begin(), corner_points.end());

	// Su dung thuat toan non-maximum suppression de loai bo bot cac diem corner trung lap hoac co vi tri qua gan nhau
	// Neu ton tai mot tap hop cac corner point co vi tri Manhattan qua gan nhau ( d <= 20) thi chon corner co R lon nhat

	float d = 20;
	vector<Corner> new_corner_points;

	for (Corner p1 : corner_points) {
		if (new_corner_points.size() > 0) {
			bool not_found = true;
			for (Corner p2 : new_corner_points) {
				not_found &= (abs(p1._x - p2._x) >= d) || (abs(p1._y - p2._y) >= d);
				if (!not_found && (p1._r > p2._r)) {
					p2._r = p1._r;
					p2._x = p1._x;
					p2._y = p1._y;

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


// show ra cac diem corner, radius = 5

void HarrisCornerDetector::showCorners(const Mat& src, float k, float alpha) {
	vector<Corner> cornerPoints = detectHarris(src, k, alpha);
	Mat dst = src.clone();
	for (Corner point : cornerPoints) {
		circle(dst, Point(point._x, point._y), 5, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("Harris corner detection", dst);
	waitKey(0);
	
}
