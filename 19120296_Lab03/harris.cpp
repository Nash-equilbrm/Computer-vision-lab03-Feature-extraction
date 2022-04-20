#include "harris.h"

vector<Corner> HarrisCornerDetector::detectHarris(const Mat& src, float k, float alpha) {
	// Chuyen doi ve anh grayscale
	Mat grayImg = toGrayScale(src);

	// Lam mo anh bang gaussian filter 
	Mat blurImg, gaussianFilter = gaussianKernel(5);
	filter2D(grayImg, blurImg, -1, gaussianFilter);

	// Tinh dao ham Gx, Gy
	Mat sobel_x = sobelXKernel(), sobel_y = sobelYKernel();
	Mat Gx, Gy;

	filter2D(blurImg, Gx, CV_32FC1, sobel_x);
	filter2D(blurImg, Gy, CV_32FC1, sobel_y);

	// Tinh (Gx)^2, (Gy)^2, Gx.Gy  
	Mat Gx_square = matrixMultiply(Gx, Gx);
	Mat Gy_square = matrixMultiply(Gy, Gy);
	Mat GxGy = matrixMultiply(Gx, Gy);

	// Thuc hien tinh chap cac ma tran tren voi gaussian filter
	filter2D(Gx_square, Gx_square, CV_32FC1, gaussianKernel(3));
	filter2D(Gy_square, Gy_square, CV_32FC1, gaussianKernel(3));
	filter2D(GxGy, GxGy, CV_32FC1, gaussianKernel(3));

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
			setPixel(M, 0, 0, getPixel(Gx_square, y, x));
			setPixel(M, 0, 1, getPixel(GxGy, y, x));
			setPixel(M, 1, 0, getPixel(GxGy, y, x));
			setPixel(M, 1, 1, getPixel(Gy_square, y, x));

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
 	vector<Corner> corners;

	for (int y = 0; y < R.rows; ++y) {
		for (int x = 0; x < R.cols; ++x) {
			float r_val = getPixel(R, y, x);
			if (r_val >= alpha * r_max)
				corners.push_back(Corner(r_val, y, x));
		}
	}
	/*sort(corners.begin(), corners.end());
	reverse(corners.begin(), corners.end());*/

	// Su dung thuat toan non-maximum suppression de loai bo bot cac diem corner trung lap hoac co vi tri qua gan nhau
	// Neu ton tai mot tap hop cac corner point co vi tri Manhattan qua gan nhau ( d <= 20) thi chon corner co R lon nhat

	float d = 10;
	vector<Corner> new_corners;

	for (Corner p1 : corners) {
		if (new_corners.size() > 0) {
			bool not_found = true;
			for (Corner p2 : new_corners) {
				not_found &= (abs(p1._x - p2._x) >= d) || (abs(p1._y - p2._y) >= d);
				if (!not_found && p1._r > p2._r) {
					p2._r = p1._r;
					p2._x = p1._x;
					p2._y = p1._y;
				}

			}
				
			if (not_found)
				new_corners.push_back(p1);
		}
		else
			new_corners.push_back(p1);
	}
	return new_corners;

}


// show ra cac diem corner, radius = 5

void HarrisCornerDetector::showCorners(const Mat& src, float k, float alpha) {
	vector<Corner> cornerPoints = detectHarris(src, k, alpha);
	Mat dst = src.clone();
	for (Corner point : cornerPoints) {
		circle(dst, Point(point._x, point._y), 5, Scalar(0, 0, 255), 2);
	}
	imshow("Harris corner detection", dst);
	waitKey(0);
	
}
