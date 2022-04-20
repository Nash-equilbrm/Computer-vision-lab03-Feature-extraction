#include "blob.h"

vector<Blob> BlobDetector::detectBlob(const Mat& src, float sigma, float k, float threshold) {
	// chuyen ve anh xam
	Mat srcGray = toGrayScale(src);


	// Tinh chap anh xam voi LoG filter.
	// Gom co 10 scale, moi scale tuong ung voi moi sigma, sigma tang theo cap so nhan k

	int number_of_scales = 10;
	vector<Mat> LoGs(number_of_scales, Mat::zeros(src.size(), CV_32FC1));
	float signma_y = sigma;

	vector<float> max_log(number_of_scales, 0);

	for (int idx = 0; idx < LoGs.size(); ++idx) {
		signma_y = (idx == 0) ? signma_y : (signma_y * k);

		Mat log_filter = LoGKernel(5, signma_y, true, true);
		Mat conv;

		filter2D(srcGray, conv, CV_32FC1, log_filter);

		conv = matrixMultiply(conv, conv);

		max_log[idx] = matrixMaxValue(conv);

		LoGs[idx] = conv;
	}

	// tim diem local extrema, bang cach so sanh gia tri LoG cua pixel dang xet voi gia tri LoG cua cac pixel lan can va cac pixel tuong ung tai 
	// cac scale lan can. Tong cong la so sanh voi 26 pixel khac.
	
	// Ngoai tru neu pixel dang xet nam o ma tran cua LoG tinh bang sigma dau tien hoac cuoi cung cua scale-space thi chi so sanh voi 17 diem khac.
	// (Vi ma tran voi scale lan can no chi co 1)
	vector<Blob> blobs;
	for (int idx = 0; idx < LoGs.size(); ++idx) {
		for (int y = 0; y < src.rows; ++y) {
			for (int x = 0; x < src.cols; ++x) {
				float val = getPixel(LoGs[idx], y, x);
				if (val <= threshold * max_log[idx]) 
					continue;

				bool foundPeak = true;

				// step_idx xac dinh matrix ket qua LoG voi scale lan can
				for (int step_idx = -1; step_idx <= 1; ++step_idx) {
					if (foundPeak == false)
						break;
					// step_x va step_y xac dinh cac Pixel lan can
					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (foundPeak == false)
							break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (foundPeak == false) 
								break;

							int cur_idx = idx + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_idx >= LoGs.size() || cur_idx < 0)
								continue;
							if (cur_y >= src.rows || cur_y < 0) 
								continue;
							if (cur_x >= src.cols || cur_x < 0) 
								continue;

							if (val < getPixel(LoGs[cur_idx], cur_y, cur_x))
								foundPeak = false;
						}
					}
				}

				if (foundPeak == true)
					blobs.push_back(Blob(pow(k, idx) * sigma, y, x));
			}
		}
	}


	return blobs;
}

void BlobDetector::showBlobsWithLoGDetector(const Mat& src, float sigma, float k, float threshold) {
	vector<Blob> blobs = detectBlob(src, sigma, k, threshold);
	
	// Moi blob co ban kinh r = sigma*sqrt(2), voi sigma luu trong gia tri thu 3 cua tuple
	Mat dst = src.clone();
	for (Blob blob : blobs)
		circle(dst, Point(blob._x, blob._y), blob._val * sqrt(2), Scalar(0, 0, 255), 2, 8, 0);

	imshow("Laplace of Gaussian Blob Detection", dst);
	waitKey(0);
}



vector<Blob> BlobDetector::detectDOG(const Mat& source, float sigma, float k, float thresholdMax) {
	// chuyen ve anh gray scale
	Mat srcGray = toGrayScale(source);

	// Tao ra cac Gaussian filter voi cac sigma khac nhau, voi sigma cang lon thi filter co kich thuoc cang lon vi nhu vay moi the hien ro tinh chat cua gaussian function
	int number_of_scales = 10;
	vector<Mat> gaussianFilters(number_of_scales, Mat::zeros(source.size(), CV_32FC1));

	for (int i = 0; i < number_of_scales; ++i)
		gaussianFilters[i] = gaussianKernel(5, pow(k, i) * sigma, true);

	// hieu cua cac anh da lam mo
	vector<Mat> DoG(number_of_scales - 1, Mat::zeros(source.size(), CV_32FC1));


	vector<float> max_DoG_value(number_of_scales, 0);

	for (int i = 0; i < DoG.size(); ++i) {
		Mat conv1, conv2;

		filter2D(srcGray, conv1, CV_32FC1, gaussianFilters[i]);
		filter2D(srcGray, conv2, CV_32FC1, gaussianFilters[i + 1]);

		// thuc hien phep tru ma tran cho 2 anh da lam mo bang 2 gaussian filter voi sigma khac nhau
		Mat conv_result = matrixMinus(conv2, conv1);

		conv_result = matrixMultiply(conv_result, conv_result);

		max_DoG_value[i] = matrixMaxValue(conv_result);
		DoG[i] = conv_result;
	}

	// tim local extrema tuong tu nhu thuat toan Laplace Blob detection
	vector<Blob> blobs;
	for (int idx = 0; idx < DoG.size(); ++idx) {
		for (int y = 0; y < source.rows; ++y) {
			for (int x = 0; x < source.cols; ++x) {
				float val = getPixel(DoG[idx], y, x);
				if (val <= thresholdMax * max_DoG_value[idx])
					continue;

				bool foundPeak = true;
				for (int step_idx = -1; step_idx <= 1; ++step_idx) {
					if (foundPeak == false) 
						break;

					for (int step_x = -1; step_x <= 1; ++step_x) {
						if (foundPeak == false)
							break;

						for (int step_y = -1; step_y <= 1; ++step_y) {
							if (foundPeak == false)
								break;

							int cur_idx = idx + step_idx, cur_y = y + step_y, cur_x = x + step_x;
							if (cur_idx >= DoG.size() || cur_idx < 0)
								continue;
							if (cur_y >= source.rows || cur_y < 0)
								continue;
							if (cur_x >= source.cols || cur_x < 0)
								continue;

							if (val < getPixel(DoG[cur_idx], cur_y, cur_x))
								foundPeak = false;
						}
					}
				}

				if (foundPeak == true)
					blobs.push_back(Blob(pow(k, idx) * sigma, y, x));
			}
		}
	}


	return blobs;
}
void BlobDetector::showBlobsWithDoGDetector(const Mat& src, float sigma, float k, float threshold) {
	vector<Blob> blobs = detectDOG(src, sigma, k, threshold);

	Mat dst = src.clone();
	for (Blob blob : blobs)
		circle(dst, Point(blob._x, blob._y), blob._val * sqrt(2), Scalar(0, 0, 255), 2, 8, 0);
	
	
	imshow("Difference of Gaussian Blob Detection", dst);
	waitKey(0);
}