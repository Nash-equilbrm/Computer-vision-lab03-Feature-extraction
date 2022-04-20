#include "utils.h"
#include "harris.h"
#include "blob.h"


void execute(int argc, const vector<string>& argv);
void printHelp();


int main(int argc, char** argv) {
	vector<string> argv_str;
	for (int i = 0; i < argc; ++i) {
		argv_str.push_back(argv[i]);
	}

	execute(argc, argv_str);
	
	return 0;

	
}


void execute(int argc, const vector<string>& argv) {
	if (argc < 2) {
		printHelp();
		return;
	}

	Mat src = imread(argv[1], IMREAD_COLOR);
	if (src.empty()) {
		cout << "Cannot open image. Please check image directory!";
		return;
	}

	if (argc == 2) {
		imshow("Source image", src);
		waitKey(0);
		return;
	}

	if (argv[2] == "1") {// thuat toan Harris corner detection
		imshow("Source image", src);
		waitKey(0);
		float k = 0.05, alpha = 0.01;
		if (argv.size() >= 4)
			k = stof(argv[3]);
		if (argv.size() >= 5)
			alpha = stof(argv[4]);
		HarrisCornerDetector detector;
		detector.showCorners(src, k, alpha);
		return;
	}


	if (argv[2] == "3" || argv[2] == "4") { // thuat toan  Laplace blob detection
		imshow("Source image", src);
		waitKey(0);
		BlobDetector detector;
		float k = sqrt(2), sigma = 1.0f, threshold = 0.3f;
		if (argv.size() >= 4)
			threshold = stof(argv[3]);
		if (argv.size() >= 5)
			sigma = stof(argv[4]);
		if (argv.size() >= 6)
			k = stof(argv[5]);
		if (argv[2] == "3") {
			detector.showBlobsWithLoGDetector(src, sigma, k, threshold);
			return;
		}
		else {
			detector.showBlobsWithDoGDetector(src, sigma, k, threshold);
			return;
		}
	}

}




void printHelp() {
	cout << "Help";
}