#pragma once

#ifndef BLOB_H
#define BLOB_H

#include "Utils.h"
#include <set>

class BlobDetector {
	// implement ca 2 thuat toan LoG va DoG

private:
	set<tuple<int, int, float>> detectBlob(const Mat& src, float signma, float k, float threshold);
	set<tuple<int, int, float>> detectDOG(const Mat& src, float signma , float k , float threshold );

public:
	void showBlobsWithLoGDetector(const Mat& src, float signma, float k, float threshold);
	void showBlobsWithDoGDetector(const Mat& src, float signma, float k, float threshold);
};

#endif