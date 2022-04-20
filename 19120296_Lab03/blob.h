#pragma once

#ifndef BLOB_H
#define BLOB_H

#include "Utils.h"
#include <set>

class Blob {
public:
	int _x, _y;
	float _val;
public:
	Blob(float val, int y, int x) {
		_x = x;
		_y = y;
		_val = val;
	}
};


class BlobDetector {
	// implement ca 2 thuat toan LoG va DoG

private:
	vector<Blob> detectBlob(const Mat& src, float signma, float k, float threshold);
	vector<Blob> detectDOG(const Mat& src, float signma , float k , float threshold );

public:
	void showBlobsWithLoGDetector(const Mat& src, float signma, float k, float threshold);
	void showBlobsWithDoGDetector(const Mat& src, float signma, float k, float threshold);
};

#endif