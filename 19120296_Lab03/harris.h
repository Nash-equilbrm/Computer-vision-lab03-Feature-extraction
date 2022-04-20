#pragma once

#ifndef HARRIS_H
#define HARRIS_H

#include "utils.h"
#include <vector>


class Corner {
public:
	int _x, _y;
	float _r;
public:
	Corner(float r, int y, int x) {
		_r = r;
		_x = x;
		_y = y;
	}
	// overload vi su dung vector de luu tru cac corner 
	bool operator < (const Corner& other) {
		return _r < other._r;
	}
};


class HarrisCornerDetector {
private:
	vector<Corner> detectHarris(const Mat& src, float k, float alpha);
public:
	void showCorners(const Mat& src, float k , float alpha );
};

#endif