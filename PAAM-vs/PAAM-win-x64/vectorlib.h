#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>

#include "constants.h"

using namespace cv;
using namespace std;

class Vector2f {
public:
	double x = 0.0;
	double y = 0.0;

	Vector2f();
	Vector2f(double x, double y);
};

class Vector3f {
public:
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

	Vector3f();
	Vector3f(double x, double y, double z);
};

class KinematicModel {
public:
	unsigned int bufferSize = 0;

	vector<Point> frameCoordinates;
	vector<Vector2f> windowCoordinates;
	vector<Vector3f> spatialCoordinates;

	void generateDepthData(void);
	void generateDepthData(vector<Vector2f> flatCoordinates);
	void generateDepthData(vector<Point> flatCoordinates);

	void pushFramePoint(Point flatCoordinate);
	void pushFramePoint(Vector2f flatCoordinate);
};