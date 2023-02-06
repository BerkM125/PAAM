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
#include "personposemodel.h"

using namespace cv;
using namespace std;
extern void custom(Mat& frame);
extern void contourify(Mat& input, unsigned int lowerBound, unsigned int upperBound);
extern void contourify(Mat input, Mat& output, unsigned int lowerBound, unsigned int upperBound);