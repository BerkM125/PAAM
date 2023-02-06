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
#include "preprocesslib.h"

using namespace cv;
using namespace std;

void custom(Mat& frame) {
	contourify(frame, 100, 179);
}
/* Simplified contour detection, creates and uses a mask of colors
   with a set range of saturation and value, the int inputs only pertain
   to lower and upper bounds for hue values. Modifies the original frame. */
void contourify(Mat &input, unsigned int lowerBound, unsigned int upperBound) {
	vector<vector<Point>> contourBuffer;
	Mat modifierMat;
	Mat masking;

	// Convert color space
	cvtColor(input, modifierMat, COLOR_BGR2HSV);
	
	// Mask
	inRange(modifierMat, Scalar(lowerBound, 0, 0), Scalar(upperBound, 255, 255), masking);
	
	erode(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	dilate(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	//morphological closing
	dilate(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	erode(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	findContours(masking, contourBuffer, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (int cIndex = 0; cIndex < contourBuffer.size(); cIndex++) {
		drawContours(input, contourBuffer, cIndex, Scalar(255, 255, 255), 1, LINE_AA);
	}
}

/* Simplified contour detection, creates and uses a mask of colors
   with a set range of saturation and value, the int inputs only pertain
   to lower and upper bounds for hue values. Draws contours on an output frame. */
void contourify(Mat input, Mat& output, unsigned int lowerBound, unsigned int upperBound) {
	vector<vector<Point>> contourBuffer;
	Mat modifierMat = input.clone();
	Mat masking = input.clone();

	// Convert color space
	cvtColor(input, modifierMat, COLOR_BGR2HSV);

	// Mask
	inRange(modifierMat, masking, Scalar(lowerBound, 0, 0), Scalar(upperBound, 255, 255));
	findContours(masking, contourBuffer, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (int cIndex = 0; cIndex < contourBuffer.size(); cIndex++) {
		drawContours(output, contourBuffer, cIndex, Scalar(255, 255, 255), 1, LINE_AA);
	}
}