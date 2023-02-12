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
#include <math.h>
#include "constants.h"
#include "personposemodel.h"
#include "preprocesslib.h"

#define BOUNDINGBOX_MARGIN 500
#define BOXASSERTION_THRESHOLD 5

using namespace cv;
using namespace std;

vector<Rect> globalRegionBuffer(1);

// This is just my custom preprocessing function; change it to whatever you need.
void custom(Mat& frame) {
	cropContourBounds(frame, frame, 0, 20);
}

// Assert rectangle coordinates, make sure nothing exceeds window bounds
void assertBox(Rect &r, Mat parentFrame) {
	r.x = (r.x < BOXASSERTION_THRESHOLD) ? BOXASSERTION_THRESHOLD : r.x;
	r.y = (r.y < BOXASSERTION_THRESHOLD) ? BOXASSERTION_THRESHOLD : r.y;
	r.width = (r.width + r.x > parentFrame.cols) 
				? parentFrame.cols - (r.x + BOXASSERTION_THRESHOLD) : r.width;
	r.height = (r.height + r.y > parentFrame.rows) 
				? parentFrame.rows - (r.y + BOXASSERTION_THRESHOLD) : r.height;

}

/* Simplified contour detection, creates and uses a mask of colors
   with a set range of saturation and value, the int inputs only pertain
   to lower and upper bounds for hue values. Modifies the original frame. */
Rect contourify(Mat &input, unsigned int lowerBound, unsigned int upperBound) {
	contourify(input, input, lowerBound, upperBound);
}

/* Simplified contour detection, creates and uses a mask of colors
   with a set range of saturation and value, the int inputs only pertain
   to lower and upper bounds for hue values. Draws contours on an output frame. */
Rect contourify(Mat input, Mat& output, unsigned int lowerBound, unsigned int upperBound) {
	vector<vector<Point>> contourBuffer;
	Mat modifierMat;
	Mat masking;
	Point upperLeft(INT_MAX, INT_MAX);
	Point lowerRight(-1, -1);
	Rect contourOutline;

	// Convert color space
	cvtColor(input, modifierMat, COLOR_BGR2HSV);

	// Mask
	inRange(modifierMat, Scalar(lowerBound, 190, 190), Scalar(upperBound, 255, 255), masking);
	
	// Morphological opening and closing
	erode(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	dilate(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	dilate(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	erode(masking, masking, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	// Find contours and store in the contour buffer
	findContours(masking, contourBuffer, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// Draw the contours and construct a contour bounding box
	for (int cIndex = 0; cIndex < contourBuffer.size(); cIndex++) {
		// Make sure to put the minimum coordinate into the upperLeft Point object
		upperLeft.x = min(contourBuffer[cIndex][0].x, upperLeft.x);
		upperLeft.y = min(contourBuffer[cIndex][0].y, upperLeft.y);

		// And vice versa into the lowerRight Point
		lowerRight.x = max(contourBuffer[cIndex][0].x, lowerRight.x);
		lowerRight.y = max(contourBuffer[cIndex][0].y, lowerRight.y);

		// Draw contours
		drawContours(output, contourBuffer, cIndex, Scalar(255, 255, 255), 1, LINE_AA);

		// If both points are valid, store coordinates into the bounding box object
		if (upperLeft.x != INT_MAX && upperLeft.y != INT_MAX &&
			lowerRight.x != -1 && lowerRight.y != -1) {
			contourOutline = Rect(upperLeft.x, upperLeft.y,
				lowerRight.x - upperLeft.x, lowerRight.y - upperLeft.y);
		}
	}

	/*if (upperLeft.x != INT_MAX && upperLeft.y != INT_MAX &&
		lowerRight.x != -1 && lowerRight.y != -1) {
		rectangle(output, contourOutline, Scalar(255, 255, 255), 2);
	}*/
	return contourOutline;

}

void cropContourBounds(Mat input, Mat &output, unsigned int lowerBound, unsigned int upperBound) {
	double dWidth, dHeight;
	double boxMarginFactor = (BOUNDINGBOX_MARGIN / 100.0f);

	Rect boundingBox = contourify(input, output, lowerBound, upperBound);
	Rect copyBox = boundingBox;

	// Enlarge the box a bit to encompass more than just the subject
	boundingBox.width *= 2;
	boundingBox.height *= 2;

	dWidth = boundingBox.width - copyBox.width;
	dHeight = boundingBox.height - copyBox.height;

	boundingBox.x -= dWidth;
	boundingBox.y -= dHeight;

	boundingBox.width += dWidth;
	boundingBox.height += dHeight;
	
	// Justify bounding box within the frame, make sure it does not exceed window bounds
	assertBox(boundingBox, input);

	rectangle(output, boundingBox, Scalar(255, 255, 255), 2);
	output = input(boundingBox);

	globalRegionBuffer[0] = boundingBox;
}