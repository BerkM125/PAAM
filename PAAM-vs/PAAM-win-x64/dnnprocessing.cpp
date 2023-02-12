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

extern cv::String PRIMARYWINDOW;

void loopProcessing(PersonPoseModel &person) {
	while (true) {
		person.forwardNet();
	}
}

PersonPoseModel::PersonPoseModel()
	: poseNet{ dnn::readNetFromCaffe(protoFile, weightsFile) } {
	keypointBuffer.reserve(TRACKINGPOINTS);
}

PersonPoseModel::PersonPoseModel(Mat poseFrame) 
	: poseFrame{ poseFrame.clone() }, poseNet{ dnn::readNetFromCaffe(protoFile, weightsFile) } {
	keypointBuffer.reserve(TRACKINGPOINTS);
}

PersonPoseModel::PersonPoseModel(Mat poseFrame, dnn::Net poseNet)
	: poseFrame{ poseFrame.clone() }, poseNet{ poseNet } {
	keypointBuffer.reserve(TRACKINGPOINTS);
}

PersonPoseModel::PersonPoseModel(Mat poseFrame, dnn::Net poseNet, vector<Point> loadingBuffer) 
	: poseFrame{ poseFrame.clone() }, poseNet{ poseNet }, keypointBuffer{ loadingBuffer } {
	keypointBuffer.reserve(TRACKINGPOINTS);
}

PersonPoseModel::PersonPoseModel(const PersonPoseModel& ppm) 
	: poseFrame{ ppm.poseFrame.clone() }, poseNet{ ppm.poseNet }, keypointBuffer{ ppm.keypointBuffer } {
	keypointBuffer.reserve(TRACKINGPOINTS);
}

void PersonPoseModel::loadPointBuffer(vector<Point> buffer) {
	keypointBuffer = buffer;
}

void PersonPoseModel::loadPoseFrame(Mat frame) {
	poseFrame = frame.clone();
}

void PersonPoseModel::loadNeuralNetwork(cv::String protoFile, cv::String weightsFile) {
	poseNet = dnn::readNetFromCaffe(protoFile, weightsFile);
}

void PersonPoseModel::loadNeuralNetwork(dnn::Net net) {
	poseNet = net;
}

// Load a Rect object with the ROI's dimensions
void PersonPoseModel::loadROI(Rect region) {
	estimationRegionInterest = region;
}

// Load a Rect object with the original or unprocessed frame's dimensions
void PersonPoseModel::loadOrigDimensions(Rect bounds) {
	originalRegion = bounds;
}

// Enable ROI shifting mode
void PersonPoseModel::enableROIMode() {
	regionalScalingMode = true;
}

// Disable ROI shifting mode
void PersonPoseModel::disableROIMode() {
	regionalScalingMode = false;
}

/* Pass the currently loaded Mat/frame in the object through the loaded
   estimation network. Store keypoints in private buffer.
*/
void PersonPoseModel::forwardNet() {
	Mat inputBlob;
	try {
		if (poseFrame.cols > 0 && poseFrame.rows > 0) {
			// Generate blob from poseFrame, pass into the network...
			inputBlob = dnn::blobFromImage(poseFrame, 1.0 / 255, Size(INPUTWIDTH, INPUTHEIGHT),
				Scalar(0, 0, 0), false, false);
		}
		else {
			throw poseFrame;
		}
	}
	catch (Mat m) {
		cout << "Input poseFrame empty, dimensions are: " << m.cols << "x" << m.rows << ", returning..." << endl;
		return;
	}

	poseNet.setInput(inputBlob);
	Mat netOut = poseNet.forward();

	int netH = netOut.size[2];
	int netW = netOut.size[3];

	int frameHeight = poseFrame.rows;
	int frameWidth = poseFrame.cols;

	// For EVERY BODY PART, make a confidence map.
	for (int bIndex = 0; bIndex < TRACKINGPOINTS; bIndex++) {
		Mat confidence(netH, netW, CV_32F, netOut.ptr(0, bIndex));

		// Point for the actual keypoint, point for the confidence map keypoints
		Point2f bodyPoint(-1, -1);
		Point mapPoint;
		double prob;

		minMaxLoc(confidence, 0, &prob, 0, &mapPoint);
		if (prob > CONFIDENCETHRESHOLD) {
			bodyPoint = mapPoint;

			// Scale keypoint from the map accordingly
			bodyPoint.y *= (double)frameHeight / netH;
			bodyPoint.x *= (double)frameWidth / netW;
		}

		if (regionalScalingMode) {
			// Surround scaling functionality with try...catch to check for invalid bounds exception
			try {
				if (estimationRegionInterest.width > 0 && estimationRegionInterest.height > 0) {
					// Calculate the shifts from original frame to bounding box
					double wFactor = estimationRegionInterest.x - originalRegion.x;
					double hFactor = estimationRegionInterest.y - originalRegion.y;
					// Fix the keypoints accordingly
					bodyPoint.x += wFactor;
					bodyPoint.y += hFactor;
				}
				else {
					throw(estimationRegionInterest);
				}
			}
			// Catch Rect exception and print dimensions
			catch (Rect region) {
				cout << "ROI error: Width and height set to zero";
				cout << "Rect dimensions - x: " << region.x << ", y: " << region.y
					<< ", W: " << region.width << ", H: " << region.height << endl;
			}
		}
		// Store keypoint into buffer
		keypointBuffer[bIndex] = bodyPoint;
	}
}

// Render a single line using two indices (0-14) of keypoints.
void PersonPoseModel::renderBodyLine(unsigned int pbIndex1, unsigned int pbIndex2) {
	if ((keypointBuffer[pbIndex1].x < 10 && keypointBuffer[pbIndex1].y < 10) ||
		(keypointBuffer[pbIndex2].x < 10 && keypointBuffer[pbIndex2].y < 10)) {
		return;
	}

	line(poseFrame, keypointBuffer[pbIndex1], keypointBuffer[pbIndex2], Scalar(255, 0, 0), 1, LINE_AA);
}

// Construct full body model
void PersonPoseModel::renderPoseLines() {
	renderBodyLine(0, 1);
	renderBodyLine(8, 9);
	renderBodyLine(11, 12);
	renderBodyLine(12, 13);
	renderBodyLine(9, 10);
	renderBodyLine(1, 2);
	renderBodyLine(1, 5);

	renderBodyLine(2, 3);
	renderBodyLine(3, 4);

	renderBodyLine(5, 6);
	renderBodyLine(6, 7);

	renderBodyLine(1, 8);
	renderBodyLine(1, 11);
}

/* Render all points and lines of the person's body after passing
*  frame through the network.
*/
void PersonPoseModel::renderPose() {
	int bIndex = 0;

	renderPoseLines();
	for (bIndex = 0; bIndex < TRACKINGPOINTS; bIndex++) {
		cv::Point bodyPoint = keypointBuffer[bIndex];
		// Draw circle around body part
		circle(poseFrame, Point((int)bodyPoint.x, (int)bodyPoint.y), 3, Scalar(255, 255, 0), -1);

		// Put text of the body part
		//putText(poseFrame, format("%d", bIndex),
			//Point((int)bodyPoint.x, (int)bodyPoint.y),
			//FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 2);
	}
	imshow(PRIMARYWINDOW, poseFrame);
}

void PersonPoseModel::renderPose(cv::String WINDOWNAME) {
	renderPose();
	imshow(WINDOWNAME, poseFrame);
}

Mat PersonPoseModel::getFrame() {
	return poseFrame;
}

PersonPoseModel::~PersonPoseModel() {
	keypointBuffer.clear();
}