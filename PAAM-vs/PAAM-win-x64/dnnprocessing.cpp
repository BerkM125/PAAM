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

PersonPoseModel::PersonPoseModel() {
	keypointBuffer.resize(TRACKINGPOINTS);
	poseNet = dnn::readNetFromCaffe(protoFile, weightsFile);
}

PersonPoseModel::PersonPoseModel(Mat poseFrame) {
	keypointBuffer.resize(TRACKINGPOINTS);
	this->poseFrame = poseFrame.clone();
	poseNet = dnn::readNetFromCaffe(protoFile, weightsFile);
}

PersonPoseModel::PersonPoseModel(Mat poseFrame, dnn::Net poseNet) {
	keypointBuffer.resize(TRACKINGPOINTS);
	this->poseFrame = poseFrame.clone();
	this->poseNet = poseNet;
}

PersonPoseModel::PersonPoseModel(Mat poseFrame, dnn::Net poseNet, vector<Point> loadingBuffer) {
	keypointBuffer.resize(TRACKINGPOINTS);
	this->poseFrame = poseFrame.clone();
	this->poseNet = poseNet;
	this->keypointBuffer = loadingBuffer;
}

PersonPoseModel::PersonPoseModel(const PersonPoseModel& ppm) {
	keypointBuffer.resize(TRACKINGPOINTS);
	this->poseFrame = ppm.poseFrame.clone();
	this->poseNet = ppm.poseNet;
	this->keypointBuffer = ppm.keypointBuffer;
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

void PersonPoseModel::forwardNet() {
	Mat inputBlob = dnn::blobFromImage(poseFrame, 1.0 / 255, Size(INPUTWIDTH, INPUTHEIGHT),
		Scalar(0, 0, 0), false, false);
	
	poseNet.setInput(inputBlob);
	Mat netOut = poseNet.forward();

	int netH = netOut.size[2];
	int netW = netOut.size[3];

	int frameHeight = poseFrame.rows;
	int frameWidth = poseFrame.cols;

	// For EVERY BODY PART, make a confidence map.
	for (int bIndex = 0; bIndex < TRACKINGPOINTS; bIndex++) {
		Mat confidence(netH, netW, CV_32F, netOut.ptr(0, bIndex));

		Point2f bodyPoint(-1, -1);
		Point mapPoint;

		double prob;

		minMaxLoc(confidence, 0, &prob, 0, &mapPoint);

		if (prob > CONFIDENCETHRESHOLD) {
			bodyPoint = mapPoint;

			bodyPoint.y *= (double)frameHeight / netH;
			bodyPoint.x *= (double)frameWidth / netW;
		}

		keypointBuffer[bIndex] = bodyPoint;
	}
}

void PersonPoseModel::forwardNet(Mat outputFrame) {

}

void PersonPoseModel::renderBodyLine(unsigned int pbIndex1, unsigned int pbIndex2) {
	if ((keypointBuffer[pbIndex1].x < 10 && keypointBuffer[pbIndex1].y < 10) ||
		(keypointBuffer[pbIndex2].x < 10 && keypointBuffer[pbIndex2].y < 10)) {
		return;
	}
	line(poseFrame, keypointBuffer[pbIndex1], keypointBuffer[pbIndex2], Scalar(255, 0, 0), 2, LINE_AA);
}

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

void PersonPoseModel::renderPose() {
	int bIndex = 0;

	renderPoseLines();
	for (bIndex = 0; bIndex < TRACKINGPOINTS; bIndex++) {
		cv::Point bodyPoint = keypointBuffer[bIndex];
		// Draw circle around body part
		circle(poseFrame, Point((int)bodyPoint.x, (int)bodyPoint.y), 5, Scalar(255, 255, 0), -1);

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