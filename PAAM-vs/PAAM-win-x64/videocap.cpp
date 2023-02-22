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
#include "preprocesslib.h"

// Before ANY preprocessing, useful to crop out extreme noise that's hard to filter through
#define RAWWIDTH_SCALE 0.8
#define RAWHEIGHT_SCALE 1.0

using namespace cv;

extern cv::String PRIMARYWINDOW;
cv::String ALTERNATEWINDOW;
extern void loopProcessing(PersonPoseModel &person);

// DEMONSTRATIONAL MULTI-PERSON TRACKING DEMO. MADE TO TRACK TWO PLAYERS IN A SET VIDEO. JUST AN EXAMPLE
void trackOneOnOne(cv::String filename) {
	VideoCapture video(filename);
	PersonPoseModel person1;
	PersonPoseModel person2;

	Mat upperFrame;
	Mat lowerFrame;

	bool frameSuccess = true;

	while (frameSuccess) {
		Mat currFrame;
		frameSuccess = video.read(currFrame);

		if (!frameSuccess) {
			break;
		}

		currFrame = currFrame(Rect(0, 0, currFrame.cols * RAWWIDTH_SCALE, currFrame.rows * RAWHEIGHT_SCALE));
		
		// Establish regions of interest for the upper court and lwoer court, obviously scaling is different due to camera angles
		Rect upperFrameROI(115, currFrame.rows*0.2, currFrame.cols-115, currFrame.rows * 0.45 - currFrame.rows * 0.2);
		Rect lowerFrameROI(0, currFrame.rows * 0.45, currFrame.cols, currFrame.rows - currFrame.rows*0.45);

		upperFrame = currFrame(upperFrameROI);
		lowerFrame = currFrame(lowerFrameROI);

		// Render ROIs
		rectangle(currFrame, upperFrameROI, Scalar(255, 255, 255), 2);
		rectangle(currFrame, lowerFrameROI, Scalar(255, 255, 255), 2);

		// Load raw frame into the pose estimator
		person1.poseFrame = lowerFrame;
		person1.preprocessFrame(defaultPreprocess);
		person1.forwardNet();

		person2.poseFrame = upperFrame;
		person2.preprocessFrame(defaultPreprocess);
		person2.forwardNet();

		// Render both model poses
		person1.renderPose();
		person2.renderPose();
		imshow(ALTERNATEWINDOW, currFrame);
		if (waitKey(30) == 27) break;
	}
	video.release();
}

void processVideo(cv::String filename) {
	VideoCapture video(filename);
	PersonPoseModel mainPerson;

	bool frameSuccess = true;

	// Establish video frame dimensions
	int frame_width = static_cast<int>(video.get(3) * RAWWIDTH_SCALE);
	int frame_height = static_cast<int>(video.get(4) * RAWHEIGHT_SCALE);

	Size frame_size(frame_width, frame_height);
	int fps = 30;
	//Initialize video writer object
	VideoWriter output("output.mp4", VideoWriter::fourcc('m','p','4','v'),
		fps, frame_size);

	while (frameSuccess) {
		Mat rawFrame;
		frameSuccess = video.read(rawFrame);
		if (!frameSuccess) {
			break;
		}
		// Before preprocessing
		rawFrame = rawFrame(Rect(0, 0, rawFrame.cols * RAWWIDTH_SCALE, rawFrame.rows * RAWHEIGHT_SCALE));
		// Load raw frame into the pose estimator
		mainPerson.poseFrame = rawFrame;

		// Pass custom preprocessor
		mainPerson.preprocessFrame(custom);

		// Load the ROI, original image dimensions for shifting, and pass image to the neural network.
		mainPerson.loadROI(globalRegionBuffer[0]);
		mainPerson.loadOrigDimensions(Rect(0, 0, rawFrame.cols, rawFrame.rows));
		mainPerson.enableROIMode();

		mainPerson.forwardNet();
		mainPerson.poseFrame = rawFrame;
		
		mainPerson.renderPose();

		// Write to video if possible; check for bad frames, because they will corrupt footage.
		try {
			if (mainPerson.poseFrame.cols > 0 && mainPerson.poseFrame.rows > 0) {
				output.write(mainPerson.poseFrame);
			}
			else {
				throw mainPerson.poseFrame;
			}
		}
		catch (Mat m) {
			cout << "ERROR: poseFrame empty, dimensions are: " << m.cols << "x" << m.rows << ", continuing..." << endl;
		}


		if (waitKey(30) == 27) break;
	}
	video.release();
	output.release();
}

void videoLoop(void) {
	PersonPoseModel mainPerson;
	VideoCapture camera(0);
	
	camera.read(mainPerson.poseFrame);

	thread processingThread(loopProcessing, std::ref(mainPerson));
	processingThread.detach();
	// Pass frame through the network only every 5 seconds.
	// Adjust by FPS: If at 30 frames per second, pass thru network with every 30*5 = 150 frames
	while (true) {
		camera.read(mainPerson.poseFrame);

		mainPerson.renderPose();
		if (cv::waitKey(30) == 27) { //Escape upon ESC key being pressed
			break;
		}
	}
	
}