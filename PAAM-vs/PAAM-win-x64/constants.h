#pragma once

#define FRAMEDURATION 30 // FPS
#define INPUTWIDTH 368 // Blob input image width. Width of blob frame passed into the network.
#define INPUTHEIGHT 368 // Blob input image height. Height of blob fram passed into the network
#define TRACKINGPOINTS 16 // Number of tracking points on person's body. In OpenPose there's 15
#define CONFIDENCETHRESHOLD 0// Confidence threshold value for the probability mapping stage.