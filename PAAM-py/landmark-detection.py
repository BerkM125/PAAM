# Goal: convert footage of people from livestream into real coordinate space data using
# MediaPipe pose landmark detection

# Import necessary modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np  # for converting frame to array for MP Image object
import cv2 as cv  # for capturing livestream with camera

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# download model to local directory 
# (https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models)
model_path = 'pose_landmarker_full.task'

results = []
RESULT = None

# To annotate a frame
def draw_landmarks_on_image(rgb_image):
  pose_landmarks_list = RESULT.pose_landmarks
  annotated_frame = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_frame,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_frame


# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  # print('pose landmarker result: {}'.format(result))
  results.append(result)
  global RESULT
  RESULT = result

# Set up configuration options
# what should num_poses be? (default is 1)
options = PoseLandmarkerOptions(
  base_options=BaseOptions(model_asset_path=model_path),
  running_mode=VisionRunningMode.LIVE_STREAM,
  result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  # Start capturing from camera
  cam = cv.VideoCapture(0)  # index specifies camera (alternatively, can pass in video file name)
  if not cam.isOpened():
    print("unable to open camera")

  # Create a loop to read the latest frame from the camera using VideoCapture#read()
  while True:
    # capture video frame
    ret, frame = cam.read()

    # check if frame was read correctly
    if not ret:
      print("error capturing frame")
      break

    # Get the timestamp of the frame
    timestamp = cam.get(cv.CAP_PROP_POS_MSEC)

    # to break out of loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the frame
    cv.imshow('Webcam', frame)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))  # np.array() necessary?

    # RUN TASK  
    # Send live image data to perform pose landmarking.
    # The results are accessible via the `result_callback` provided in `PoseLandmarkerOptions` object.
    landmarker.detect_async(mp_image,int(timestamp))

    # VISUALIZE DETECTION RESULT -- WORK IN PROGRESS
    if type(RESULT) is not type(None):
      annotated_frame = draw_landmarks_on_image(mp_image.numpy_view())
      cv.imshow('Frame', annotated_frame)

  cam.release()
  cv.destroyAllWindows()

print("num frames saved: ", len(results))


# TODOS:
  # make markers between poses where player assumes neutral position