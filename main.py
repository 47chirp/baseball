import pandas as pd
import cv2
import os
import numpy as np
from pytube import YouTube
import tensorflow as tf
import tensorflow_hub as hub

# Load pitch data
pitch_data_path = "baseball-cardsVSdodgers.xlsx"
column_names = [
    'Team', 'Pitcher', 'Pitch Number', 'Speed', 'Pitch Type', 'Result',
    'Pre_Pitch Time (min)', 'Pitch Motion Start (min)', 'Pitch Motion Stop (min)',
    'Time (min)', 'PrePitchSec', 'MotionStartSec', 'MotionStopSec', 'PlayerNumber', 'JerseyColor'
]
pitch_data = pd.read_excel(pitch_data_path, header=None, skiprows=2, names=column_names)
pitch_data['PrePitchSec'] = pd.to_numeric(pitch_data['PrePitchSec'], errors='coerce')
pitch_data['MotionStartSec'] = pd.to_numeric(pitch_data['MotionStartSec'], errors='coerce')
pitch_data['MotionStopSec'] = pd.to_numeric(pitch_data['MotionStopSec'], errors='coerce')

# Download and load the video
url = "https://www.youtube.com/watch?v=NbpNf097nqE"
video = YouTube(url)
stream = video.streams.get_highest_resolution()
video_path = stream.download()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
main_folder = "cardsVSdodgersPitches"
os.makedirs(main_folder, exist_ok=True)

# Load MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

# Function to preprocess the image for pose estimation
def preprocess_image(frame):
    image = tf.image.convert_image_dtype(frame, tf.uint8)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_with_pad(image, 256, 256)
    image = tf.cast(image, dtype=tf.int32)
    return image

# Function to run pose estimation on an image
def run_pose_estimation(image):
    outputs = movenet(image)
    return outputs['output_0'].numpy()[0]

# Function to draw keypoints on the frame
def draw_keypoints(frame, keypoints, pitcher_name, team):
    keypoints = np.squeeze(keypoints)
    # Draw keypoints
    for keypoint in keypoints:
        y, x, confidence = keypoint
        if confidence > 0.5:
            x = int(x * frame.shape[1])
            y = int(y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    # Annotate pitcher's name and team
    cv2.putText(frame, f"Pitcher: {pitcher_name}, Team: {team}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def extract_and_annotate_frames(video_path, pitch_data, main_folder, fps):
    for index, row in pitch_data.iterrows():
        if pd.isna(row['MotionStartSec']) or pd.isna(row['MotionStopSec']):
            continue
        pitcher_name = row['Pitcher']
        team = row['Team']
        pitch_folder = os.path.join(main_folder, f"Pitch_{index+1}")
        os.makedirs(pitch_folder, exist_ok=True)
        start_frame = int(row['MotionStartSec'] * fps)
        stop_frame = int(row['MotionStopSec'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= stop_frame:
            ret, frame = cap.read()
            if not ret:
                break
            image = preprocess_image(tf.convert_to_tensor(frame, dtype=tf.uint8))
            keypoints = run_pose_estimation(image)  # Compute keypoints for each frame
            frame_with_keypoints = draw_keypoints(frame, keypoints, pitcher_name, team)
            cv2.putText(frame_with_keypoints, f"Pitch Type: {row['Pitch Type']}, Speed: {row['Speed']} mph", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_filename = os.path.join(pitch_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}s.jpg")
            cv2.imwrite(frame_filename, frame_with_keypoints)
    cap.release()

# Run the extraction and annotation
extract_and_annotate_frames(video_path, pitch_data, main_folder, fps)

'''
import pandas as pd
import cv2
import os
import numpy as np
import pytesseract
from pytesseract import Output
from pytube import YouTube
import tensorflow as tf
import tensorflow_hub as hub

# Load pitch data without assuming headers
pitch_data_path = "baseball-cardsVSdodgers.xlsx"
# manually define column names
column_names = [
    'Team', 'Pitcher', 'Pitch Number', 'Speed', 'Pitch Type', 'Result',
    'Pre_Pitch Time', 'Pitch Motion Start', 'Pitch Motion Stop',
    'Time', 'PrePitchSec', 'MotionStartSec', 'MotionStopSec', 'PlayerNumber', 'JerseyColor'
]
# the first two rows are not part of the data, so we skip them
pitch_data = pd.read_excel(pitch_data_path, header=None, skiprows=[0, 1], names=column_names)

# Convert columns to numeric as needed
pitch_data['PrePitchSec'] = pd.to_numeric(pitch_data['PrePitchSec'], errors='coerce')
pitch_data['MotionStartSec'] = pd.to_numeric(pitch_data['MotionStartSec'], errors='coerce')
pitch_data['MotionStopSec'] = pd.to_numeric(pitch_data['MotionStopSec'], errors='coerce')

# Download and load the video
url = "https://www.youtube.com/watch?v=NbpNf097nqE"
video = YouTube(url)
stream = video.streams.get_highest_resolution()
video_path = stream.download()

# Open the video file
cap = cv2.VideoCapture(video_path)
# Get the frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# Storing the frames in a folder
main_folder = "cardsVSdodgersPitches"
os.makedirs(main_folder, exist_ok=True)

# Load MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

# Function to preprocess the image for pose estimation
def preprocess_image(frame):
    image = tf.image.convert_image_dtype(frame, tf.uint8)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.cast(image, dtype=tf.int32)
    return image

# Function to run pose estimation on an image
def run_pose_estimation(image):
    outputs = movenet(image)
    return outputs['output_0'].numpy()[0]

# Function to identify pitcher's color and player number
def identify_pitcher(frame, player_number, team):
    # Define ROI for pitcher's expected location
    roi_x = 630
    roi_y_top = 350
    roi_y_bottom = 930 - 220
    roi_width = 225

    # Crop the region of interest (ROI) from the frame
    roi_frame = frame[roi_y_top:roi_y_bottom, roi_x:roi_x+roi_width]

    # Convert ROI to grayscale and apply Gaussian Blur to reduce noise
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Apply adaptive thresholding to enhance the number
    thresh_roi = cv2.adaptiveThreshold(blur_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Use OCR to read the player's number
    custom_config = r'--oem 3 --psm 6'  # psm 6 tends to work well for a single uniform block of text
    ocr_result = pytesseract.image_to_string(thresh_roi, config=custom_config)

    # Clean OCR results to remove non-numeric characters
    extracted_number = ''.join(filter(str.isdigit, ocr_result))

    # Validate extracted number against expected player number
    if extracted_number and extracted_number == str(player_number):
        # If the number matches, draw a rectangle and label the frame to indicate a successful match
        cv2.rectangle(frame, (roi_x, roi_y_top), (roi_x + roi_width, roi_y_bottom), (0, 255, 0), 2)
        cv2.putText(frame, extracted_number, (roi_x, roi_y_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, extracted_number

def extract_and_annotate_frames(video_path, pitch_data, main_folder, fps):
    for index, row in pitch_data.iterrows():
        if pd.isna(row['MotionStartSec']) or pd.isna(row['MotionStopSec']):
            continue
        pitcher_name = row['Pitcher']
        team = row['Team']
        player_number = row['PlayerNumber']
        pitch_folder = os.path.join(main_folder, f"Pitch_{str(index+1)}")
        os.makedirs(pitch_folder, exist_ok=True)
        start_frame = int(row['MotionStartSec'] * fps)
        stop_frame = int(row['MotionStopSec'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= stop_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame, player_number = identify_pitcher(frame, player_number, team)
            cv2.putText(frame, f"Pitcher: {pitcher_name}, Team: {team}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Player Number: {player_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch Type: {row['Pitch Type']}, Speed: {row['Speed']} mph", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_filename = os.path.join(pitch_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)}s.jpg")
            cv2.imwrite(frame_filename, frame)
    cap.release()

# Run the extraction and annotation
extract_and_annotate_frames(video_path, pitch_data, main_folder, fps)
'''