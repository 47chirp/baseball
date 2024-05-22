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
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
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
