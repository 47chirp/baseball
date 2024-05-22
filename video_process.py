# take as an input a video file and break it down into frames
# highlight certain aspects of the video
# output a video file with the baseball pitch detected, and a percentile breakdown at each frame of the video

# questions, how do i identify who the pitcher is?

# assumption is the there exists data to show the type of pitch thrown

# ideally the goal will be to build a model that can analyze the position of the player's arm position and the ball's position to determine the type of pitch thrown

# the model will be trained on a dataset of pitches thrown by a specific player

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# training data process

# the link is a pitching compilation, tagging either right before pitch or right after pitch
https://www.youtube.com/watch?v=D3ziNCxNnWs&t=11s
pre-pitch times:

motion times:
12 seconds

end of motion times:
