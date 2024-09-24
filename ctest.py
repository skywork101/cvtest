import time
import cv2 as cv
import numpy as np
from collections import deque

# Open the video file
video = cv.VideoCapture('test.mp4')
video.set(cv.CAP_PROP_FPS, 10)

# Set the desired resolution (720p)
width = 1280
height = 720
frame_count = 0
start_time = time.time()

# Desired FPS
target_fps = 30
frame_delay = int(1000 / target_fps)

# Read the first frame to select ROI
ret, frame = video.read()
frame = cv.resize(frame, (width, height))
if not ret:
    print("Failed to read video")
    video.release()
    exit()

# Display the first frame to select the needle (ROI)
cv.imshow('Select ROI', frame)
x, y, w, h = cv.selectROI('Select ROI', frame, fromCenter=False, showCrosshair=True)
cv.destroyWindow('Select ROI')

# Crop the selected ROI from the first frame
needle_img = frame[y:y+h, x:x+w].copy()

# Create image pyramid for the needle
# def create_image_pyramid(image, levels=3):
#     return [cv.pyrDown(image) if i > 0 else image for i in range(levels)]


def create_image_pyramid(image, levels=4):
    pyramid = [image]  # Start with the original image
    for i in range(1, levels):
        downsampled = cv.pyrDown(pyramid[i - 1])  # Downsample the last image
        pyramid.append(downsampled)

    # Optionally, create upsampled images from the downsampled images
    for i in range(levels - 1):
        upsampled = cv.pyrUp(pyramid[i + 1])  # Upsample the next image
        pyramid.append(upsampled)

    return pyramid

def visualize_image_pyramid(pyramid):
    # Create a blank image to stack the pyramid images
    total_width = max(img.shape[1] for img in pyramid)
    total_height = sum(img.shape[0] for img in pyramid)
    stacked_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    current_y = 0
    for img in pyramid:
        stacked_image[current_y:current_y + img.shape[0], :img.shape[1]] = img
        current_y += img.shape[0]

    cv.imshow('Image Pyramid', stacked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


needle_pyramid = create_image_pyramid(needle_img)
visualize_image_pyramid(needle_pyramid)

# Deque to store templates
template_deque = deque(maxlen=10)
template_deque.append(needle_img)

last_top_left = None
threshold = 0.8

while True:
    ret, frame = video.read()

    # Check if frame is successfully read
    if not ret or frame is None:
        print("No more frames or failed to read frame.")
        break

    frame_count += 1

    # FPS calculation every second
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = time.time()
        frame_count = 0

    frame = cv.resize(frame, (width, height))

    # Perform template matching if no last top left established
    if last_top_left is None:
        for needle in needle_pyramid:
            result = cv.matchTemplate(frame, needle, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            if max_val >= threshold:
                print('Found needle at scale.')
                last_top_left = max_loc
                break

        if last_top_left:
            needle_h, needle_w = needle.shape[:2]
            bottom_right = (last_top_left[0] + needle_w, last_top_left[1] + needle_h)
            cv.rectangle(frame, last_top_left, bottom_right, color=(0, 255, 0), thickness=2)

    # Limit the search area to twice the size of the needle image
    if last_top_left is not None:
        # Define the search area
        search_area_w = needle_w * 2
        search_area_h = needle_h * 2
        search_area_top_left = (max(0, last_top_left[0] - needle_w), max(0, last_top_left[1] - needle_h))
        search_area_bottom_right = (min(frame.shape[1], last_top_left[0] + search_area_w),
                                     min(frame.shape[0], last_top_left[1] + search_area_h))
        cv.rectangle(frame, search_area_top_left, search_area_bottom_right, color=(255, 0, 0), thickness=2)

        search_area = frame[search_area_top_left[1]:search_area_bottom_right[1],
                            search_area_top_left[0]:search_area_bottom_right[0]]

        # Perform template matching on the search area
        for needle in needle_pyramid:
            result = cv.matchTemplate(search_area, needle, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            adjusted_top_left = (search_area_top_left[0] + max_loc[0], search_area_top_left[1] + max_loc[1])

            if max_val >= threshold:
                last_top_left = adjusted_top_left
                cv.rectangle(frame, adjusted_top_left, (adjusted_top_left[0] + needle_w, adjusted_top_left[1] + needle_h),
                             color=(0, 255, 255), thickness=2)
                break  # Stop after the first match found

    # Display the current frame
    cv.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv.destroyAllWindows()