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
def create_image_pyramid(image, levels=3):
    return [cv.pyrDown(image) if i > 0 else image for i in range(levels)]

# Deque to store templates
template_deque = deque(maxlen=5)
needle_pyramid = create_image_pyramid(needle_img)
template_deque.append(needle_img)

# Dynamic updating of templates
def update_template_deque(new_template):
    if new_template is not None:
        template_deque.append(new_template)
        return create_image_pyramid(new_template)

last_top_left = None
threshold = 0.4

while True:
    ret, frame = video.read()
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
        # Create a list from the deque to avoid mutation during iteration
        templates = list(template_deque)
        for needle in templates:
            needle_pyramid = create_image_pyramid(needle)  # Create pyramid for each needle
            for needle_scale in needle_pyramid:
                result = cv.matchTemplate(frame, needle_scale, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

                if max_val >= threshold:
                    print('Found needle at scale.')
                    last_top_left = max_loc
                    break

            if last_top_left:
                break

    # Limit the search area to twice the size of the needle image
    if last_top_left is not None:
        search_area_w = w * 2
        search_area_h = h * 2
        search_area_top_left = (max(0, last_top_left[0] - w), max(0, last_top_left[1] - h))
        search_area_bottom_right = (min(frame.shape[1], last_top_left[0] + search_area_w),
                                     min(frame.shape[0], last_top_left[1] + search_area_h))
        cv.rectangle(frame, search_area_top_left, search_area_bottom_right, color=(255, 0, 0), thickness=2)

        search_area = frame[search_area_top_left[1]:search_area_bottom_right[1],
                            search_area_top_left[0]:search_area_bottom_right[0]]

        # Perform template matching on the search area
        templates = list(template_deque)  # Create a list from the deque
        for needle in templates:
            needle_pyramid = create_image_pyramid(needle)
            for needle_scale in needle_pyramid:
                result = cv.matchTemplate(search_area, needle_scale, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
                adjusted_top_left = (search_area_top_left[0] + max_loc[0], search_area_top_left[1] + max_loc[1])

                if max_val >= threshold:
                    last_top_left = adjusted_top_left
                    cv.rectangle(frame, adjusted_top_left, (adjusted_top_left[0] + w, adjusted_top_left[1] + h),
                                 color=(0, 255, 255), thickness=2)

                    # Update template deque with the new template
                    new_template = search_area[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
                    update_template_deque(new_template)
                    break  # Stop after the first match found

    # Display the current frame
    cv.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv.destroyAllWindows()