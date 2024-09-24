import time
import cv2 as cv
import numpy as np

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

# Crop the selected ROI from the first frame and make a deep copy
needle_img = frame[y:y+h, x:x+w].copy()

# Process each frame in the video
last_top_left = None
threshold = 0.4

while True:
    ret, frame = video.read()

    # Check if frame is successfully read
    if not ret or frame is None:
        print("No more frames or failed to read frame.")
        break  # Exit if there are no more frames

    frame_count += 1  # Increment frame count

    # FPS calculation every second
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # Calculate FPS every second
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = time.time()  # Reset start time
        frame_count = 0  # Reset frame count

    frame = cv.resize(frame, (width, height))

    # Load the saved needle image for matching
    needle_h, needle_w = needle_img.shape[:2]  # Get dimensions of the needle image

    # Perform template matching with the static needle image if no search area established
    if last_top_left is None:
        result = cv.matchTemplate(frame, needle_img, cv.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        print('Best match top left position: %s' % str(max_loc))
        print('Best match confidence: %s' % max_val)

        if max_val >= threshold:
            print('Found needle.')
            last_top_left = max_loc  # Update last matched position

            # Draw a rectangle around the matched template
            bottom_right = (last_top_left[0] + needle_w, last_top_left[1] + needle_h)
            cv.rectangle(frame, last_top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    # Limit the search area to twice the size of the needle image
    if last_top_left is not None:
        search_area_w = needle_w * 2
        search_area_h = needle_h * 2

        # Define the search area centered at the last matched position
        search_area_top_left = (max(0, last_top_left[0] - needle_w), max(0, last_top_left[1] - needle_h))
        search_area_bottom_right = (min(frame.shape[1], last_top_left[0] + search_area_w),
                                     min(frame.shape[0], last_top_left[1] + search_area_h))

        # Draw the search area rectangle
        cv.rectangle(frame, search_area_top_left, search_area_bottom_right, color=(255, 0, 0), thickness=2, lineType=cv.LINE_4)

        # Crop the search area from the frame
        search_area = frame[search_area_top_left[1]:search_area_bottom_right[1],
                            search_area_top_left[0]:search_area_bottom_right[0]]

        # Check if the search area is valid and larger than the needle image
        if search_area.shape[0] >= needle_h and search_area.shape[1] >= needle_w:
            # Perform template matching on the restricted search area
            result = cv.matchTemplate(search_area, needle_img, cv.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            # print('Search area best match position: %s' % str(max_loc))
            # print('Search area best match confidence: %s' % max_val)

            # Adjust max_loc to the coordinates in the original frame
            adjusted_top_left = (search_area_top_left[0] + max_loc[0], search_area_top_left[1] + max_loc[1])

            if max_val >= threshold:
                # print('Found needle in search area.')
                # Draw a rectangle around the newly found match
                cv.rectangle(frame, adjusted_top_left, (adjusted_top_left[0] + needle_w, adjusted_top_left[1] + needle_h),
                             color=(0, 255, 255), thickness=2, lineType=cv.LINE_4)

            last_top_left = adjusted_top_left  # Update last matched position

    # Display the current frame
    cv.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv.destroyAllWindows()