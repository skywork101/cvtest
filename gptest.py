import cv2 as cv
import numpy as np

# Open the video file
video = cv.VideoCapture('test.mp4')
# Set the desired resolution (720p)
width = 1280
height = 720

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

# Save the cropped image as a new file
cv.imwrite('needle_image.jpg', needle_img)

# Process each frame in the video
last_top_left = None
search_area = None
threshold = 0.3

while True:
    ret, frame = video.read()
    frame = cv.resize(frame, (width, height))

    if not ret:
        break  # Exit if there are no more frames

    # Load the saved needle image for matching
    needle_img = cv.imread('needle_image.jpg')
    if search_area is None:
        # Perform template matching with the static needle image
        result = cv.matchTemplate(frame, needle_img, cv.TM_CCOEFF_NORMED)

        # Get the best match position from the match result
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        print('Best match top left position: %s' % str(max_loc))
        print('Best match confidence: %s' % max_val)

        # If the best match value is greater than the threshold, trust that we found a match
        if max_val >= threshold:
            print('Found needle.')
            last_top_left = max_loc  # Update last matched position

            # Calculate the bottom right corner of the rectangle to draw
            needle_w = needle_img.shape[1]
            needle_h = needle_img.shape[0]
            top_left = max_loc
            bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

            # Draw a rectangle around the matched template
            cv.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)


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

        # Perform template matching on the restricted search area
        result = cv.matchTemplate(search_area, needle_img, cv.TM_CCOEFF_NORMED)
        
        # Get the best match position in the search area
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        
        print('Search area best match position: %s' % str(max_loc))
        print('Search area best match confidence: %s' % max_val)

        # Adjust max_loc to the coordinates in the original frame
        adjusted_top_left = (last_top_left[0] - needle_w + max_loc[0], last_top_left[1] - needle_h + max_loc[1])

        last_top_left = adjusted_top_left  # Update last matched position

        if max_val >= threshold:
            print('Found needle in search area.')
            # Draw a rectangle around the newly found match
            cv.rectangle(frame, adjusted_top_left, (adjusted_top_left[0] + needle_w, adjusted_top_left[1] + needle_h),
                         color=(0, 255, 255), thickness=2, lineType=cv.LINE_4)

    # Display the current frame
    cv.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv.destroyAllWindows()