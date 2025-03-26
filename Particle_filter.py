import sys
import cv2
import numpy as np

# Initialize tracker types
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]
# Initialize variables at the beginning
particle_frames_count = 0  # Counter for frames using particle filter
max_particle_frames = 5     # Number of frames to use particle filter before reinitializing tracker


if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create() 
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create() 
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create() 
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Read video
video = cv2.VideoCapture("test.mp4")
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame and initialize bounding box
ok, frame = video.read()
frame = cv2.resize(frame, (1280, 720))
if not ok:
    print('Cannot read video file')
    sys.exit()

bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# Particle filter parameters
num_of_particles = 450
half_width = bbox[2] // 2
half_height = bbox[3] // 2
expanded_half_width = int(half_width * 2.5)
expanded_half_height = int(half_height * 2.5)
weights = np.ones(num_of_particles)
particles = np.tile(np.array([bbox[0] + half_width, bbox[1] + half_height, 0, 0]), (num_of_particles, 1)).T

def compute_weight(p, q):
    bc = np.sum(np.sqrt(p * q))
    return np.exp(20 * bc)

def compute_norm_hist(image, state):
    x_min = max(np.round(state[0] - half_width).astype(int), 0)
    x_max = min(np.round(state[0] + half_width).astype(int), image.shape[1])
    y_min = max(np.round(state[1] - half_height).astype(int), 0)
    y_max = min(np.round(state[1] + half_height).astype(int), image.shape[0])
    
    roi = image[y_min:y_max+1, x_min:x_max+1]
    roi_reduced = roi // 16
    roi_indexing = (roi_reduced[..., 0] + roi_reduced[..., 1] * 16 + roi_reduced[..., 2] * 16 ** 2).flatten()
    hist, _ = np.histogram(roi_indexing, bins=4096, range=(0, 4096))
    return hist / np.sum(hist)

# Dynamic particle count adjustment based on tracking confidence
def adjust_particle_count(weights, num_of_particles):
    weight_variance = np.var(weights)
    average_weight = np.mean(weights)

    if weight_variance > 0.1:  # Threshold for high variance
        num_of_particles = min(500, num_of_particles + 100)
    elif average_weight < 0.1:  # Low average weight
        num_of_particles = max(100, num_of_particles - 50)
    
    return num_of_particles

# Get initial histogram for the particle filter
q = compute_norm_hist(frame, np.array([bbox[0] + half_width, bbox[1] + half_height]))
while True:
    ok, frame = video.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    if not ok:
        # If tracking fails, use particle filter
        particles = np.clip(particles + np.random.normal(0, 10, particles.shape), 0, None)

        # Calculate weights based on the histogram
        for i in range(num_of_particles):
            p = compute_norm_hist(frame, particles[:, i])
            weights[i] = compute_weight(p, q)

        # Resample particles
        weights /= np.sum(weights)
        indices = np.random.choice(np.arange(num_of_particles), size=num_of_particles, p=weights)
        particles = particles[:, indices]

        # Draw bounding box based on particles
        x_c_mean, y_c_mean = np.round(np.average(particles[:2], axis=1)).astype(int)
        cv2.rectangle(frame, (x_c_mean - half_width, y_c_mean - half_height),
                      (x_c_mean + half_width, y_c_mean + half_height), (0, 255, 0), 2)

        # Increment the particle frame counter
        particle_frames_count += 1
    else:
        # Draw bounding box from tracker
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

        # Reset the particle frame counter since tracking succeeded
        particle_frames_count = 0

    # Check if we need to reinitialize the MOSSE tracker
    if particle_frames_count >= max_particle_frames:
        # Reinitialize the tracker with the last known position
        bbox = (x_c_mean - half_width, y_c_mean - half_height, half_width * 2, half_height * 2)
        tracker = cv2.legacy.TrackerMOSSE_create()  # Create a new MOSSE tracker
        tracker.init(frame, bbox)  # Initialize the tracker with the current frame and new bounding box
        particle_frames_count = 0  # Reset the counter

    # Display results
    cv2.imshow("Tracking", frame)

    # Exit on ESC
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Release resources
video.release()
cv2.destroyAllWindows()