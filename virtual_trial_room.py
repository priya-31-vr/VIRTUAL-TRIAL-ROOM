import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Known width of the reference object in centimeters (e.g., a credit card is 8.5 cm)
REFERENCE_WIDTH_CM = 8.5

def compute_real_size(landmark1, landmark2, frame_shape, ref_pixel_width):
    # Calculate distance in pixels between two landmarksDAQ .39*6
    83720.302472    
    landmark1_px = (int(landmark1.x * frame_shape[1]), int(landmark1.y * frame_shape[0]))
    landmark2_px = (int(landmark2.x * frame_shape[1]), int(landmark2.y * frame_shape[0]))
    pixel_distance = np.linalg.norm(np.array(landmark2_px) - np.array(landmark1_px))
    
    # Calculate real-world distance in centimeters
    cm_per_pixel = REFERENCE_WIDTH_CM / ref_pixel_width
    real_distance_cm = pixel_distance * cm_per_pixel
    
    return real_distance_cm

def compute_chest_width(landmarks, frame_shape, ref_pixel_width):
    # Calculate chest width between left and right shoulder landmarks
    chest_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    chest_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    return compute_real_size(chest_left, chest_right, frame_shape, ref_pixel_width)

def compute_waist_width(landmarks, frame_shape, ref_pixel_width):
    # Calculate waist width between left and right hip landmarks
    waist_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    waist_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
    return compute_real_size(waist_left, waist_right, frame_shape, ref_pixel_width)

def compute_arm_length(landmarks, frame_shape, ref_pixel_width):
    # Calculate arm length between shoulder and elbow landmarks
    shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    return compute_real_size(shoulder, elbow, frame_shape, ref_pixel_width)

def estimate_tshirt_size(shoulder_width_cm, waist_width_cm, chest_width_cm, arm_length_cm):
    # Example size estimation based on body measurements
    if shoulder_width_cm < 38 and chest_width_cm < 85 and waist_width_cm < 70 and arm_length_cm < 60:
        return "Small"
    elif shoulder_width_cm < 42 and chest_width_cm < 95 and waist_width_cm < 80 and arm_length_cm < 65:
        return "Medium"
    elif shoulder_width_cm < 46 and chest_width_cm < 105 and waist_width_cm < 90 and arm_length_cm < 70:
        return "Large"
    elif shoulder_width_cm < 50 and chest_width_cm < 115 and waist_width_cm < 100 and arm_length_cm < 75:
        return "Extra Large"
    else:
        return "XXL"  # For larger body types

def overlay_tshirt(frame, landmarks, tshirt_img, scale_factor=1.5):
    # Get shoulder and hip landmarks to estimate T-shirt size
    shoulder_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate T-shirt dimensions based on shoulder width and height (shoulder to hip distance)
    shoulder_width = int(abs(shoulder_right.x - shoulder_left.x) * frame.shape[1] * scale_factor)
    height = int(abs(hip_left.y - shoulder_left.y) * frame.shape[0] * scale_factor)

    # Ensure T-shirt dimensions are valid
    if shoulder_width <= 0 or height <= 0:
        print("Invalid shoulder width or height detected, skipping overlay.")
        return frame

    # Resize the T-shirt image to fit the body
    tshirt_resized = cv2.resize(tshirt_img, (shoulder_width, height))

    # Check if T-shirt image has an alpha channel (RGBA)
    if tshirt_resized.shape[2] == 3:  # No alpha channel
        # Add an alpha channel (fully opaque)
        alpha_channel = np.ones((tshirt_resized.shape[0], tshirt_resized.shape[1], 1), dtype=tshirt_resized.dtype) * 255
        tshirt_resized = np.concatenate((tshirt_resized, alpha_channel), axis=2)

    # Calculate the position to overlay the T-shirt
    x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1]) - int(shoulder_width / 2)
    y = int(shoulder_left.y * frame.shape[0]) - int(tshirt_resized.shape[0] / 4)

    # Ensure T-shirt stays within the frame boundaries
    if x < 0:
        x = 0
    if x + shoulder_width > frame.shape[1]:
        x = frame.shape[1] - shoulder_width
    if y + height > frame.shape[0]:
        height = frame.shape[0] - y
        tshirt_resized = cv2.resize(tshirt_img, (shoulder_width, height))

    # Overlay the T-shirt on the frame
    alpha_tshirt = tshirt_resized[:, :, 3] / 255.0
    for c in range(0, 3):
        frame[y:y+height, x:x+shoulder_width, c] = (alpha_tshirt * tshirt_resized[:, :, c] + 
                                                    (1 - alpha_tshirt) * frame[y:y+height, x:x+shoulder_width, c])

    return frame

# Load multiple T-shirt images with transparency (PNG format)
tshirt_images = [
    cv2.imread('tshirt1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt3.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt4.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt5.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt6.png', cv2.IMREAD_UNCHANGED),
    
]

# Initialize variables for automatically changing T-shirts
tshirt_index = 0
change_interval = 3  # Time interval (in seconds) to automatically change the T-shirt

def change_tshirt_color(user_preference):
    global tshirt_index
    if user_preference == 'red':
        tshirt_index = 0
    elif user_preference == 'blue':
        tshirt_index = 1
    elif user_preference == 'green':
        tshirt_index = 2
    elif user_preference == 'black':
        tshirt_index = 3
    else:
        tshirt_index = 4  # Default to white

def run_virtual_trial_room():
    global tshirt_index
    # Capture Video from Webcam
    cap = cv2.VideoCapture(0)

    # Set the OpenCV window to be maximized
    cv2.namedWindow("Virtual Trial Room", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Trial Room", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_change_time = time.time()  # Track the time for automatic change

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Assume reference object is visible and measured in pixels
                ref_pixel_width = 150  # Example value; replace with actual measurement

                # Compute body measurements in centimeters
                shoulder_width_cm = compute_real_size(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], frame.shape, ref_pixel_width)
                waist_width_cm = compute_real_size(results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[24], frame.shape, ref_pixel_width)
                chest_width_cm = compute_chest_width(results.pose_landmarks, frame.shape, ref_pixel_width)
                arm_length_cm = compute_arm_length(results.pose_landmarks, frame.shape, ref_pixel_width)

                # Estimate T-shirt size based on body measurements
                tshirt_size = estimate_tshirt_size(shoulder_width_cm, waist_width_cm, chest_width_cm, arm_length_cm)

                # Overlay the T-shirt on the frame
                frame = overlay_tshirt(frame, results.pose_landmarks, tshirt_images[tshirt_index])

                # Display measurements and estimated size on the frame
                cv2.putText(frame, f'Shoulder Width: {shoulder_width_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Waist Width: {waist_width_cm:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Chest Width: {chest_width_cm:.2f} cm', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Arm Length: {arm_length_cm:.2f} cm', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'T-Shirt Size: {tshirt_size}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame with annotations
            cv2.imshow('Virtual Trial Room', frame)

            # Automatically change T-shirt every 'change_interval' seconds
            if time.time() - last_change_time > change_interval:
                tshirt_index = (tshirt_index + 1) % len(tshirt_images)  # Change the T-shirt
                last_change_time = time.time()  # Reset the time tracker

            # Exit the loop when 'q' or 'ESC' is pressed
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the virtual trial room in a separate thread
thread = threading.Thread(target=run_virtual_trial_room)
thread.start()
