# Virtual Trial Room

The **Virtual Trial Room** is a real-time application that uses computer vision and machine learning to estimate T-shirt sizes based on body measurements captured from a webcam. This project leverages OpenCV for video processing and Mediapipe for pose detection to calculate key body dimensions and provide size recommendations.

## Features

- **Real-Time Pose Detection**: Utilizes Mediapipe to detect and track body landmarks in real-time.
- **Measurement Calculation**: Computes shoulder and waist widths in centimeters using a reference object.
- **T-Shirt Size Estimation**: Provides size recommendations (Small, Medium, Large, Extra Large) based on measured dimensions.
- **Visual Feedback**: Displays measurements and estimated T-shirt size directly on the video feed.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- NumPy

## Installation



 1.**Install Dependencies**:
   Use the `requirements.txt` file to install all necessary packages.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:
   Execute the script to start capturing video from your webcam.
   ```bash
   python virtual_trial_room.py
   ```

2. **Setup**:
   - Ensure your webcam is properly connected.
   - Place a reference object with a known width (e.g., a credit card) within the camera's view for accurate measurement scaling.

3. **Operation**:
   - The application will display the video feed with pose landmarks annotated.
   - Shoulder and waist measurements will be calculated and displayed on the screen along with the estimated T-shirt size.

4. **Exit**:
   - Press 'q' or 'ESC' to close the application.

## Configuration

- **Reference Object**: Adjust the `ref_pixel_width` variable in the script to match the pixel width of your reference object for accurate measurements.

## Notes

- For best results, ensure good lighting conditions and a clear view of the entire upper body.
- The accuracy of size estimation depends on the correct calibration of the reference object's pixel width.

This project demonstrates how computer vision can be applied to practical problems like virtual fitting rooms, enhancing online shopping experiences by providing personalized size recommendations.
