# YOLOv11 Object Detection with ONNX Runtime WebGPU (Webcam & Static Image)

This project demonstrates object detection using the YOLOv11 model, running in the browser with ONNX Runtime and the WebGPU execution provider (with WASM as a fallback). It supports both live webcam input and detection on a static test image.

### Setup

1.  **Model**: The `yolo11n-2.onnx` (FP32) model is expected to be in the `public/models/` directory.
2.  **Test Image (Optional)**: For static image detection, a test image named `test_image.jpg` should be placed in the `public/` directory.
3.  **Dependencies**:
    *   `onnxruntime-web`: For running the ONNX model.

### Running the Detection

1.  Start the Vite development server (e.g., `npm run dev`).
2.  The application will load the ONNX model.
3.  **Webcam Mode**:
    *   Select a camera from the dropdown menu.
    *   Click "Start Webcam".
    *   Grant camera permissions if prompted.
    *   Detected objects will be outlined on the live video feed displayed on the canvas.
    *   Click "Stop Webcam" to end live detection.
4.  **Static Image Mode**:
    *   Ensure the webcam is stopped.
    *   Click the "Detect on Image" button to run the YOLOv11 model on `/test_image.jpg`.
    *   Detected objects will be outlined on a canvas element displaying the image.

### UI Controls

*   **Camera Selector**: A dropdown menu allows you to select from available video input devices.
*   **Start/Stop Webcam Button**: Toggles the live webcam detection mode.
*   **Detect on Image Button**: Triggers detection on the pre-loaded static image (disabled if webcam is active).
*   **Reset Model Button**: Allows manual re-initialization of the ONNX session.
*   **Enable Debug Mode Checkbox**: Shows a debug canvas with the preprocessed input to the model.
*   **Class Selector**: Checkboxes to filter which object classes (from 80 COCO classes) are displayed.

### Key Components (`src/App.tsx`)

*   **ONNX Session Initialization**: In a `useEffect` hook, the ONNX runtime is initialized, preferring the WebGPU execution provider.
*   **Webcam Handling**:
    *   `useEffect` for enumerating video devices.
    *   `startWebcam()` and `stopWebcam()` functions to manage the camera stream.
    *   `detectWebcamFrame()`: Asynchronous function in a `requestAnimationFrame` loop that:
        *   Captures a frame from the webcam video element.
        *   Preprocesses the frame (direct resize to 640x640, normalization).
        *   Runs inference using the ONNX session.
        *   Processes model output (extracts bounding boxes and class scores, filters by class confidence, applies NMS).
        *   Draws the webcam feed and visualizes detections on the main canvas.
*   **Static Image Handling**:
    *   `useEffect` to load `/test_image.jpg` on mount.
    *   `runDetectionOnStaticImage()`: Asynchronous function that performs preprocessing, inference, and postprocessing for the static image, similar to the webcam loop.
*   **Preprocessing**: Input images/frames are directly resized to 640x640 and pixel values normalized to [0,1].
*   **Postprocessing**: Model output is transposed. Detections are filtered based on class confidence scores (threshold 0.25). Non-Maximum Suppression (NMS) is applied to remove redundant boxes.
*   **`drawBox()`**: Helper function to render bounding boxes and labels on the canvas.

### Notes

*   **Class Names (`CLASSES_LIST`)**: The application uses a predefined list of 80 COCO class names. This should match the classes your ONNX model was trained on.
*   **Model Precision**: The implementation is configured for an FP32 model (`yolo11n-2.onnx`).
