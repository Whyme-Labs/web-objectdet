import React, { useState, useEffect, useRef } from 'react'
import type { DragEvent } from 'react'
import * as ort from 'onnxruntime-web/webgpu'
// import reactLogo from './assets/react.svg' // Removed
// import viteLogo from '/vite.svg' // Removed
import './App.css'

// Moved CLASSES_LIST outside the component and to the top to be accessible by useState initial value
const CLASSES_LIST_FULL: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
  'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush' // 80th class: toothbrush
];

// Use all 80 COCO classes, consistent with Python script's apparent expectation
const CLASSES_LIST: string[] = CLASSES_LIST_FULL;

type DetectionMode = 'initial' | 'webcam' | 'image';

function App() {
  // const [count, setCount] = useState(0) // Removed count state
  const [session, setSession] = useState<ort.InferenceSession | null>(null)
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(new Set(CLASSES_LIST))
  // const [enableDebug, setEnableDebug] = useState(true) // Removed enableDebug state
  const [modelLoading, setModelLoading] = useState(false)
  const [inferenceStats, setInferenceStats] = useState({
    preprocessTime: 0,
    inferenceTime: 0,
    postprocessTime: 0,
    fps: 0
  })
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // const debugCanvasRef = useRef<HTMLCanvasElement>(null) // Removed debugCanvasRef
  const videoRef = useRef<HTMLVideoElement>(null);
  const isInferring = useRef(false);
  const [currentDisplayImage, setCurrentDisplayImage] = useState<HTMLImageElement | null>(null);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const animationFrameId = useRef<number | null>(null);
  const lastFrameTime = useRef(performance.now());
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('initial');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [isLoadingTestImage, setIsLoadingTestImage] = useState(false);
  const loadedImageRef = useRef<HTMLImageElement>(null);

  // Placeholder for OpenCV.js - for compatibility but not used for critical processing
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  // const cv: any = isCvReady ? (window as any).cv : null // cv removed

  // Improved drawBox function
  const drawBox = (
    ctx: CanvasRenderingContext2D | null,
    x: number,
    y: number,
    w: number,
    h: number,
    className: string,
    rawScore: number, // This is the 'score' from nmsResults
    canvasWidth: number,
    canvasHeight: number
  ) => {
    if (!ctx) {
      // console.warn("drawBox called with null context"); // Optional: uncomment if needed
      return;
    }

    // console.log(`drawBox inputs: x=${x.toFixed(2)}, y=${y.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}, score=${rawScore.toExponential(2)}, canvasW=${canvasWidth}, canvasH=${canvasHeight}`);

    let color = '#FF0000'; // Default red
    if (className) {
      let hash = 0;
      for (let i = 0; i < className.length; i++) {
        hash = className.charCodeAt(i) + ((hash << 5) - hash);
        hash = hash & hash;
      }
      const r = (hash & 0xFF0000) >> 16;
      const g = (hash & 0x00FF00) >> 8;
      const b = hash & 0x0000FF;
      color = `rgb(${r},${g},${b})`;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    const drawX = Math.max(0, x);
    const drawY = Math.max(0, y);
    const drawW = Math.min(w, canvasWidth - drawX);
    const drawH = Math.min(h, canvasHeight - drawY);

    if (drawW <= 0 || drawH <= 0) {
      // console.log(`Skipping drawRect due to non-positive dimensions: drawW=${drawW.toFixed(2)}, drawH=${drawH.toFixed(2)}. Original w=${w.toFixed(2)}, h=${h.toFixed(2)}. x=${x.toFixed(2)}, y=${y.toFixed(2)}`);
      return; 
    }

    ctx.strokeRect(drawX, drawY, drawW, drawH);

    const textToDisplay = `${className}: ${rawScore.toExponential(2)}`; // Use exponential for small scores
    ctx.fillStyle = color;
    ctx.font = '14px Arial';
    const textMetrics = ctx.measureText(textToDisplay);
    const textWidth = textMetrics.width;
    const textHeight = 14; // Approximate height based on font size

    let textX = drawX;
    let textY = drawY - 5;

    // Background for text for better readability
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)'; 
    ctx.fillRect(
      textX -1, 
      textY - textHeight, 
      textWidth + 2, 
      textHeight + 2
    );

    ctx.fillStyle = color;
    ctx.fillText(textToDisplay, textX, textY);
    
    // console.log(`Drawing box: ${className} at (${x}, ${y}) with size (${w}, ${h}) and score ${rawScore}`);
  };

  // Non-Maximum Suppression for YOLO detection results
  const performNMS = (
    boxes: number[][],            // Format: [[x1, y1, x2, y2], ...]
    scores: number[],             // Detection confidence scores
    classIndices: number[],       // Class indices for each box
    iouThreshold = 0.45,          // IOU threshold for NMS
    maxDetectionsPerClass = 20    // Maximum detections to return per class
  ): number[][] => {
    if (boxes.length === 0) return [];
    
    // Group boxes by class
    const classBins = new Map<number, {
      boxes: number[][],
      scores: number[],
      indices: number[]  // Original indices
    }>();
    
    // Organize detections by class
    boxes.forEach((box, i) => {
      const classId = classIndices[i];
      if (!classBins.has(classId)) {
        classBins.set(classId, {
          boxes: [],
          scores: [],
          indices: []
        });
      }
      
      const bin = classBins.get(classId)!;
      bin.boxes.push(box);
      bin.scores.push(scores[i]);
      bin.indices.push(i);
    });
    
    const results: number[][] = [];
    
    // Apply NMS for each class separately
    classBins.forEach((bin, classId) => {
      const { boxes: classBoxes, scores: classScores /*, indices: classIndices */ } = bin; // Removed unused classIndices from destructuring
      
      // Early return if no boxes for this class
      if (classBoxes.length === 0) return;
      
      // Sort by score (descending)
      const sortedIndices = classScores
        .map((_, i) => i)
        .sort((a, b) => classScores[b] - classScores[a]);
      
      const picked: number[] = [];
      
      // Apply NMS algorithm
      while (sortedIndices.length > 0 && picked.length < maxDetectionsPerClass) {
        const currentIdx = sortedIndices[0];
        
        // Keep the box with highest score
        picked.push(currentIdx);
        
        // Remove the current index
        sortedIndices.shift();
        
        // Skip if we've used all indices
        if (sortedIndices.length === 0) break;
        
        // Current box
        const currentBox = classBoxes[currentIdx];
        
        // Calculate IoU with remaining boxes and filter out overlapping ones
        const remainingIndices: number[] = [];
        
        for (const idx of sortedIndices) {
          // Calculate IoU between boxes
          const box = classBoxes[idx];
          
          const xA = Math.max(currentBox[0], box[0]);
          const yA = Math.max(currentBox[1], box[1]);
          const xB = Math.min(currentBox[2], box[2]);
          const yB = Math.min(currentBox[3], box[3]);
          
          // Area of intersection
          const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
          
          // Area of both boxes
          const boxAArea = (currentBox[2] - currentBox[0]) * (currentBox[3] - currentBox[1]);
          const boxBArea = (box[2] - box[0]) * (box[3] - box[1]);
          
          // Calculate IoU
          const iou = intersectionArea / (boxAArea + boxBArea - intersectionArea);
          
          // Keep boxes below the IoU threshold
          if (iou <= iouThreshold) {
            remainingIndices.push(idx);
          }
        }
        
        // Update sorted indices with remaining boxes
        sortedIndices.length = 0;
        sortedIndices.push(...remainingIndices);
      }
      
      // Add kept boxes to results
      for (const idx of picked) {
        const box = classBoxes[idx];
        const score = classScores[idx];
        
        results.push([...box, classId, score]);
      }
    });
    
    // Sort all detections by score (descending)
    results.sort((a, b) => b[5] - a[5]);
    
    return results;
  };

  const enumerateAvailableCameras = async () => {
    try {
      console.log("Enumerating available cameras...");
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedCamera) {
        console.log("Defaulting to first camera: " + (videoDevices[0].label || videoDevices[0].deviceId));
        setSelectedCamera(videoDevices[0].deviceId);
      } else if (videoDevices.length === 0) {
        console.warn("No video input devices found.");
      }
      // console.log("Available cameras set:", videoDevices); // Can be noisy
    } catch (err) {
      console.error("Error enumerating cameras:", err);
    }
  };

  // Initialize ONNX session and get cameras when component mounts or when explicitly requested
  useEffect(() => {
    async function setupApp() {
      if (!session) {
        try {
          setModelLoading(true);
          const newSession = await ort.InferenceSession.create('./models/yolo11n-2.onnx', {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: 'all',
          });
          setSession(newSession);
          console.log("ONNX Runtime session initialized successfully.");
          await enumerateAvailableCameras(); 
        } catch (e) {
          console.error("Failed to initialize ONNX session or load model:", e);
        } finally {
          setModelLoading(false);
        }
      } else {
        if (availableCameras.length === 0) {
            await enumerateAvailableCameras();
        }
      }
    }
    setupApp();

    // Cleanup function for when the component unmounts or session changes
    return () => {
      if (session) {
        // session.release(); // Based on ORT Web docs, explicit release might not be needed if session is just nulled
        // console.log("ONNX session resources would be released here if applicable.");
      }
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
      // Stop webcam stream if active
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
        console.log("Webcam stream stopped on component unmount/cleanup.");
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Main setup effect, runs once

  useEffect(() => {
    if (detectionMode === "webcam" && session && availableCameras.length === 0) {
        console.log("useEffect [detectionMode]: webcam mode, session ready, no cameras. Fetching.");
        enumerateAvailableCameras();
    }
    // Re-evaluate dependencies: selectedCamera is set by enumerateAvailableCameras, 
    // so including it might cause a loop if not careful. 
    // availableCameras.length is sufficient to know if enumeration happened.
  }, [detectionMode, session, availableCameras.length]);

  const startWebcam = async () => {
    if (videoRef.current && selectedCamera) {
      try {
        // if (canvasRef.current) { // Corrected: This was for debugCanvasRef, which is now removed.
        //   const mainCtx = canvasRef.current.getContext('2d');
        //   mainCtx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        //   console.log("Main canvas cleared on webcam start.");
        // }

        const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: selectedCamera } });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsWebcamActive(true); // This will trigger the useEffect to start the loop
          console.log("Webcam started, isWebcamActive set to true.");
          // detectWebcamFrame(); // Removed direct call
        };
      } catch (err) {
        console.error("Error starting webcam: ", err);
        setIsWebcamActive(false); // Ensure state is false if start fails
      }
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      // Setting isWebcamActive to false will trigger the useEffect to stop the loop
      setIsWebcamActive(false); 
      console.log("Webcam stopped, isWebcamActive set to false.");
      
      // No need to manually cancel animationFrameId.current here, useEffect will handle it.
      // if (animationFrameId.current) {
      //   cancelAnimationFrame(animationFrameId.current);
      //   animationFrameId.current = null;
      // }
      
      if(canvasRef.current) {
        const mainCtx = canvasRef.current.getContext('2d');
        mainCtx?.clearRect(0,0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  // useEffect to manage the detection loop based on isWebcamActive state
  useEffect(() => {
    if (isWebcamActive) {
      console.log("[useEffect] Webcam is active, starting detection loop.");
      isInferring.current = false; 
      lastFrameTime.current = performance.now(); 
      animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
    } else {
      // console.log("[useEffect] Webcam is not active, ensuring detection loop is stopped."); // Can be a bit noisy
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
        // console.log("[useEffect cleanup] Cancelled animation frame on cleanup."); // Can be noisy
      }
    }
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
        // console.log("[useEffect cleanup] Cancelled animation frame on cleanup."); // Can be noisy
      }
    };
  }, [isWebcamActive]);

  const detectWebcamFrame = async () => {
    if (!isWebcamActive || !session || !videoRef.current || videoRef.current.paused || videoRef.current.ended || !canvasRef.current) {
      if (isWebcamActive) { 
        animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
      }
      return;
    }
    if (isInferring.current) {
        if (isWebcamActive) {
            animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
        }
        return;
    }

    isInferring.current = true;
    const video = videoRef.current!;
    const mainCanvas = canvasRef.current!;
    const mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });

    if (!mainCtx) {
      console.error("Failed to get 2D context from main canvas for webcam detection.");
      isInferring.current = false;
      if (isWebcamActive) animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
      return;
    }

    const modelInputSize = 640;

    try {
      const preProcessStart = performance.now();
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = modelInputSize;
      tempCanvas.height = modelInputSize;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) {
        console.error("Failed to get context from temp canvas for webcam.");
        isInferring.current = false;
        if (isWebcamActive) animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
        return;
      }

      // Direct resize from video to modelInputSize x modelInputSize
      tempCtx.drawImage(video, 0, 0, modelInputSize, modelInputSize);
      
      const imageData = tempCtx.getImageData(0, 0, modelInputSize, modelInputSize);
      const pixels = imageData.data;
      const tensorData = new Float32Array(1 * 3 * modelInputSize * modelInputSize);
      const rOffset = 0, gOffset = modelInputSize * modelInputSize, bOffset = 2 * modelInputSize * modelInputSize;
      for (let y = 0; y < modelInputSize; y++) {
        for (let x = 0; x < modelInputSize; x++) {
          const pixelIndex = (y * modelInputSize + x) * 4;
          const r = pixels[pixelIndex] / 255.0;
          const g = pixels[pixelIndex + 1] / 255.0;
          const b = pixels[pixelIndex + 2] / 255.0;
          const tensorIndex = y * modelInputSize + x;
          tensorData[rOffset + tensorIndex] = r;
          tensorData[gOffset + tensorIndex] = g;
          tensorData[bOffset + tensorIndex] = b;
        }
      }
      
      const preProcessEnd = performance.now();

      const inferenceStart = performance.now();
      const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, modelInputSize, modelInputSize]);
      const feeds = { images: inputTensor };
      const results = await session.run(feeds);
      const inferenceEnd = performance.now();

      const postProcessStart = performance.now();
      const output = results.output0;
      if (!output || !output.dims || output.dims.length !== 3) {
        console.error("Invalid model output format for webcam.");
        isInferring.current = false;
        if (isWebcamActive) animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
        return;
      }
      const numOutputFeatures = output.dims[1];
      const numCandidates = output.dims[2];
      const numClasses = CLASSES_LIST.length;
      if (numOutputFeatures !== 4 + numClasses) {
        console.error(`Output feature size mismatch. Expected ${4 + numClasses}, got ${numOutputFeatures}.`);
        isInferring.current = false;
        if (isWebcamActive) animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
        return;
      }
      const outputData = output.data as Float32Array;
      const transposedData: number[][] = [];
      for (let i = 0; i < numCandidates; i++) {
        const rowData: number[] = [];
        for (let j = 0; j < numOutputFeatures; j++) {
          rowData.push(outputData[j * numCandidates + i]);
        }
        transposedData.push(rowData);
      }
      
      const filteredBoxes: number[][] = [];
      const filteredScores: number[] = [];
      const filteredClassIndices: number[] = [];
      const confidenceThreshold = 0.25;
      
      for (const detection of transposedData) {
        const classScores = detection.slice(4);
        let currentMaxClassScore = 0;
        let currentMaxClassIndex = -1;
        for (let i = 0; i < classScores.length; i++) {
          if (classScores[i] > currentMaxClassScore) {
            currentMaxClassScore = classScores[i];
            currentMaxClassIndex = i;
          }
        }
        if (currentMaxClassScore > confidenceThreshold) {
          const x = detection[0], y = detection[1], w = detection[2], h = detection[3];
          filteredBoxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
          filteredScores.push(currentMaxClassScore);
          filteredClassIndices.push(currentMaxClassIndex);
        }
      }
      const nmsResults = performNMS(filteredBoxes, filteredScores, filteredClassIndices, 0.45, 20);
      const postProcessEnd = performance.now();

      const canvasWidth = mainCanvas.width;
      const canvasHeight = mainCanvas.height;
      mainCtx.clearRect(0, 0, canvasWidth, canvasHeight);
      // Draw video frame to main canvas, maintaining aspect ratio
      const videoAspectRatio = video.videoWidth / video.videoHeight;
      let drawWidth = canvasWidth;
      let drawHeight = canvasWidth / videoAspectRatio;
      if (drawHeight > canvasHeight) {
          drawHeight = canvasHeight;
          drawWidth = canvasHeight * videoAspectRatio;
      }
      const drawX = (canvasWidth - drawWidth) / 2;
      const drawY = (canvasHeight - drawHeight) / 2;
      mainCtx.drawImage(video, drawX, drawY, drawWidth, drawHeight);

      setInferenceStats({
          preprocessTime: preProcessEnd - preProcessStart,
          inferenceTime: inferenceEnd - inferenceStart,
          postprocessTime: postProcessEnd - postProcessStart,
          fps: 1000 / (performance.now() - lastFrameTime.current)
      });
      lastFrameTime.current = performance.now();

      for (const result of nmsResults) {
        const [x1, y1, x2, y2, classIndex, score] = result;
        const className = CLASSES_LIST[classIndex];
        if (selectedClasses.has(className)) {
          const scaleX = drawWidth / modelInputSize; 
          const scaleY = drawHeight / modelInputSize;
          
          const actualX1 = drawX + x1 * scaleX;
          const actualY1 = drawY + y1 * scaleY;
          const actualW = (x2 - x1) * scaleX;
          const actualH = (y2 - y1) * scaleY;
          drawBox(mainCtx, actualX1, actualY1, actualW, actualH, className, score, canvasWidth, canvasHeight);
        }
      }
    } catch (error) {
      console.error("Error during webcam detection loop:", error);
    } finally {
      isInferring.current = false;
      if (isWebcamActive) animationFrameId.current = requestAnimationFrame(detectWebcamFrame);
    }
  };

  const runDetectionOnStaticImage = async () => {
    if (!session || !currentDisplayImage || !canvasRef.current) {
      console.warn("Cannot run detection on static image: session, currentDisplayImage, or canvas not ready.");
      return;
    }

    if (isInferring.current) {
      console.warn("Inference already in progress for static image.");
      return;
    }

    isInferring.current = true;
    const imageToProcess = currentDisplayImage;
    const mainCanvas = canvasRef.current;
    const mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });

    if (!mainCtx) {
      console.error("Failed to get 2D context from main canvas for static image detection.");
      isInferring.current = false;
      return;
    }

    const modelInputSize = 640;

    try {
      console.log("Starting preprocessing for static image...");
      const preProcessStart = performance.now();

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = modelInputSize;
      tempCanvas.height = modelInputSize;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) {
        console.error("Failed to get context from temp canvas for static image.");
        isInferring.current = false;
        return;
      }
      
      tempCtx.drawImage(imageToProcess, 0, 0, modelInputSize, modelInputSize);
      console.log(`Static image directly resized and drawn to tempCanvas (${modelInputSize}x${modelInputSize})`);
      
      const imageData = tempCtx.getImageData(0, 0, modelInputSize, modelInputSize);
      const pixels = imageData.data;
      const tensorData = new Float32Array(1 * 3 * modelInputSize * modelInputSize);
      const rOffset = 0, gOffset = modelInputSize * modelInputSize, bOffset = 2 * modelInputSize * modelInputSize;

      for (let y = 0; y < modelInputSize; y++) {
        for (let x = 0; x < modelInputSize; x++) {
          const pixelIndex = (y * modelInputSize + x) * 4;
          const r = pixels[pixelIndex] / 255.0;
          const g = pixels[pixelIndex + 1] / 255.0;
          const b = pixels[pixelIndex + 2] / 255.0;
          const tensorIndex = y * modelInputSize + x;
          tensorData[rOffset + tensorIndex] = r;
          tensorData[gOffset + tensorIndex] = g;
          tensorData[bOffset + tensorIndex] = b;
        }
      }

      const preProcessEnd = performance.now();
      setInferenceStats(prev => ({ 
        ...prev, 
        preprocessTime: preProcessEnd - preProcessStart 
      }));

      console.log("Starting inference for static image...");
      const inferenceStart = performance.now();
      const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, modelInputSize, modelInputSize]);
      const feeds = { images: inputTensor };
      const results = await session.run(feeds);
      const inferenceEnd = performance.now();
      setInferenceStats(prev => ({ 
        ...prev, 
        inferenceTime: inferenceEnd - inferenceStart 
      }));

      console.log("Starting postprocessing for static image...");
      const postProcessStart = performance.now();
      const output = results.output0;
      if (!output || !output.dims || output.dims.length !== 3) {
        console.error("Invalid model output format for static image.");
        isInferring.current = false;
        return;
      }
      const numOutputFeatures = output.dims[1];
      const numCandidates = output.dims[2];
      const numClasses = CLASSES_LIST.length;
      if (numOutputFeatures !== 4 + numClasses) {
        console.error(
          `Output feature size mismatch. Expected ${4 + numClasses} (4 bbox + ${numClasses} classes), got ${numOutputFeatures}. ` +
          `Ensure CLASSES_LIST length (${CLASSES_LIST.length}) matches model output class count.`
        );
        isInferring.current = false;
        return;
      }
      const outputData = output.data as Float32Array;
      const transposedData: number[][] = [];
      for (let i = 0; i < numCandidates; i++) {
        const rowData: number[] = [];
        for (let j = 0; j < numOutputFeatures; j++) {
          rowData.push(outputData[j * numCandidates + i]);
        }
        transposedData.push(rowData);
      }

      const filteredBoxes: number[][] = [];
      const filteredScores: number[] = [];
      const filteredClassIndices: number[] = [];
      const confidenceThreshold = 0.25;

      for (const detection of transposedData) {
        const classScores = detection.slice(4);
        let currentMaxClassScore = 0;
        let currentMaxClassIndex = -1;
        for (let i = 0; i < classScores.length; i++) {
          if (classScores[i] > currentMaxClassScore) {
            currentMaxClassScore = classScores[i];
            currentMaxClassIndex = i;
          }
        }
        if (currentMaxClassScore > confidenceThreshold) {
          const x = detection[0], y = detection[1], w = detection[2], h = detection[3];
          filteredBoxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
          filteredScores.push(currentMaxClassScore);
          filteredClassIndices.push(currentMaxClassIndex);
        }
      }
      
      console.log(`Filtered ${filteredBoxes.length} detections before NMS (static image).`);
      const nmsResults = performNMS(filteredBoxes, filteredScores, filteredClassIndices, 0.45, 20);
      console.log(`NMS returned ${nmsResults.length} final detections (static image).`);
      const postProcessEnd = performance.now();
      setInferenceStats(prev => ({ 
        ...prev, 
        postprocessTime: postProcessEnd - postProcessStart
      }));

      const canvasWidth = mainCanvas.width;
      const canvasHeight = mainCanvas.height;
      const imgAspectRatioDisplay = imageToProcess.naturalWidth / imageToProcess.naturalHeight;
      let drawWidthDisplay = canvasWidth;
      let drawHeightDisplay = canvasWidth / imgAspectRatioDisplay;
      if (drawHeightDisplay > canvasHeight) {
        drawHeightDisplay = canvasHeight;
        drawWidthDisplay = canvasHeight * imgAspectRatioDisplay;
      }
      const drawXDisplay = (canvasWidth - drawWidthDisplay) / 2;
      const drawYDisplay = (canvasHeight - drawHeightDisplay) / 2;

      mainCtx.clearRect(0, 0, canvasWidth, canvasHeight);
      mainCtx.drawImage(imageToProcess, drawXDisplay, drawYDisplay, drawWidthDisplay, drawHeightDisplay);
      
      for (const result of nmsResults) {
        const [x1, y1, x2, y2, classIndex, score] = result;
        const className = CLASSES_LIST[classIndex];
        if (selectedClasses.has(className)) {
          const scaleX = drawWidthDisplay / modelInputSize;
          const scaleY = drawHeightDisplay / modelInputSize;
          
          const actualX1 = drawXDisplay + x1 * scaleX;
          const actualY1 = drawYDisplay + y1 * scaleY;
          const actualW = (x2 - x1) * scaleX;
          const actualH = (y2 - y1) * scaleY;
          
          console.log(
            `Static Img Draw: ${className} (score: ${score.toExponential(3)}) ` +
            `Box [${actualX1.toFixed(0)}, ${actualY1.toFixed(0)}, ${actualW.toFixed(0)}, ${actualH.toFixed(0)}]`
          );
          drawBox(mainCtx, actualX1, actualY1, actualW, actualH, className, score, canvasWidth, canvasHeight);
        }
      }
      
    } catch (error) {
      console.error("Error during static image detection:", error);
      if (mainCtx) {
        mainCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
        mainCtx.fillRect(50, 50, mainCanvas.width - 100, 100);
        mainCtx.fillStyle = 'white';
        mainCtx.font = '16px Arial';
        mainCtx.fillText(`Static Img Error: ${error instanceof Error ? error.message : 'Unknown error'}`, 70, 100);
    }
    } finally {
      isInferring.current = false;
    }
  };

  const handleClassSelectionChange = (className: string) => {
    setSelectedClasses(prevSelectedClasses => {
      const newSelectedClasses = new Set(prevSelectedClasses);
      if (newSelectedClasses.has(className)) {
        newSelectedClasses.delete(className);
      } else {
        newSelectedClasses.add(className);
      }
      return newSelectedClasses;
    });
  };

  // Function to reinitialize the model manually
  const reinitializeModel = async () => {
    if (isWebcamActive) stopWebcam(); // Stop webcam if active before reinitializing
    if (session) {
      try {
        await session.release();
        console.log("Previous ONNX session released.");
      } catch (e) {
        console.error("Error releasing previous session:", e);
      }
      setSession(null);
    }
    
    try {
      console.log("Manually reinitializing ONNX session...");
      setModelLoading(true);
      await new Promise(resolve => setTimeout(resolve, 100));
      const modelPath = '/models/yolo11n-2.onnx';
      console.log("Loading ONNX model from:", modelPath);
      try {
        const response = await fetch(modelPath);
        if (!response.ok) {
          console.error(`Model file not found: ${modelPath}. Status: ${response.status}`);
          throw new Error(`Model file not found: ${modelPath}`);
        }
        console.log("Model file exists, proceeding with session creation");
      } catch (error) {
        console.error("Error checking model file:", error);
        throw error;
      }
      const newSession = await ort.InferenceSession.create(
        modelPath,
        { executionProviders: ['webgpu', 'wasm'] }
      );
      setSession(newSession);
      console.log("ONNX session created successfully:", newSession);
      console.log("Testing session with dummy tensor...");
      const dummyTensor = new ort.Tensor('float32', new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
      try {
        const dummyResult = await newSession.run({ images: dummyTensor });
        console.log("Test inference successful, output shape:", dummyResult.output0.dims);
      } catch (error) {
        console.error("Test inference failed:", error);
      }
      setModelLoading(false);
      console.log('ONNX session reinitialized successfully.');
    } catch (error) {
      console.error('Failed to reinitialize ONNX runtime:', error);
      console.error('Error details:', JSON.stringify(error, Object.getOwnPropertyNames(error)));
      setModelLoading(false);
    }
  };

  const handleSelectAllClasses = () => {
    setSelectedClasses(new Set(CLASSES_LIST));
  };

  const handleDeselectAllClasses = () => {
    setSelectedClasses(new Set());
  };

  const loadTestImage = () => {
    console.log("Loading test image /test_image.jpg");
    setIsLoadingTestImage(true);
    const img = new Image();
    img.src = '/test_image.jpg'; 
    img.onload = () => {
      console.log("Test image loaded successfully into Image object.");
      // setStaticImageElement(img); // Removed
      // setUploadedImageElement(null); // Removed, as setCurrentDisplayImage handles this logic flow
      setCurrentDisplayImage(img);
      setIsLoadingTestImage(false);
    };
    img.onerror = () => {
      console.error("Failed to load static image /test_image.jpg. Make sure it exists in the public folder.");
      setCurrentDisplayImage(null);
      setIsLoadingTestImage(false);
    };
  };

  const handleImageUpload = (file: File) => {
    if (file && file.type.startsWith('image/')){
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          console.log("Uploaded image loaded successfully.");
          // setUploadedImageElement(img); // Removed
          // setStaticImageElement(null); // Removed
          setCurrentDisplayImage(img);
        };
        img.onerror = () => {
          console.error("Error loading uploaded image data.");
          setCurrentDisplayImage(null);
        }
        img.src = e.target?.result as string;
      };
      reader.onerror = () => {
        console.error("Error reading uploaded file.");
        setCurrentDisplayImage(null);
      }
      reader.readAsDataURL(file);
    } else {
      console.warn("Invalid file type. Please upload an image.");
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const selectMode = (mode: DetectionMode) => {
    if (isWebcamActive) stopWebcam();
    setCurrentDisplayImage(null); 
    // setUploadedImageElement(null); // Removed
    // setStaticImageElement(null); // Removed
    setIsLoadingTestImage(false);
    if (canvasRef.current) {
        const mainCtx = canvasRef.current.getContext('2d');
        mainCtx?.clearRect(0,0, canvasRef.current.width, canvasRef.current.height);
    }
    setDetectionMode(mode);
    if (mode === 'webcam' && session && availableCameras.length === 0) {
      console.log("selectMode: Switched to webcam, session ready, no cameras. Fetching.");
      enumerateAvailableCameras();
    }
  };

  return (
    <>
      <div className="controls-container">
        {/* Mode Selection Buttons */}
        {!session && (
          <div className="info-text">Initializing ONNX Runtime and loading model...</div>
        )}
        {session && detectionMode === "initial" && (
          <div className="mode-selection">
            <button onClick={() => selectMode('webcam')} disabled={!session || modelLoading} className="action-button">
              {modelLoading ? "Loading..." : "Start Webcam Detection"}
            </button>
            <button onClick={() => selectMode('image')} disabled={!session || modelLoading} className="action-button">
              {modelLoading ? "Loading..." : "Detect from Image"}
            </button>
          </div>
        )}

        {detectionMode === 'webcam' && (
          <div className="webcam-controls">
            <select onChange={(e) => setSelectedCamera(e.target.value)} value={selectedCamera} disabled={isWebcamActive || !session || availableCameras.length === 0 || modelLoading} style={{ marginRight: '10px', padding: '5px' }} className="action-button">
              {availableCameras.length === 0 && <option>No cameras found</option>}
              {availableCameras.map(device => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${availableCameras.indexOf(device) + 1}`}
                </option>
              ))}
            </select>
            <button onClick={isWebcamActive ? stopWebcam : startWebcam} disabled={!selectedCamera || !session || modelLoading} className="action-button">
              {session ? (isWebcamActive ? "Stop Webcam" : "Start Webcam") : "Model..."}
            </button>
            <button onClick={() => {
              stopWebcam();
              selectMode('initial');
            }} disabled={modelLoading} className="action-button">
              Back to Mode Selection
            </button>
          </div>
        )}

        {detectionMode === 'image' && (
          <div className="image-controls">
            <button onClick={loadTestImage} disabled={isInferring.current || !session || modelLoading || isLoadingTestImage} className="action-button">
              {isLoadingTestImage ? "Loading Test..." : "Use Test Image"} 
            </button>
            
            <div 
              className="drop-zone"
              onDrop={handleDrop} 
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              role="button" 
              tabIndex={0} 
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
            >
              {currentDisplayImage ? 'Drag & drop or click to change loaded image' : 'Drag & drop image here, or click to select'}
            </div>
            
            <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} style={{display: 'none'}}/>

            <button onClick={() => {
              selectMode('initial');
              setCurrentDisplayImage(null);
              if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            }} disabled={modelLoading} className="action-button">
              Back to Mode Selection
            </button>
          </div>
        )}
        
        {/* Inference Stats - Display if not initial mode */} 
        {(detectionMode === 'webcam' || (detectionMode === 'image' && currentDisplayImage)) && inferenceStats.preprocessTime > 0 && (
            <div className="stats-display">
                {detectionMode === 'webcam' && <span>FPS: {inferenceStats.fps.toFixed(1)} | </span>}
                <span>Pre: {inferenceStats.preprocessTime.toFixed(1)}ms | </span>
                <span>Infer: {inferenceStats.inferenceTime.toFixed(1)}ms | </span>
                <span>Post: {inferenceStats.postprocessTime.toFixed(1)}ms</span>
            </div>
        )}
      </div>

      <div className="media-display">
        {/* Video element is now always hidden, but still used for the stream */}
        <video ref={videoRef} width="640" height="480" style={{ display: 'none' }} playsInline />
        {/* Canvas for displaying detections - visible if webcam is active OR an image is loaded */}
        <canvas 
            ref={canvasRef} 
            width="640" 
            height="480" 
            style={{
                border: '1px solid black', 
                display: (detectionMode === 'webcam' && isWebcamActive) || (detectionMode === 'image' && currentDisplayImage) ? 'block' : 'none' 
            }}
        />

        {/* Hidden image tag to drive canvas update and detection for BOTH test image and uploaded image */}
        {detectionMode === "image" && currentDisplayImage && currentDisplayImage.src && (
          <img 
            key={currentDisplayImage.src} // Add key to ensure re-render if src changes to same object but different content (though new Image() helps)
            src={currentDisplayImage.src} 
            alt="Content for detection" 
            ref={loadedImageRef} // This ref might not be strictly necessary if we operate directly on currentDisplayImage
            style={{ display: 'none' }} 
            onLoad={() => {
                const imageToProcessOnLoad = currentDisplayImage;
                if (imageToProcessOnLoad && canvasRef.current) {
                    console.log(`Hidden img onLoad: Image source '${imageToProcessOnLoad.src.substring(0,100)}...' loaded. Drawing to canvas and running detection.`);
                    canvasRef.current.width = imageToProcessOnLoad.naturalWidth;
                    canvasRef.current.height = imageToProcessOnLoad.naturalHeight;
                    const ctx = canvasRef.current.getContext('2d');
                    if (ctx) {
                        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                        ctx.drawImage(imageToProcessOnLoad, 0, 0, canvasRef.current.width, canvasRef.current.height);
                        console.log("Image drawn to main canvas from hidden img tag.");
                        if (session) {
                            runDetectionOnStaticImage();
                        } else {
                            console.warn("Session not available when trying to run detection on image from hidden img load.");
                        }
                    } else {
                        console.error("Failed to get context from main canvas in hidden img onLoad.");
                    }
                } else {
                    console.warn("Hidden img onLoad: imageToProcess or canvasRef not ready.", {imageToProcess: imageToProcessOnLoad, canvasRefCurrent: canvasRef.current});
                }
            }}
            onError={(e) => {
                console.error("Error loading image in hidden img tag:", currentDisplayImage?.src, e);
                // Optionally, clear currentDisplayImage or show an error to the user
            }}
          />
        )}
      </div>

      {/* Class Selector - Display if a mode is active and content is ready or webcam is on */} 
      {((detectionMode === 'webcam' && isWebcamActive) || (detectionMode === 'image' && currentDisplayImage)) && (
        <div className="class-selector-container">
          <h4>Select classes to detect:</h4>
          <div style={{ marginBottom: '10px' }}>
            <button onClick={handleSelectAllClasses} style={{ marginRight: '5px' }} className="action-button">Select All</button>
            <button onClick={handleDeselectAllClasses} className="action-button">Deselect All</button>
          </div>
          <div className="class-selector-grid">
            {CLASSES_LIST.map(className => (
              <div key={className} className="class-selector-item">
                <input
                  type="checkbox"
                  id={`checkbox-${className}`}
                  checked={selectedClasses.has(className)}
                  onChange={() => handleClassSelectionChange(className)}
                />
                <label htmlFor={`checkbox-${className}`}>{className}</label>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Reset Model Button - MOVED here, to the bottom */}
      {session && (
        <div style={{ marginTop: '20px', textAlign: 'center' }}> {/* Added a wrapper for centering and margin */}
          <button onClick={reinitializeModel} disabled={modelLoading} className="action-button">
              {modelLoading ? 'Loading Model...' : 'Reset Model'}
          </button>
        </div>
      )}
    </>
  )
}

export default App


