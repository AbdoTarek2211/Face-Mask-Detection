# Face Mask Detection System

A comprehensive face mask detection system using deep learning and computer vision for real-time detection through webcam feed.

## 🎯 Project Overview

This project implements a robust face mask detection system that can:
- Train a deep learning model using CNN and MobileNetV2 architecture
- Detect faces in real-time using webcam
- Classify whether a person is wearing a mask or not
- Display confidence scores and visual feedback
- Save screenshots of detections

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loading  │ -> │  Model Training  │ -> │ Real-time       │
│   & Processing  │    │  & Evaluation    │    │ Detection       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: GlobalAveragePooling2D, Dense layers with Dropout
- **Training Strategy**: Two-phase training (initial + fine-tuning)
- **Input Size**: 224×224×3 RGB images
- **Output**: Binary classification (WithMask/WithoutMask)

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn pandas scikit-learn
```

### 1. Train the Model

```bash
python face_mask_detection_optimized.py
```

This will:
- Load and preprocess the dataset
- Train the model with data augmentation
- Apply fine-tuning for better performance
- Save the trained model as `face_mask_detection_final.h5`

### 2. Run Real-time Detection

```bash
python realtime_detection.py
```

#### Command Line Options:

```bash
python realtime_detection.py --model path/to/model.h5 --camera 0 --confidence 0.7
```

- `--model`: Path to trained model (default: `face_mask_detection_final.h5`)
- `--camera`: Camera device ID (default: 0)
- `--confidence`: Minimum confidence threshold (default: 0.5)

### 3. Controls

- **'q'**: Quit the application
- **'s'**: Save screenshot
- **ESC**: Exit (alternative)

## 📊 Model Performance

### Training Results
- **Architecture**: Hybrid CNN + MobileNetV2
- **Input Resolution**: 224×224 pixels
- **Training Strategy**: Transfer learning + Fine-tuning
- **Data Augmentation**: Rotation, flip, zoom, contrast adjustment

### Key Features
- **Real-time Performance**: ~30 FPS on average hardware
- **High Accuracy**: Optimized for face mask detection
- **Robust Detection**: Works with various lighting conditions
- **Low Latency**: Efficient preprocessing and inference

## 🔧 Technical Details

### Data Preprocessing
1. **Image Resizing**: All images resized to 224×224
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Data Augmentation**: Applied only to training set
4. **Batch Processing**: Efficient data pipeline with prefetching

### Model Training Process
1. **Phase 1**: Train custom head with frozen MobileNetV2
2. **Phase 2**: Fine-tune top layers of MobileNetV2
3. **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
4. **Regularization**: Dropout and L2 regularization

### Real-time Detection Pipeline
1. **Face Detection**: Haar Cascade Classifier
2. **Face Extraction**: ROI extraction and preprocessing
3. **Classification**: Model inference
4. **Visualization**: Bounding boxes and confidence scores

## 📈 Performance Optimization

### Training Optimizations
- **Mixed Precision**: Faster training on modern GPUs
- **Data Pipeline**: TensorFlow's `tf.data` API with prefetching
- **Callbacks**: Intelligent learning rate scheduling
- **Model Checkpointing**: Save best performing models

### Inference Optimizations
- **Batch Processing**: Efficient face processing
- **Model Caching**: Load model once, use multiple times
- **Memory Management**: Proper cleanup and resource management
- **FPS Monitoring**: Real-time performance tracking

## 🎨 Visual Features

### Real-time Display
- **Green Box**: Person wearing mask (confidence > threshold)
- **Red Box**: Person not wearing mask (confidence > threshold)
- **Gray Box**: Low confidence detection
- **Info Overlay**: FPS, timestamp, controls

### Screenshots
- Automatic timestamping
- Organized in `screenshots/` directory
- Full resolution capture

## 🔍 Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```
   Error: Could not open camera 0
   ```
   - Try different camera IDs (1, 2, etc.)
   - Check camera permissions
   - Ensure camera is not used by another application

2. **Model Loading Error**
   ```
   Error loading model: [error message]
   ```
   - Verify model file exists
   - Check TensorFlow version compatibility
   - Ensure sufficient memory

3. **Poor Detection Performance**
   - Adjust confidence threshold
   - Ensure good lighting
   - Position face clearly in frame
   - Check camera resolution

### Performance Tips

1. **Improve Speed**
   - Lower camera resolution
   - Reduce face detection parameters
   - Use GPU acceleration if available

2. **Improve Accuracy**
   - Increase confidence threshold
   - Ensure consistent lighting
   - Train with more diverse data

## 📝 Dataset Requirements

### Expected Structure
```
Face Mask Dataset/
├── Train/
│   ├── WithMask/
│   │   ├── image1.jpg
│   │   └── ...
│   └── WithoutMask/
│       ├── image1.jpg
│       └── ...
├── Validation/
│   ├── WithMask/
│   └── WithoutMask/
└── Test/
    ├── WithMask/
    └── WithoutMask/
```

### Data Quality Guidelines
- **Image Resolution**: Minimum 224×224 pixels
- **Face Visibility**: Clear, unobstructed faces
- **Lighting**: Varied lighting conditions
- **Angles**: Multiple face angles and orientations
- **Balance**: Equal distribution of masked/unmasked samples

## 🔬 Model Evaluation

### Metrics Used
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for each class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Validation Strategy
- **Train/Validation/Test Split**: Separate datasets
- **Cross-Validation**: Robust performance estimation
- **Stratified Sampling**: Balanced class distribution

## 🚀 Deployment Options

### Local Deployment
- **Real-time Detection**: Webcam-based detection
- **Batch Processing**: Process multiple images
- **API Integration**: Flask/FastAPI wrapper

### Edge Deployment
- **TensorFlow Lite**: Mobile/embedded deployment
- **ONNX**: Cross-platform inference
- **TensorRT**: NVIDIA GPU optimization

### Cloud Deployment
- **TensorFlow Serving**: Scalable model serving
- **Docker**: Containerized deployment
- **REST API**: Web service integration

## 📚 References and Resources

### Technical Papers
- MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Transfer Learning for Computer Vision
- Data Augmentation Techniques for Deep Learning

### Libraries Used
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install dependencies
3. Download/prepare dataset
4. Run training script
5. Test real-time detection

### Code Style
- Follow PEP 8 guidelines
- Add comprehensive comments
- Include error handling
- Write unit tests

## 🙋‍♂️ Support

For questions, issues, or suggestions:
1. Check the troubleshooting section
2. Review common issues
3. Create an issue with detailed description
4. Include system specifications and error logs

---

**Happy Detecting! 😷🤖**
