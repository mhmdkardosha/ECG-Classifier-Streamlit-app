# ECG Heart Rate Condition Predictor

A Streamlit application for extracting and classifying heart conditions from ECG images using computer vision and machine learning.

## Features

- **Image Upload**: Upload an ECG image (JPG, JPEG, PNG)
- **Waveform Extraction**: Advanced OpenCV-based pipeline to extract 187-point ECG waveform from the image
- **Signal Visualization**: View the extracted waveform as a line chart
- **Classification**: Predicts one of five heart conditions using a pre-trained model
- **User Feedback**: Warnings and error messages for low-quality or unclear images

## Heartbeat Classes

- Normal beat (N)
- Supraventricular premature beat (S)
- Premature ventricular contraction (V)
- Fusion of ventricular and normal beat (F)
- Unclassifiable beat (Q)

## Installation

### Prerequisites

- Python 3.7 or higher

### Setup

1. Clone or download this repository.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Ensure `heartbeat_model.pkl` is present in the project directory.

## Usage

1. Start the app:

   ```powershell
   streamlit run app.py
   ```

2. Open the provided local URL in your browser.
3. Upload a clear image of a single ECG period.
4. Click **Analyze ECG** to extract and classify the waveform.

## How It Works

- The app preprocesses the image (blurring, contrast enhancement, thresholding, grid removal).
- It detects the ECG waveform, samples 187 points, and normalizes the signal.
- The extracted data is fed to a machine learning model for classification.
- Results and probabilities for each class are displayed.

## File Structure

```
app.py                  # Main Streamlit app
heartbeat_model.pkl     # Pre-trained classification model
requirements.txt        # Python dependencies
README.md               # Project documentation
```

## Limitations

- Works best with clear, high-contrast ECG images showing a single period
- Not for clinical or diagnostic use
- Image quality and grid artifacts can affect accuracy

## License

MIT License

## Disclaimer

This app is for demonstration and educational purposes only. Not for medical use.
