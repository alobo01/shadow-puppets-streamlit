# 🖐️ Shadow Puppet Parametrisation Viewer

An interactive web application that captures hand gestures using your webcam, extracts a parametric representation via MediaPipe, and visualizes the hand geometry along with shadow projections onto a virtual wall.

## ✨ Features

- **Real-time Hand Capture**: Uses MediaPipe's hand tracking to capture hand landmarks from your webcam
- **Parametric Representation**: Extracts meaningful parameters including:
  - Palm-thumb plane angles
  - Inter-finger angles
  - Individual joint angles for each finger
- **Invariant Normalization**: Applies geometric invariances (translation, scale, rotation) for consistent representation
- **Interactive 3D Visualization**: View captured hands in 3D space with torch and projection cone
- **Shadow Projection**: Projects hand shadows onto a virtual wall from a torch light source
- **Parameter Editor**: Manually adjust hand parameters and see the synthetic hand reconstruction in real-time
- **Freeze & Edit**: Capture a pose and fine-tune its parameters

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for live hand capture)
- Modern web browser

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alobo01/shadow-puppets-streamlit.git
cd shadow-puppets-streamlit
```

2. Install the required dependencies:
```bash
pip install -r "requirements (1).txt"
```

Or install packages individually:
```bash
pip install streamlit>=1.36 streamlit-webrtc>=0.47 mediapipe>=0.10 opencv-python-headless>=4.9 numpy>=1.24 plotly>=5.20 av>=11.0.0
```

### Running the App

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### Tab 1: Live Capture
1. Allow camera access when prompted
2. Show your hand to the webcam
3. The app will automatically detect and track your hand
4. View the extracted parameters in real-time
5. See the normalized 3D hand representation
6. Click "Freeze to editor" to copy current parameters for editing

### Tab 2: Parameter Editor & Synthesis
1. Adjust hand parameters using intuitive sliders:
   - **Palm-thumb angle**: Controls the thumb's plane relative to the palm
   - **Inter-finger angles**: Controls spacing between fingers
   - **Joint angles**: Controls the bend at each finger joint
2. View the synthetic hand reconstruction in real-time
3. Experiment with different poses and shapes

### Tab 3: Projection
1. Choose between captured or synthesized hand
2. View the 3D setup with torch position and projection cone
3. See the shadow projection on the virtual wall in 2D

## 🔬 Technical Details

### Geometric Invariances

The app applies three key invariances to normalize hand poses:

1. **Translation**: Centers the hand at the palm (MCP) center
2. **Scale**: Normalizes to unit distance between index and pinky MCPs
3. **Rotation**: Aligns palm plane to +Z axis and wraps in-plane rotation to ±90°

### Hand Parametrization

The parametric representation includes:
- **φ_thumb**: Palm-thumb plane angle
- **Inter-finger angles**: T-I, I-M, M-R, R-P angles
- **Joint angles**: 3 angles per finger (thumb, index, middle, ring, pinky)

### MediaPipe Integration

Uses MediaPipe's hand tracking solution to detect 21 3D landmarks representing:
- Wrist (1 point)
- Thumb (4 points)
- Index, Middle, Ring, Pinky fingers (4 points each)

## 📁 Project Structure

```
shadow-puppets-streamlit/
├── app.py                  # Main Streamlit application
├── requirements (1).txt    # Python dependencies
├── sample_params.json      # Sample hand parameters
└── README.md              # This file
```

## 🎨 Customization

### Modify Torch and Wall Setup

Edit the `DEFAULT_SETUP` in `app.py`:
```python
DEFAULT_SETUP = Setup(
    T=np.array([0.0, 0.0, -2.0]),  # Torch position
    wall_n=unit(np.array([0.0, 0.0, 1.0])),  # Wall normal
    wall_d=4.0,  # Wall distance
    base_radius=2.0  # Wall circle radius
)
```

### Adjust Hand Synthesis Parameters

Modify segment lengths and default angles in the `synthesize_from_params()` function.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive 3D visualizations

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an illustrative tool for exploring hand parametrization. The synthesis is not a full biomechanical model but serves to demonstrate parameter effects.
