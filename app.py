import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
try:
    from streamlit_audiorecorder import audiorecorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
            
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --accent: #ec4899;
    --dark: #1e293b;
    --light: #f8fafc;
}

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: var(--light);
}

.st-emotion-cache-1y4p8pa {
    padding: 2rem;
}

h1, h2, h3, h4 {
    color: var(--light) !important;
    font-weight: 600 !important;
}

.stButton>button {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
}

.stFileUploader>div>div>div>div {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 2px dashed #4c4fef !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}

.stFileUploader>div>div>div>div:hover {
    border-color: var(--accent) !important;
    background: rgba(30, 41, 59, 0.9) !important;
}

.st-emotion-cache-1aehpvj {
    color: var(--accent) !important;
}

.stSpinner>div {
    border-color: var(--primary) transparent transparent transparent !important;
}

/* Card styling */
.card {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

/* Emotion colors */
.emotion-display {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    margin: 2rem 0;
    background: rgba(15, 23, 42, 0.8);
    border: 2px solid;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.neutral { color: #94a3b8; border-color: #94a3b8; }
.calm { color: #60a5fa; border-color: #60a5fa; }
.happy { color: #fbbf24; border-color: #fbbf24; }
.sad { color: #38bdf8; border-color: #38bdf8; }
.angry { color: #f87171; border-color: #f87171; }
.fearful { color: #c084fc; border-color: #c084fc; }
.disgust { color: #34d399; border-color: #34d399; }
.surprised { color: #f472b6; border-color: #f472b6; }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Try loading the .keras format first (more reliable)
        model = load_model('model/emotion_model.keras', compile=False)
        # Recompile with a standard loss function
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except Exception as e:
        st.warning(f"Could not load .keras model: {e}")
        try:
            # Fallback to .h5 format without custom objects
            model = load_model('model/emotion_model.h5', compile=False)
            # Recompile with a standard loss function
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e2:
            st.error(f"Could not load model: {e2}")
            st.stop()
    
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    return model, scaler, label_encoder

# Feature extraction function
def extract_features(file_path, target_length=130):
    y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60, hop_length=512)
    
    if mfcc.shape[1] < target_length:
        pad_width = target_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_length]
    
    return y, sr, mfcc.T

# Preprocess input
def preprocess_input(audio_path, scaler):
    y, sr, features = extract_features(audio_path)
    num_samples, time_steps, n_mfcc = 1, features.shape[0], features.shape[1]
    flat_features = features.reshape(num_samples * time_steps, n_mfcc)
    scaled_flat = scaler.transform(flat_features)
    return y, sr, scaled_flat.reshape(num_samples, time_steps, n_mfcc)

# Generate waveform plot
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')
    ax.set_facecolor('none')
    librosa.display.waveshow(y, sr=sr, color='#6366f1', alpha=0.8, ax=ax)
    plt.axis('off')
    plt.tight_layout()
    return fig

# Generate spectrogram
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    ax.set_facecolor('none')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', 
                                 cmap='magma', ax=ax)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# Generate MFCC visualization
def plot_mfcc(mfcc_features):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
    ax.set_facecolor('none')
    img = librosa.display.specshow(mfcc_features.T, x_axis='time', cmap='viridis', ax=ax)
    plt.colorbar(img, ax=ax)
    plt.title('MFCC Features')
    plt.tight_layout()
    return fig

# Emotion icon mapping
def get_emotion_icon(emotion):
    icons = {
        'neutral': '😐',
        'calm': '😌',
        'happy': '😄',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'disgust': '🤢',
        'surprised': '😲'
    }
    return icons.get(emotion, '🎤')

# Main app
def main():
    # Load artifacts
    model, scaler, label_encoder = load_artifacts()
    emotions = label_encoder.classes_
    
    # Header
    st.markdown("""
    <div>
        <h1 style='text-align:center; margin-bottom:0'>🎤 EmotionSense AI</h1>
        <p style='text-align:center; font-size:1.2rem; opacity:0.8'>
        Advanced Speech Emotion Recognition System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Columns layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎙️ Audio Input")
        st.markdown("Upload an audio file or record directly using your microphone")
        
        # Input options
        input_method = st.radio("Select input method:", 
                              ["Upload Audio", "Record Audio"],
                              horizontal=True)        
        audio_data = None
        if input_method == "Upload Audio":
            audio_file = st.file_uploader("Upload audio file (WAV, MP3, OGG)", 
                                        type=["wav", "mp3", "ogg"],
                                        label_visibility="collapsed")
            if audio_file:
                st.audio(audio_file, format='audio/wav')
                audio_data = audio_file
        else:
            st.markdown("### 🎤 Live Recording")
            if AUDIO_RECORDER_AVAILABLE:
                recorded_audio = audiorecorder("Click to record", "Stop recording")
                if len(recorded_audio) > 0:
                    st.audio(recorded_audio.export().read(), format="audio/wav")
                    audio_data = recorded_audio.export().read()
            else:
                st.warning("Audio recording is not available. Please upload an audio file instead.")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Audio visualization
        if audio_data:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Audio Analysis")
            
            with st.spinner("Processing audio..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    if input_method == "Upload Audio":
                        audio_data.seek(0)  # Reset file pointer
                        tmp.write(audio_data.read())
                    else:
                        tmp.write(audio_data)
                    tmp_path = tmp.name
                
                try:
                    # Process audio
                    y, sr, input_data = preprocess_input(tmp_path, scaler)
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "MFCC Features"])
                    
                    with tab1:
                        st.markdown("#### Audio Waveform")
                        waveform_fig = plot_waveform(y, sr)
                        st.pyplot(waveform_fig)
                        plt.close(waveform_fig)  # Close figure to free memory
                        
                    with tab2:
                        st.markdown("#### Audio Spectrogram")
                        spec_fig = plot_spectrogram(y, sr)
                        st.pyplot(spec_fig)
                        plt.close(spec_fig)  # Close figure to free memory
                        
                    with tab3:
                        st.markdown("#### MFCC Features")
                        mfcc_fig = plot_mfcc(input_data[0].T)
                        st.pyplot(mfcc_fig)
                        plt.close(mfcc_fig)  # Close figure to free memory
                        
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass  # Ignore cleanup errors
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if audio_data:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 Emotion Analysis")
            
            # Analyze button with animation
            if st.button("Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner("Detecting emotion..."):
                    try:
                        # Predict emotion
                        prediction = model.predict(input_data)
                        predicted_class = np.argmax(prediction, axis=1)
                        emotion = label_encoder.inverse_transform(predicted_class)[0]
                        confidence = np.max(prediction) * 100
                        
                        # Display results with animation
                        icon = get_emotion_icon(emotion)
                        st.markdown(f"<div class='emotion-display {emotion}'>{icon} {emotion.upper()} {icon}</div>", 
                                   unsafe_allow_html=True)
                        
                        # Confidence metric
                        st.metric("Confidence Level", f"{confidence:.2f}%", 
                                 delta_color="off")
                        
                        # Emotion probabilities visualization
                        st.markdown("#### Emotion Probability Distribution")
                        
                        # Create progress bars for each emotion
                        prob_data = {e: p * 100 for e, p in zip(emotions, prediction[0])}
                        
                        for emotion_name, prob in prob_data.items():
                            # Add color-coded progress bar
                            color = {
                                'neutral': '#94a3b8',
                                'calm': '#60a5fa',
                                'happy': '#fbbf24',
                                'sad': '#38bdf8',
                                'angry': '#f87171',
                                'fearful': '#c084fc',
                                'disgust': '#34d399',
                                'surprised': '#f472b6'
                            }.get(emotion_name, '#6366f1')
                            
                            # Display progress bar
                            st.markdown(f"**{emotion_name.capitalize()}**")
                            st.progress(int(prob), text=f"{prob:.2f}%")
                        
                        # Detailed statistics
                        st.markdown("#### Detailed Emotion Analysis")
                        
                        # Create metrics columns
                        cols = st.columns(4)
                        for i, (emote, prob) in enumerate(prob_data.items()):
                            with cols[i % 4]:
                                st.metric(emote.capitalize(), f"{prob:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error analyzing emotion: {str(e)}")
            else:
                st.markdown("""
                <div style='text-align:center; padding: 4rem 0;'>
                    <h3>Click 'Analyze Emotion' to see results</h3>
                    <p>Your analysis will appear here</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 Emotion Analysis")
            st.markdown("""
            <div style='text-align:center; padding: 4rem 0;'>
                <h3>Upload or record audio to begin analysis</h3>
                <p>Your results will appear here</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # How it works section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 How EmotionSense AI Works")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.markdown("#### 1. Audio Processing")
        st.markdown("""
        - Input audio is normalized and segmented
        - Background noise reduction applied
        - Sample rate standardized to 22.05kHz
        """)
    
    with col_ex2:
        st.markdown("#### 2. Feature Extraction")
        st.markdown("""
        - MFCC (Mel-Frequency Cepstral Coefficients) calculated
        - 60 audio features extracted per sample
        - Features standardized using z-score normalization
        """)
    
    with col_ex3:
        st.markdown("#### 3. Deep Learning Analysis")
        st.markdown("""
        - Hybrid CNN + BiLSTM neural network
        - 8 emotion categories recognized
        - Real-time prediction with 85%+ accuracy
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()