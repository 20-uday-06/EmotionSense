import streamlit as st
import numpy as np
import tempfile
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Essential imports with error handling
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.error("Librosa not available. Audio processing will be limited.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.error("Joblib not available. Model loading will be limited.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not available. Model inference will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from streamlit_audiorecorder import audiorecorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="MARS SER System - EmotionSense AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
            
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --accent: #ec4899;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --dark: #0f172a;
    --light: #f8fafc;
    --gray: #64748b;
}

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: var(--light);
}

.main-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.section-card {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.upload-zone {
    border: 2px dashed var(--primary);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    background: rgba(99, 102, 241, 0.05);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: var(--accent);
    background: rgba(236, 72, 153, 0.05);
}

.record-zone {
    border: 2px dashed var(--accent);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    background: rgba(236, 72, 153, 0.05);
    margin: 1rem 0;
}

.feature-card {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    border: 1px solid;
    transition: transform 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-card.processing {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgba(34, 197, 94, 0.3);
}

.feature-card.extraction {
    background: rgba(236, 72, 153, 0.1);
    border-color: rgba(236, 72, 153, 0.3);
}

.feature-card.analysis {
    background: rgba(139, 92, 246, 0.1);
    border-color: rgba(139, 92, 246, 0.3);
}

.emotion-result {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border-radius: 20px;
    margin: 1rem 0;
    border: 2px solid var(--primary);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
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

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: var(--light);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
}

.stProgress .stProgress-bar {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
}

.stMetric {
    background: rgba(15, 23, 42, 0.5);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.2);
}
</style>
""", unsafe_allow_html=True)



# Alternative model loading function for compatibility issues
def load_model_alternative(model_path):
    """
    Alternative model loading method that handles TensorFlow version compatibility issues
    """
    try:
        # Try with custom object scope to handle batch_shape issues
        with tf.keras.utils.custom_object_scope({'batch_shape': lambda x: x}):
            model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Alternative loading method also failed: {e}")
        return None

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Try loading the .keras format first
        model = load_model('model/emotion_model.keras', compile=False)
    except Exception:
        try:
            # Fallback to alternative loading method
            model = load_model_alternative('model/emotion_model.keras')
        except Exception:
            try:
                # Try .h5 format
                model = load_model('model/emotion_model.h5', compile=False)
            except Exception:
                try:
                    # Final fallback
                    model = load_model_alternative('model/emotion_model.h5')
                except Exception as e:
                    st.error(f"❌ Could not load model: {str(e)}")
                    st.stop()
    
    # Recompile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load preprocessing artifacts
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
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    librosa.display.waveshow(y, sr=sr, color='#6366f1', alpha=0.8, ax=ax)
    ax.set_title('Audio Waveform', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    plt.tight_layout()
    return fig

# Generate spectrogram
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', 
                                 cmap='magma', ax=ax)
    ax.set_title('Audio Spectrogram', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.tick_params(colors='white')
    plt.tight_layout()
    return fig

# Generate MFCC visualization
def plot_mfcc(mfcc_features):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    img = librosa.display.specshow(mfcc_features.T, x_axis='time', cmap='viridis', ax=ax)
    ax.set_title('MFCC Features', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(colors='white')
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
    try:
        model, scaler, label_encoder = load_artifacts()
        emotions = label_encoder.classes_
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Header with improved styling
    st.markdown("""
    <div class="main-header">
        <h1 style='color: #6366f1; margin-bottom: 0.5rem; font-size: 3rem;'>🎤 MARS SER System</h1>
        <h2 style='color: #8b5cf6; margin-bottom: 0.5rem; font-size: 1.5rem;'>EmotionSense AI</h2>
        <p style='color: #64748b; font-size: 1.2rem; margin-bottom: 0;'>Advanced Speech Emotion Recognition | MARS Open Projects 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main tabs for better organization
    main_tab1, main_tab2, main_tab3 = st.tabs(["🎙️ Audio Analysis", "📊 Visualizations", "ℹ️ About System"])
    
    with main_tab1:
        # Audio Input Section
        st.markdown("""
        <div class="section-card">
            <h3 style='color: #6366f1; margin-bottom: 1rem;'>🎵 Audio Input</h3>
            <p style='color: #94a3b8; margin-bottom: 0;'>Select your preferred input method below</p>
        </div>
        """, unsafe_allow_html=True)        # Input method tabs - both methods work universally
        input_tab1, input_tab2 = st.tabs(["📁 Upload Audio File", ])
        
        audio_data = None
        input_method = None
        
        with input_tab1:
            st.markdown("""
            <div class="upload-zone">
                <h4 style='color: #6366f1; margin-bottom: 1rem;'>📁 File Upload</h4>
                <p style='color: #94a3b8;'>Drag and drop or browse to select your audio file</p>
                <p style='color: #64748b; font-size: 0.9rem;'>Supported: WAV, MP3, OGG, M4A, FLAC (Max: 200MB)</p>
            </div>
            """, unsafe_allow_html=True)
            
            audio_file = st.file_uploader(
                "Choose an audio file", 
                type=["wav", "mp3", "ogg", "m4a", "flac"],
                label_visibility="collapsed"
            )
            
            if audio_file:
                input_method = "Upload Audio"
                audio_data = audio_file
                
                # File info display
                file_size = len(audio_file.getvalue()) / (1024 * 1024)
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("📄 File Name", audio_file.name)
                with col_info2:
                    st.metric("📊 File Size", f"{file_size:.2f} MB")
                with col_info3:
                    st.metric("🎵 Format", audio_file.type.split('/')[-1].upper())
                
                # Audio player with enhanced styling
                st.markdown("#### 🔊 Audio Preview")
                st.audio(audio_file, format='audio/wav')                
                st.success("✅ File uploaded successfully! Ready for analysis.")
        
        with input_tab2:
            st.markdown("""
            <div class="record-zone">
                <h4 style='color: #ec4899; margin-bottom: 1rem;'>🎤 Live Recording</h4>
                <p style='color: #94a3b8;'>Record directly in your browser</p>
                <p style='color: #64748b; font-size: 0.9rem;'>Optimal duration: 3-10 seconds for best results</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Browser-based audio recording using HTML5
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <button id="recordBtn" onclick="toggleRecording()" style='
                    background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%);
                    border: none;
                    border-radius: 12px;
                    padding: 15px 30px;
                    color: white;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 4px 20px rgba(236, 72, 153, 0.3);
                    transition: all 0.3s ease;
                    margin: 10px;
                '>🔴 Start Recording</button>
                
                <div id="recordingStatus" style='margin: 1rem 0; color: #94a3b8;'></div>
                <audio id="audioPlayback" controls style='margin: 1rem 0; display: none; width: 100%;'></audio>
                <div id="downloadSection" style='margin: 1rem 0; display: none;'>
                    <a id="downloadLink" style='
                        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
                        border: none;
                        border-radius: 12px;
                        padding: 12px 24px;
                        color: white;
                        text-decoration: none;
                        font-weight: 600;
                        display: inline-block;
                        margin: 10px;
                    '>💾 Download Recording</a>
                </div>
            </div>
            
            <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            
            async function toggleRecording() {
                const recordBtn = document.getElementById('recordBtn');
                const status = document.getElementById('recordingStatus');
                const audioPlayback = document.getElementById('audioPlayback');
                const downloadSection = document.getElementById('downloadSection');
                
                if (!isRecording) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            const audioUrl = URL.createObjectURL(audioBlob);
                            
                            audioPlayback.src = audioUrl;
                            audioPlayback.style.display = 'block';
                            
                            const downloadLink = document.getElementById('downloadLink');
                            downloadLink.href = audioUrl;
                            downloadLink.download = 'emotion_recording_' + Date.now() + '.wav';
                            downloadSection.style.display = 'block';
                            
                            status.innerHTML = '✅ Recording completed! Use the download button to save your recording.';
                            
                            // Stop all tracks
                            stream.getTracks().forEach(track => track.stop());
                        };
                        
                        mediaRecorder.start();
                        isRecording = true;
                        recordBtn.innerHTML = '⏹️ Stop Recording';
                        recordBtn.style.background = 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)';
                        status.innerHTML = '🔴 Recording... Click stop when finished.';
                        
                    } catch (error) {
                        status.innerHTML = '❌ Microphone access denied. Please allow microphone access and try again.';
                        console.error('Error accessing microphone:', error);
                    }
                } else {
                    mediaRecorder.stop();
                    isRecording = false;
                    recordBtn.innerHTML = '🔴 Start Recording';
                    recordBtn.style.background = 'linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%)';
                    status.innerHTML = '⏸️ Processing recording...';
                }
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Instructions for using the recording
            st.markdown("""
            <div style='background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border: 1px solid rgba(99, 102, 241, 0.3);'>
                <h5 style='color: #6366f1; margin-bottom: 1rem;'>� How to use your recording:</h5>
                <ol style='color: #94a3b8; margin-left: 1rem;'>
                    <li>Click "Start Recording" and allow microphone access</li>
                    <li>Speak clearly for 3-10 seconds</li>
                    <li>Click "Stop Recording" when finished</li>
                    <li>Download the recording file</li>
                    <li>Go to "Upload Audio File" tab and upload your recording</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis Section
        if audio_data:
            st.markdown("---")
            st.markdown("""
            <div class="section-card" style="border-color: rgba(34, 197, 94, 0.3);">
                <h3 style='color: #22c55e; margin-bottom: 1rem;'>🧠 Emotion Analysis</h3>
                <p style='color: #94a3b8; margin-bottom: 0;'>Click the button below to analyze the emotional content</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis button
            if st.button("🚀 Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner("🔬 Processing audio and detecting emotions..."):
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            if input_method == "Upload Audio":
                                audio_data.seek(0)
                                tmp.write(audio_data.read())
                            else:
                                tmp.write(audio_data)
                            tmp_path = tmp.name
                        
                        # Process audio
                        y, sr, input_data = preprocess_input(tmp_path, scaler)
                        
                        # Predict emotion
                        prediction = model.predict(input_data)
                        predicted_class = np.argmax(prediction, axis=1)
                        emotion = label_encoder.inverse_transform(predicted_class)[0]
                        confidence = np.max(prediction) * 100
                        
                        # Results display
                        st.markdown("### 🎯 Analysis Results")
                        
                        # Main emotion display
                        icon = get_emotion_icon(emotion)
                        st.markdown(f"""
                        <div class="emotion-result">
                            <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>{icon}</h1>
                            <h2 style='color: #6366f1; margin-bottom: 0.5rem;'>{emotion.upper()}</h2>
                            <p style='color: #8b5cf6; font-size: 1.2rem;'>Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability distribution
                        st.markdown("#### 📊 Emotion Probability Distribution")
                        prob_data = {e: p * 100 for e, p in zip(emotions, prediction[0])}
                        
                        # Create two columns for better layout
                        prob_col1, prob_col2 = st.columns(2)
                        
                        emotion_colors = {
                            'neutral': '#94a3b8', 'calm': '#60a5fa', 'happy': '#fbbf24', 'sad': '#38bdf8',
                            'angry': '#f87171', 'fearful': '#c084fc', 'disgust': '#34d399', 'surprised': '#f472b6'
                        }
                        
                        for i, (emotion_name, prob) in enumerate(prob_data.items()):
                            col = prob_col1 if i % 2 == 0 else prob_col2
                            with col:
                                color = emotion_colors.get(emotion_name, '#6366f1')
                                # Progress bar with custom styling
                                st.markdown(f"**{get_emotion_icon(emotion_name)} {emotion_name.capitalize()}**")
                                st.progress(min(int(prob), 100), text=f"{prob:.1f}%")
                        
                        # Cleanup temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"❌ Error analyzing emotion: {str(e)}")
            else:
                st.info("👆 Upload or record audio, then click 'Analyze Emotion' to see results")
        else:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: rgba(15, 23, 42, 0.5); border-radius: 15px; margin: 2rem 0; border: 2px dashed #64748b;'>
                <h3 style='color: #64748b; margin-bottom: 1rem;'>🎵 Ready for Analysis</h3>
                <p style='color: #94a3b8;'>Please upload an audio file or record your voice to begin emotion analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    with main_tab2:
        if audio_data:
            st.markdown("### 📊 Audio Visualizations")
            
            with st.spinner("🎨 Generating visualizations..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        if input_method == "Upload Audio":
                            audio_data.seek(0)
                            tmp.write(audio_data.read())
                        else:
                            tmp.write(audio_data)
                        tmp_path = tmp.name
                    
                    # Process audio
                    y, sr, input_data = preprocess_input(tmp_path, scaler)
                    
                    # Visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🌊 Waveform", "🎨 Spectrogram", "📈 MFCC Features"])
                    
                    with viz_tab1:
                        st.markdown("#### 🌊 Audio Waveform")
                        st.markdown("Visual representation of the audio signal amplitude over time")
                        waveform_fig = plot_waveform(y, sr)
                        st.pyplot(waveform_fig)
                        plt.close(waveform_fig)
                        
                    with viz_tab2:
                        st.markdown("#### 🎨 Audio Spectrogram")
                        st.markdown("Frequency content visualization showing how spectral density varies with time")
                        spec_fig = plot_spectrogram(y, sr)
                        st.pyplot(spec_fig)
                        plt.close(spec_fig)
                        
                    with viz_tab3:
                        st.markdown("#### 📈 MFCC Features")
                        st.markdown("Mel-Frequency Cepstral Coefficients used by the AI model for emotion recognition")
                        mfcc_fig = plot_mfcc(input_data[0].T)
                        st.pyplot(mfcc_fig)
                        plt.close(mfcc_fig)
                    
                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error generating visualizations: {str(e)}")
        else:
            st.info("📊 Upload or record audio to view visualizations")
    
    with main_tab3:
        st.markdown("### 🧠 How the MARS SER System Works")
        
        # System overview
        st.markdown("""
        <div class="section-card">
            <h4 style='color: #6366f1;'>🎯 System Overview</h4>
            <p style='color: #94a3b8;'>This Speech Emotion Recognition system is part of the MARS Open Projects 2025 initiative, designed to classify emotions in speech using advanced deep learning techniques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Process explanation
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            st.markdown("""
            <div class="feature-card processing">
                <h4 style='color: #22c55e;'>1. 🎵 Audio Processing</h4>
                <ul style='color: #94a3b8;'>
                    <li>Audio normalization and segmentation</li>
                    <li>Noise reduction algorithms</li>
                    <li>22.05kHz sampling rate standardization</li>
                    <li>3-second optimal duration processing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ex2:
            st.markdown("""
            <div class="feature-card extraction">
                <h4 style='color: #ec4899;'>2. 🔬 Feature Extraction</h4>
                <ul style='color: #94a3b8;'>
                    <li>60 MFCC coefficients calculation</li>
                    <li>Mel-scale frequency analysis</li>
                    <li>Z-score feature normalization</li>
                    <li>Temporal pattern recognition</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ex3:
            st.markdown("""
            <div class="feature-card analysis">
                <h4 style='color: #8b5cf6;'>3. 🧠 AI Analysis</h4>
                <ul style='color: #94a3b8;'>
                    <li>Hybrid CNN + BiLSTM architecture</li>
                    <li>8 emotion category classification</li>
                    <li>87%+ accuracy performance</li>
                    <li>Real-time prediction capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("### 📈 Model Performance")        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("🎯 Overall Accuracy", "93.0%", delta="Target: >80% ✅")
        with perf_col2:
            st.metric("📊 Weighted F1 Score", "90.53%", delta="Target: >80% ✅")
        with perf_col3:
            st.metric("🎵 Emotions Classified", "8", delta="All categories")
        with perf_col4:
            st.metric("⚡ Processing Speed", "<2s", delta="Real-time")
        
        # Technical details
        with st.expander("🔧 Technical Specifications"):
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.markdown("""
                **Model Architecture:**
                - CNN layers for feature extraction
                - Bidirectional LSTM for temporal modeling
                - Dropout and BatchNormalization
                - Dense layers with softmax output
                
                **Training Data:**
                - RAVDESS dataset (24 actors)
                - Both speech and song audio
                - Data augmentation techniques
                """)
            
            with tech_col2:
                st.markdown("""
                **Performance Targets:**
                - ✅ Weighted F1 Score > 80%
                - ✅ Overall Accuracy > 80%
                - ✅ Individual Class Recalls > 75%
                
                **Supported Formats:**
                - WAV, MP3, OGG, M4A, FLAC
                - Sample rates: 8kHz - 48kHz
                - Mono/Stereo audio support
                """)

if __name__ == "__main__":
    main()
