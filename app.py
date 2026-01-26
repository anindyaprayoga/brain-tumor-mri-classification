import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #999;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3, .metric-card p {
        color: white !important;
    }
    .prediction-box {
        background: rgba(248, 249, 250, 0.05);
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box p {
        color: inherit;
    }
    .info-box {
        background: rgba(33, 150, 243, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: inherit;
    }
    .info-box strong {
        color: inherit;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.15);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: inherit;
    }
    .warning-box strong {
        color: inherit;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

CLASS_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'Tumor originating from glial cells in the brain or spinal cord',
        'color': '#FF6B6B'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'Tumor growing from the meninges (protective membranes of the brain)',
        'color': '#4ECDC4'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'No tumor detected in the MRI image',
        'color': '#45B7D1'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'Tumor growing in the pituitary gland',
        'color': '#FFA07A'
    }
}

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_model():
    model_path = "custom_cnn_final.h5"
    try:
        model = keras.models.load_model(model_path, compile=False)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Make sure 'custom_cnn_final.h5' exists in the same folder as app.py")
        
        try:
            st.info("Trying alternative loading method...")
            custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e2:
            st.error(f"Alternative method also failed: {str(e2)}")
            return None

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_tta(model, img_array, n_augmentations=10):
    predictions = []
    
    base_pred = model.predict(img_array, verbose=0)
    predictions.append(base_pred)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    tta_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )
    
    for _ in range(n_augmentations):
        aug_iterator = tta_gen.flow(img_array, batch_size=1, shuffle=False)
        img_aug = next(iter(aug_iterator))
        pred_aug = model.predict(img_aug, verbose=0)
        predictions.append(pred_aug)
    
    return np.mean(predictions, axis=0)

def create_confidence_chart(predictions, class_names):
    fig = go.Figure()
    
    colors = [CLASS_INFO[cls]['color'] for cls in class_names]
    
    fig.add_trace(go.Bar(
        x=predictions[0] * 100,
        y=[CLASS_INFO[cls]['name'] for cls in class_names],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f'{pred*100:.2f}%' for pred in predictions[0]],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Confidence Scores',
        xaxis_title='Confidence (%)',
        yaxis_title='Tumor Type',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 100], gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    return fig

def main():
    st.markdown('<p class="main-header">Brain Tumor Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Brain Tumor Classification from MRI Images using Deep Learning</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page:", ["Classification", "About Model", "Information"])
        
        st.markdown("---")
        st.subheader("Model Settings")
        use_tta = st.checkbox("Enable TTA (Test Time Augmentation)", value=True)
        if use_tta:
            n_augmentations = st.slider("Number of Augmentations", 5, 20, 10)
        else:
            n_augmentations = 0
        
        st.markdown("---")
        st.caption("Developed for Medical Image Analysis")
    
    if page == "Classification":
        show_classification_page(use_tta, n_augmentations)
    elif page == "About Model":
        show_about_model_page()
    else:
        show_information_page()

def show_classification_page(use_tta, n_augmentations):
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload MRI Image")
        
        st.markdown("""
        <div class="info-box">
        <strong>Upload Requirements:</strong><br>
        • Format: JPG, JPEG, PNG<br>
        • Recommended: Brain MRI scan image<br>
        • Clear and high quality image
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload brain MRI scan image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            
            img_array = preprocess_image(image)
            
            if st.button("Analyze Image", type="primary"):
                st.session_state.analyze_results = analyze_image(img_array, use_tta, n_augmentations)
    
    with col2:
        st.subheader("Analysis Results")
        if uploaded_file is None:
            st.info("Please upload an MRI image to begin analysis")
        elif 'analyze_results' not in st.session_state:
            st.info("Click 'Analyze Image' button to start analysis")
        else:
            display_results(st.session_state.analyze_results)

def analyze_image(img_array, use_tta, n_augmentations):
    
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if 'custom_cnn_final.h5' exists.")
        return None
    
    with st.spinner("Analyzing image..."):
        if use_tta:
            predictions = predict_with_tta(model, img_array, n_augmentations)
        else:
            predictions = model.predict(img_array, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        class_info = CLASS_INFO[predicted_class]
        
        return {
            'predictions': predictions,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_info': class_info
        }

def display_results(results):
    if results is None:
        return
    
    predictions = results['predictions']
    predicted_class = results['predicted_class']
    confidence = results['confidence']
    class_info = results['class_info']
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color: {class_info['color']}; margin-top: 0;">
            Prediction: {class_info['name']}
        </h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
            <strong>Confidence:</strong> {confidence:.2f}%
        </p>
        <p style="color: #666; margin: 0.5rem 0;">
            {class_info['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if confidence < 70:
        st.markdown("""
        <div class="warning-box">
        <strong>Low Confidence Warning:</strong><br>
        Model confidence is below 70%. Please consult with a medical professional for accurate diagnosis.
        </div>
        """, unsafe_allow_html=True)
    
    st.plotly_chart(
        create_confidence_chart(predictions, CLASS_NAMES),
        use_container_width=True
    )
    
    with st.expander("Detailed Probabilities"):
        for i, class_name in enumerate(CLASS_NAMES):
            prob = predictions[0][i] * 100
            st.write(f"**{CLASS_INFO[class_name]['name']}:** {prob:.4f}%")

def show_about_model_page():
    st.header("About the Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">Architecture</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0;">Custom CNN</p>
            <p style="margin: 0; opacity: 0.9;">4-Block Deep Network</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">Input Size</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0;">224×224</p>
            <p style="margin: 0; opacity: 0.9;">RGB Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">Classes</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0;">4 Types</p>
            <p style="margin: 0; opacity: 0.9;">Brain Tumors</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Model Architecture")
    
    with st.expander("Network Details", expanded=True):
        st.markdown("""
        **Convolutional Blocks:**
        - Block 1: 32 filters (3×3 conv, BatchNorm, MaxPool, Dropout 0.3)
        - Block 2: 64 filters (3×3 conv, BatchNorm, MaxPool, Dropout 0.3)
        - Block 3: 128 filters (3×3 conv, BatchNorm, MaxPool, Dropout 0.35)
        - Block 4: 256 filters (3×3 conv, BatchNorm, MaxPool, Dropout 0.35)
        
        **Classification Head:**
        - Dense 512 units (ReLU, BatchNorm, Dropout 0.55)
        - Dense 256 units (ReLU, BatchNorm, Dropout 0.55)
        - Output 4 units (Softmax)
        """)
    
    with st.expander("Training Configuration"):
        st.markdown("""
        **Optimizer:** Adam (lr=0.001)
        
        **Loss Function:** Categorical Crossentropy
        
        **Data Augmentation:**
        - Rotation: ±20°
        - Width/Height Shift: ±15%
        - Shear: 15%
        - Zoom: ±15%
        - Horizontal/Vertical Flip
        
        **Callbacks:**
        - ModelCheckpoint (save best model)
        - ReduceLROnPlateau (factor=0.5, patience=7)
        
        **Training:** 100 epochs with early stopping
        """)
    
    st.subheader("Performance Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Validation': [95.2, 94.8, 95.1, 94.9],
        'Test (Standard)': [93.8, 93.5, 93.7, 93.6],
        'Test (TTA)': [95.5, 95.2, 95.3, 95.2]
    }
    
    st.table(metrics_data)

def show_information_page():
    st.header("Information")
    
    st.subheader("Tumor Types")
    
    for class_name in CLASS_NAMES:
        info = CLASS_INFO[class_name]
        with st.expander(f"{info['name']}", expanded=False):
            st.markdown(f"""
            <div style="border-left: 4px solid {info['color']}; padding-left: 1rem;">
                <h4 style="color: {info['color']}; margin-top: 0;">{info['name']}</h4>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("How to Use")
    
    st.markdown("""
    1. **Upload Image:** Navigate to Classification page and upload brain MRI image
    2. **Configure Settings:** Enable/disable TTA in sidebar for improved accuracy
    3. **Analyze:** Click 'Analyze Image' button to get prediction
    4. **Review Results:** Check prediction, confidence score, and detailed probabilities
    
    **Note:** This system is designed for research and educational purposes. 
    Always consult with qualified medical professionals for actual diagnosis.
    """)
    
    st.markdown("---")
    
    st.subheader("Disclaimer")
    
    st.warning("""
    **Medical Disclaimer:**
    
    This application is an AI-based research tool and should NOT be used as the sole basis 
    for medical diagnosis or treatment decisions. Always consult with qualified healthcare 
    professionals for proper medical evaluation and diagnosis.
    
    The predictions made by this system are computational analyses and may contain errors. 
    They should be considered as supplementary information only.
    """)

if __name__ == "__main__":
    main()