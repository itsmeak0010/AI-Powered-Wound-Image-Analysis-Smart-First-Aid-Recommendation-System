import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import os

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Wound Care AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Wound Classification & First Aid System"
    }
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 15px;
        border-left: 4px solid #ff0000;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-low {
        background-color: #d4edda;
        padding: 15px;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODELS =====
@st.cache_resource
def load_tensorflow_model(model_name):
    """Load TensorFlow/Keras models"""
    try:
        model = tf.keras.models.load_model(f"{model_name}")
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

@st.cache_resource
def load_vit_model():
    """Load Vision Transformer model from transformers"""
    try:
        from transformers import ViTForImageClassification
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with correct number of classes
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        
        # Load the trained weights
        model_path = "vit_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ViT model: {e}")
        return None

# ===== INITIALIZE SESSION STATE =====
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

# ===== HEADER =====
st.markdown("<h1 class='main-header'>🏥 AI Wound Care Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_choice = st.radio(
        "Select AI Model:",
        options=["VGG19", "MobileNet", "EfficientNet", "Vision Transformer"],
        help="Different models may provide different accuracy levels"
    )
    
    st.markdown("---")
    st.subheader("📊 Model Info")
    model_info = {
        "VGG19": "Deep CNN, high accuracy, slower",
        "MobileNet": "Lightweight, fast inference",
        "EfficientNet": "Balanced efficiency & accuracy",
        "Vision Transformer": "State-of-the-art, transformer-based"
    }
    st.info(model_info[model_choice])
    
    st.markdown("---")
    st.subheader("📋 Wound Classifications")
    classes_info = {
        "🩹 Cut": "Clean laceration or wound",
        "🔥 Burn": "Thermal injury",
        "💙 Bruise": "Blunt trauma, contusion",
        "🦠 Infection": "Signs of bacterial/fungal infection"
    }
    for class_name, description in classes_info.items():
        st.write(f"**{class_name}**: {description}")

# ===== MAIN CONTENT =====
tab1, tab2, tab3 = st.tabs(["🔍 Diagnosis", "📚 Medical Guide", "ℹ️ About"])

with tab1:
    # ===== IMAGE UPLOAD =====
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Upload Wound Image")
        uploaded_file = st.file_uploader(
            "Choose a wound image",
            type=["jpg", "png", "jpeg", "bmp"],
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        # ===== LOAD MODEL BASED ON SELECTION =====
        model = None
        
        if model_choice == "VGG19":
            if "vgg" not in st.session_state.model_cache:
                with st.spinner("Loading VGG19 model..."):
                    st.session_state.model_cache["vgg"] = load_tensorflow_model("vgg_model.h5")
            model = st.session_state.model_cache["vgg"]
        
        elif model_choice == "MobileNet":
            if "mobilenet" not in st.session_state.model_cache:
                with st.spinner("Loading MobileNet model..."):
                    st.session_state.model_cache["mobilenet"] = load_tensorflow_model("mobilenet_model.h5")
            model = st.session_state.model_cache["mobilenet"]
        
        elif model_choice == "EfficientNet":
            if "efficientnet" not in st.session_state.model_cache:
                with st.spinner("Loading EfficientNet model..."):
                    st.session_state.model_cache["efficientnet"] = load_tensorflow_model("efficientnet_model.h5")
            model = st.session_state.model_cache["efficientnet"]
        
        elif model_choice == "Vision Transformer":
            if "vit" not in st.session_state.model_cache:
                with st.spinner("Loading Vision Transformer model..."):
                    st.session_state.model_cache["vit"] = load_vit_model()
            model = st.session_state.model_cache["vit"]
        
        if model is not None:
            # ===== IMAGE PROCESSING =====
            img = Image.open(uploaded_file).convert("RGB")
            img_resized = img.resize((224, 224))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img, caption="Original Image", use_container_width=True)
            
            with col2:
                st.image(img_resized, caption="Model Input (224x224)", use_container_width=True)
            
            # ===== PREDICTION =====
            with st.spinner(f"Analyzing image with {model_choice}..."):
                classes = ['bruise', 'burn', 'cut', 'infection']
                
                if model_choice == "Vision Transformer":
                    # PyTorch ViT prediction
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # ViT requires specific normalization
                    vit_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
                    
                    img_tensor = vit_transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        logits = outputs.logits
                        prediction = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    idx = np.argmax(prediction[0])
                    confidence = float(np.max(prediction[0]))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                else:
                    # TensorFlow models prediction
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    prediction = model.predict(img_array, verbose=0)
                    idx = np.argmax(prediction)
                    confidence = float(np.max(prediction))
                
                label = classes[idx]
                
                # ===== SEVERITY ANALYSIS =====
                red_pixels = np.mean(img_array[:, :, 0])
                
                severity = "Mild"
                recommendation = ""
                risk = "Low"
                infection_prob = "Low"
                alert_level = "low"
                
                if label == "cut":
                    if red_pixels > 0.55:
                        severity = "Severe"
                        recommendation = "🚨 Go to hospital immediately. Stitches may be required."
                        risk = "High"
                        alert_level = "high"
                    else:
                        severity = "Minor"
                        recommendation = "Clean and cover with sterile bandage at home."
                        alert_level = "low"
                
                elif label == "burn":
                    if red_pixels > 0.6:
                        severity = "Severe"
                        recommendation = "🚨 Hospital treatment required immediately."
                        risk = "High"
                        infection_prob = "High"
                        alert_level = "high"
                    else:
                        severity = "Moderate"
                        recommendation = "Cool under running water for 10-20 minutes, then apply burn ointment."
                        risk = "Medium"
                        alert_level = "medium"
                
                elif label == "infection":
                    severity = "High Risk"
                    recommendation = "⚠️ Consult doctor immediately. Antibiotics may be needed."
                    risk = "High"
                    infection_prob = "High"
                    alert_level = "high"
                
                elif label == "bruise":
                    severity = "Low Risk"
                    recommendation = "Apply ice pack for 15-20 minutes, then rest the area."
                    alert_level = "low"
                
                st.markdown("---")
                st.subheader("📋 AI Medical Assessment Report")
                
                # ===== METRICS DISPLAY =====
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Detected Injury", label.upper(), delta=None)
                
                with metric_col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with metric_col3:
                    st.metric("Severity", severity)
                
                with metric_col4:
                    st.metric("Risk Level", risk)
                
                # ===== PREDICTION CONFIDENCE CHART =====
                st.subheader("📊 Prediction Confidence")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#ff6b6b' if c == label else '#e0e0e0' for c in classes]
                bars = ax.barh(classes, prediction[0], color=colors)
                ax.set_xlabel("Confidence Score")
                ax.set_xlim(0, 1)
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{prediction[0][i]:.2%}',
                            ha='left', va='center', fontsize=10, fontweight='bold')
                
                ax.set_title(f"Model Predictions ({model_choice})", fontweight='bold')
                st.pyplot(fig, use_container_width=True)
                
                # ===== ALERT BOX =====
                st.markdown("---")
                if alert_level == "high":
                    st.markdown(f"""
                    <div class='alert-high'>
                        <strong>🚨 HIGH ALERT</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                elif alert_level == "medium":
                    st.markdown(f"""
                    <div class='alert-medium'>
                        <strong>⚠️ MODERATE ALERT</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='alert-low'>
                        <strong>✅ LOW ALERT</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                
                # ===== DETAILED ASSESSMENT =====
                st.markdown("---")
                st.subheader("📋 Detailed Assessment")
                
                assessment_col1, assessment_col2 = st.columns(2)
                
                with assessment_col1:
                    st.markdown("""
                    **Infection Risk Assessment**
                    - Infection Probability: {}
                    - Signs to Watch: Redness, swelling, warmth
                    - Healing Timeline: Varies by severity
                    """.format(infection_prob))
                
                with assessment_col2:
                    st.markdown("""
                    **When to Seek Medical Attention**
                    - Continuous bleeding (> 10 mins)
                    - Increasing swelling or redness
                    - Pus or discharge formation
                    - Fever or severe pain
                    - Wound not improving in 7 days
                    """)
                
                # ===== CARE INSTRUCTIONS =====
                st.markdown("---")
                st.subheader("🏥 First Aid Instructions")
                
                care_instructions = {
                    'cut': """
                    1. **Stop the bleeding**: Apply pressure with clean cloth for 5-10 minutes
                    2. **Clean**: Use antiseptic or clean water to rinse
                    3. **Protect**: Cover with sterile bandage
                    4. **Monitor**: Check daily for signs of infection
                    """,
                    'burn': """
                    1. **Cool immediately**: Run cool (not cold) water for 10-20 minutes
                    2. **Remove**: Take off constrictive items
                    3. **Cover**: Apply dry, non-stick bandage
                    4. **Relief**: Take over-the-counter pain relievers if needed
                    5. **Avoid**: Do not apply ice directly or use butter/oil
                    """,
                    'bruise': """
                    1. **Rest**: Avoid using the affected area
                    2. **Ice**: Apply ice pack for 15-20 minutes, 3-4 times daily
                    3. **Compress**: Use elastic bandage if swelling
                    4. **Elevate**: Keep affected area raised above heart level
                    5. **Medical**: Seek help if severe pain or inability to move
                    """,
                    'infection': """
                    1. **Medical attention**: Consult doctor immediately
                    2. **Clean**: Gently clean with antiseptic
                    3. **Cover**: Keep wound covered and clean
                    4. **Medications**: Follow doctor's prescriptions (antibiotics, etc.)
                    5. **Monitor**: Track temperature and symptoms
                    """
                }
                
                st.markdown(care_instructions.get(label, "Consult medical professional"))

with tab2:
    st.subheader("📚 Medical Information Guide")
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.markdown("""
        ### 🩹 Cuts & Lacerations
        **Causes**: Sharp objects, falls, accidents
        **Severity Levels**:
        - Minor: Shallow, < 1cm, minimal bleeding
        - Moderate: Deeper, > 1cm, controlled bleeding
        - Severe: Deep, uncontrolled bleeding
        
        **Treatment**:
        - Clean with soap and water
        - Apply pressure to stop bleeding
        - Use antibiotic ointment
        - Cover with sterile bandage
        
        **When to see doctor**: Deep cuts, uncontrolled bleeding, dirty wound
        """)
        
        st.markdown("""
        ### 🔥 Burns
        **Causes**: Heat, hot liquids, flames
        **Severity Levels**:
        - 1st Degree: Red, painful, no blistering
        - 2nd Degree: Blistering, swelling
        - 3rd Degree: Charred, severe (EMERGENCY)
        
        **First Aid**:
        - Cool with water (10-20 mins)
        - Remove tight items
        - Cover with clean cloth
        - Elevate if possible
        
        **When to see doctor**: Large area, 2nd/3rd degree, face/hands
        """)
    
    with guide_col2:
        st.markdown("""
        ### 💙 Bruises & Contusions
        **Causes**: Blunt force trauma
        **Severity Levels**:
        - Mild: Light discoloration
        - Moderate: Visible swelling, darker color
        - Severe: Severe swelling, severe pain
        
        **Treatment**:
        - Rest the area
        - Ice for 15-20 minutes
        - Compress with bandage
        - Elevate above heart level
        
        **When to see doctor**: Severe swelling, inability to move, severe pain
        """)
        
        st.markdown("""
        ### 🦠 Infections
        **Signs of Infection**:
        - Increasing redness or swelling
        - Pus or discharge
        - Increased warmth
        - Fever
        - Red streaks (serious!)
        
        **Prevention**:
        - Keep wound clean and dry
        - Change bandages daily
        - Use prescribed antibiotics
        - Avoid touching wound
        
        **Treatment**: SEE DOCTOR IMMEDIATELY
        """)

with tab3:
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This AI-powered wound classification system helps users:
    - Quickly identify wound types
    - Assess severity levels
    - Receive appropriate first aid recommendations
    - Know when to seek medical attention
    
    ### 🧠 Technology
    - **AI Models**: VGG19, MobileNet, EfficientNet, Vision Transformer
    - **Framework**: TensorFlow/PyTorch
    - **Interface**: Streamlit
    - **Image Input**: JPEG, PNG (224x224 pixels)
    
    ### ⚠️ Disclaimer
    This application provides educational information only and is NOT a substitute for professional medical advice. Always consult qualified healthcare professionals for serious wounds or injuries.
    
    ### 🔒 Privacy
    - Images are processed locally
    - No data is stored or transmitted
    - Each analysis is independent
    
    ### 📧 Support
    For issues or feedback, please contact the development team.
    """)