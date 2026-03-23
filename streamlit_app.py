# ============================================================
# STREAMLIT WEB APP - Skin Cancer Detection System
# Theme: Dark Red Medical + Animations + Team Names
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import numpy as np
import cv2
import datetime
import tensorflow as tf
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# CSS - DARK RED MEDICAL THEME + ANIMATIONS + FONTS
# ============================================================
st.markdown("""
<style>
    /* Import Medical Font */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Raleway:wght@400;600;700;800&display=swap');

    /* Dark Red Medical Background */
    .stApp {
        background-color: #1a0a0a;
        background-image:
            radial-gradient(circle at 10% 10%, rgba(192,57,43,0.18) 0%, transparent 45%),
            radial-gradient(circle at 90% 10%, rgba(192,57,43,0.12) 0%, transparent 45%),
            radial-gradient(circle at 10% 90%, rgba(192,57,43,0.12) 0%, transparent 45%),
            radial-gradient(circle at 90% 90%, rgba(192,57,43,0.18) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(120,30,20,0.15) 0%, transparent 60%),
            linear-gradient(135deg, #1a0a0a 0%, #2c1010 50%, #1a0a0a 100%);
        background-attachment: fixed;
        font-family: 'Nunito', sans-serif !important;
        color: #ffe0e0;
    }

    /* Font for all text */
    .stMarkdown, p, h1, h2, h3, h4, label, span {
        font-family: 'Nunito', sans-serif !important;
        color: #ffe0e0 !important;
    }

    /* Fade-in Animation */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(192,57,43,0.4); }
        70%  { box-shadow: 0 0 0 10px rgba(192,57,43,0); }
        100% { box-shadow: 0 0 0 0 rgba(192,57,43,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    .fade-in {
        animation: fadeInUp 0.8s ease forwards;
    }
    .fade-in-slow {
        animation: fadeIn 1.2s ease forwards;
    }

    /* Buttons - Red Gradient */
    .stButton button {
        background: linear-gradient(90deg, #922b21, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 12px !important;
        font-family: 'Nunito', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        animation: pulse 2s infinite !important;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #c0392b, #922b21) !important;
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(192,57,43,0.5) !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #6b1a1a, #922b21) !important;
        color: white !important;
        border: 1px solid #c0392b !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30,10,10,0.9) !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #e74c3c !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        font-family: 'Nunito', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #922b21 !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* File uploader */
    .stFileUploader {
        background-color: rgba(44,16,16,0.8) !important;
        border: 2px dashed #c0392b !important;
        border-radius: 12px !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background-color: rgba(44,16,16,0.9) !important;
        color: #ffe0e0 !important;
        border: 1px solid #c0392b !important;
        border-radius: 8px !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* Metrics */
    .stMetric {
        background-color: rgba(44,16,16,0.8) !important;
        border: 1px solid #c0392b !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Cards */
    .report-card {
        background: linear-gradient(135deg, rgba(20,5,5,0.95), rgba(44,16,16,0.95));
        border: 1px solid #c0392b;
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(192,57,43,0.2);
        animation: fadeInUp 0.6s ease forwards;
    }
    .section-header {
        background: linear-gradient(90deg, #922b21, #6b1a1a);
        padding: 10px 20px;
        border-radius: 8px;
        margin: 15px 0 10px 0;
        border-left: 4px solid #e74c3c;
    }
    .tip-card {
        background: linear-gradient(135deg, rgba(20,5,5,0.95), rgba(44,16,16,0.95));
        border: 1px solid #c0392b;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        text-align: center;
        animation: fadeInUp 0.6s ease forwards;
    }
    .stat-box {
        background: linear-gradient(135deg, rgba(92,28,28,0.9), rgba(146,43,33,0.9));
        border: 1px solid #c0392b;
        border-top: 4px solid #e74c3c;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(192,57,43,0.3);
        animation: fadeInUp 0.5s ease forwards;
    }

    /* Risk banners */
    .risk-low {
        background: linear-gradient(135deg, #1a7a3c, #27ae60);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #2ecc71;
        box-shadow: 0 0 25px rgba(46,204,113,0.5);
        animation: fadeIn 0.5s ease forwards;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #7a5c00, #f39c12);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #f39c12;
        box-shadow: 0 0 25px rgba(243,156,18,0.5);
        animation: fadeIn 0.5s ease forwards;
    }
    .risk-high {
        background: linear-gradient(135deg, #7a0d0d, #c0392b);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #e74c3c;
        box-shadow: 0 0 25px rgba(231,76,60,0.5);
        animation: fadeIn 0.5s ease forwards;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: rgba(44,16,16,0.8) !important;
        border: 1px solid #c0392b !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
CLASS_NAMES = {0:'akiec',1:'bcc',2:'bkl',3:'df',4:'mel',5:'nv',6:'vasc'}

DISEASE_INFO = {
    'akiec': {
        'full_name':'Actinic Keratosis','risk_level':'MODERATE',
        'risk_color':'🟡','risk_class':'risk-moderate','type':'Pre-cancerous','icon':'⚠️',
        'description':"Actinic Keratosis is a rough, scaly patch caused by years of sun exposure. It is a pre-cancerous condition that can develop into skin cancer if left untreated.",
        'symptoms':["🔸 Rough, dry, scaly patch","🔸 Flat to slightly raised bump","🔸 Color: pink, red, or brown","🔸 Itching, burning, or tenderness"],
        'treatment':["💊 Cryotherapy","💊 Topical medications","💊 Photodynamic therapy","💊 Laser therapy"],
        'prevention':["☀️ Apply sunscreen SPF 30+ daily","👒 Wear protective hats","👕 UV-protective clothing","🕶️ UV-blocking sunglasses","🏠 Avoid peak sun hours"],
        'recommendation':"Consult a dermatologist soon. Early treatment prevents progression to cancer."
    },
    'bcc': {
        'full_name':'Basal Cell Carcinoma','risk_level':'HIGH',
        'risk_color':'🔴','risk_class':'risk-high','type':'Malignant (Cancer)','icon':'🚨',
        'description':"Basal Cell Carcinoma is the most common type of skin cancer. It begins in the basal cells and often appears as a transparent bump on sun-exposed skin.",
        'symptoms':["🔸 Pearly or waxy bump","🔸 Bleeding sore that returns","🔸 Pink growth with raised edges","🔸 Flat scar-like lesion"],
        'treatment':["💊 Surgical excision","💊 Mohs surgery","💊 Radiation therapy","💊 Targeted drug therapy"],
        'prevention':["☀️ Broad-spectrum sunscreen daily","🚫 Avoid tanning beds","👒 Wide-brimmed hats","🔍 Monthly self-skin checks","🏥 Annual dermatologist visit"],
        'recommendation':"URGENT: Seek immediate medical attention from a certified dermatologist."
    },
    'bkl': {
        'full_name':'Benign Keratosis','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Benign Keratosis includes seborrheic keratoses and solar lentigines. These are non-cancerous growths very common in older adults and generally harmless.",
        'symptoms':["🔸 Waxy, scaly raised growth","🔸 Color: light tan to black","🔸 Round or oval shape","🔸 Occasional itching"],
        'treatment':["💊 Usually no treatment needed","💊 Cryotherapy","💊 Curettage","💊 Laser treatment"],
        'prevention':["☀️ Sun protection from early age","💧 Keep skin moisturized","🥗 Antioxidant-rich foods","💧 Stay hydrated","🔍 Monitor moles regularly"],
        'recommendation':"Generally benign. Monitor for changes in size, shape, or color."
    },
    'df': {
        'full_name':'Dermatofibroma','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Dermatofibroma is a common benign skin growth appearing on the legs. It is a harmless fibrous nodule in the deep skin layers.",
        'symptoms':["🔸 Small hard bump under skin","🔸 Brown, red, or purple color","🔸 Skin dimples when pinched","🔸 Mild tenderness"],
        'treatment':["💊 Usually no treatment needed","💊 Surgical removal","💊 Cryotherapy","💊 Steroid injections"],
        'prevention':["🛡️ Protect skin from injuries","🦟 Use insect repellent","🧴 Keep skin moisturized","🔍 Monitor new skin bumps","💪 Maintain healthy immunity"],
        'recommendation':"Benign and rarely needs treatment. Monitor for any changes."
    },
    'mel': {
        'full_name':'Melanoma','risk_level':'VERY HIGH',
        'risk_color':'🔴🔴','risk_class':'risk-high','type':'Malignant (Most Dangerous)','icon':'🚨🚨',
        'description':"Melanoma is the most dangerous skin cancer. Early detection is CRITICAL — survival rate is 98% when caught early vs only 23% in advanced stages.",
        'symptoms':["🔸 Mole changing size/shape/color","🔸 Asymmetrical shape","🔸 Irregular borders","🔸 Multiple colors","🔸 Diameter > 6mm","🔸 Bleeding or itching"],
        'treatment':["💊 Surgical excision","💊 Immunotherapy","💊 Targeted therapy","💊 Radiation therapy"],
        'prevention':["☀️ Never skip sunscreen SPF 50+","🚫 Never use tanning beds","👒 Always cover skin in sun","🔍 Check moles monthly","🏥 Annual skin cancer screening","👨‍👩‍👧 Know your family history"],
        'recommendation':"⚠️ URGENT: See an oncologist IMMEDIATELY. Do NOT delay!"
    },
    'nv': {
        'full_name':'Melanocytic Nevi (Mole)','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Melanocytic Nevi are common moles that are almost always harmless. Monitor using the ABCDE rule for suspicious changes.",
        'symptoms':["🔸 Small dark brown spot","🔸 Symmetric round shape","🔸 Smooth even border","🔸 Uniform color","🔸 Less than 6mm"],
        'treatment':["💊 Usually no treatment needed","💊 Surgical removal if suspicious","💊 Regular monitoring","💊 Dermoscopy"],
        'prevention':["☀️ Limit sun exposure as child","🧴 Use sunscreen from young age","🔍 Monitor with ABCDE rule","📸 Photograph to track changes","🏥 Regular dermatologist visits"],
        'recommendation':"Benign mole. Monitor using ABCDE: Asymmetry, Border, Color, Diameter, Evolution."
    },
    'vasc': {
        'full_name':'Vascular Lesion','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Vascular lesions are blood vessel abnormalities in the skin. Most are benign and caused by abnormal blood vessel growth near the skin surface.",
        'symptoms':["🔸 Red/purple/blue spot","🔸 Flat or slightly raised","🔸 May bleed easily","🔸 Spider-vein appearance"],
        'treatment':["💊 Usually no treatment needed","💊 Laser therapy","💊 Electrosurgery","💊 Cryotherapy"],
        'prevention':["🧴 Protect skin from trauma","☀️ Use sun protection","💧 Stay hydrated","🥗 Vitamin C rich foods","🏃 Healthy blood circulation"],
        'recommendation':"Typically benign. See doctor if it bleeds or grows rapidly."
    }
}

SKIN_TIPS = [
    {"icon":"☀️","title":"Sun Protection","tip":"Apply SPF 30+ sunscreen every day, even on cloudy days!"},
    {"icon":"💧","title":"Stay Hydrated","tip":"Drink 8+ glasses of water daily for healthy glowing skin."},
    {"icon":"🥗","title":"Eat Healthy","tip":"Antioxidant-rich foods protect skin from free radical damage."},
    {"icon":"😴","title":"Sleep Well","tip":"Get 7-8 hours of sleep. Skin repairs itself during deep sleep."},
    {"icon":"🚭","title":"Avoid Smoking","tip":"Smoking reduces blood flow causing premature aging."},
    {"icon":"🔍","title":"Self Check Monthly","tip":"Use ABCDE rule to monitor moles for suspicious changes."},
    {"icon":"🏥","title":"Annual Checkup","tip":"Visit a dermatologist annually for skin cancer screening."},
    {"icon":"👒","title":"Wear Protection","tip":"Wide-brimmed hats and UV-protective clothing outdoors."},
    {"icon":"🧴","title":"Moisturize Daily","tip":"Apply moisturizer after bathing for healthy skin barrier."},
    {"icon":"🚫","title":"No Tanning Beds","tip":"Tanning beds increase melanoma risk by 75%!"},
    {"icon":"🍊","title":"Vitamin C","tip":"Boosts collagen and protects against UV-induced damage."},
    {"icon":"🏃","title":"Exercise Daily","tip":"Improves circulation delivering nutrients to skin cells."},
]

# ============================================================
# MODELS
# ============================================================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model('cnn_model.h5')
    tl  = tf.keras.models.load_model('transfer_learning_model.h5')
    return cnn, tl

def predict_image(image, cnn_model, tl_model):
    img_gray  = np.array(image.convert('L').resize((28,28)))
    img_norm  = img_gray / 255.0
    cnn_input = img_norm.reshape(1,28,28,1)
    cnn_probs = cnn_model.predict(cnn_input, verbose=0)[0]
    cnn_pred  = np.argmax(cnn_probs)
    cnn_conf  = cnn_probs[cnn_pred] * 100
    return CLASS_NAMES[cnn_pred], cnn_conf

# ============================================================
# PROJECT TITLE BANNER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #5c1c1c 0%, #922b21 50%, #5c1c1c 100%);
    padding: 8px 20px; border-radius: 0px; text-align: center;
    border-bottom: 2px solid #e74c3c; margin-bottom: 0px;
    font-family: 'Nunito', sans-serif;
    animation: fadeIn 1s ease forwards;
">
    <p style="color:#ffcccc; font-size:0.85em; margin:0; letter-spacing:1px;">
        🎓 Department of Computer Science &amp; Engineering &nbsp;|&nbsp;
        Project Expo 2026 &nbsp;|&nbsp;
        Team Members:
        <b style="color:white;">Tejaswini</b> •
        <b style="color:white;">Sai Sreenidhi</b> •
        <b style="color:white;">Anusha</b> •
        <b style="color:white;">Sreekala</b> •
        <b style="color:white;">Poojitha</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(92,28,28,0.95) 0%, rgba(146,43,33,0.95) 100%);
    padding: 35px 40px; border-radius: 0 0 20px 20px; text-align: center;
    border: 1px solid #c0392b; border-top: none; margin-bottom: 25px;
    box-shadow: 0 0 50px rgba(192,57,43,0.4);
    font-family: 'Raleway', sans-serif;
    animation: fadeInUp 0.8s ease forwards;
">
    <div style="font-size:3em; margin-bottom:10px;">🔬</div>
    <h1 style="color:#ffffff; font-size:2.5em; margin:0; font-weight:800;
               font-family:'Raleway',sans-serif; letter-spacing:2px;
               text-shadow: 0 0 20px rgba(231,76,60,0.6);">
        SKIN CANCER DETECTION SYSTEM
    </h1>
    <div style="width:80px; height:3px; background:#e74c3c;
                margin:15px auto; border-radius:2px;"></div>
    <p style="color:#ffcccc; font-size:1.1em; margin:5px 0;
              font-family:'Nunito',sans-serif;">
        AI-Powered Early Detection • Deep Learning • NLP Medical Reports
    </p>
    <p style="color:#ff9999; font-size:0.9em; margin:5px 0;">
        🏥 HAM10000 Dataset &nbsp;|&nbsp; 🧠 CNN Model &nbsp;|&nbsp;
        🤖 MobileNetV2 &nbsp;|&nbsp; 🔬 7 Class Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Welcome",
    "🔬 Analyze & Diagnose",
    "💚 Skin Wellness",
    "📊 AI Performance"
])

# ============================================================
# TAB 1 - WELCOME (HOME PAGE)
# ============================================================
with tab1:

    # Hero Banner
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(20,5,5,0.95), rgba(44,16,16,0.95));
        border: 2px solid #c0392b; border-radius: 20px;
        padding: 50px 40px; text-align: center; margin-bottom: 28px;
        box-shadow: 0 0 40px rgba(192,57,43,0.3);
        animation: fadeInUp 0.8s ease forwards;
        font-family: 'Raleway', sans-serif;
    ">
        <div style="font-size:4em; margin-bottom:10px;">🏥</div>
        <h1 style="color:#ffffff; font-size:2.8em; margin:0; font-weight:800;
                   text-shadow: 0 0 20px rgba(231,76,60,0.6);
                   font-family:'Raleway',sans-serif; letter-spacing:1px;">
            Welcome to AI Skin Cancer Detection
        </h1>
        <div style="width:80px; height:3px; background:#e74c3c;
                    margin:15px auto; border-radius:2px;"></div>
        <p style="color:#ffcccc; font-size:1.2em; margin:10px 0; line-height:1.6;
                  font-family:'Nunito',sans-serif;">
            AI-Powered Early Detection using Deep Learning &amp; NLP
        </p>
        <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-top:15px;">
            <span style="background:rgba(192,57,43,0.2); color:#e74c3c;
                border:1px solid #c0392b; padding:6px 16px; border-radius:20px; font-size:0.9em;">
                🔬 HAM10000 Dataset</span>
            <span style="background:rgba(192,57,43,0.2); color:#e74c3c;
                border:1px solid #c0392b; padding:6px 16px; border-radius:20px; font-size:0.9em;">
                🧠 CNN Model</span>
            <span style="background:rgba(192,57,43,0.2); color:#e74c3c;
                border:1px solid #c0392b; padding:6px 16px; border-radius:20px; font-size:0.9em;">
                🤖 MobileNetV2</span>
            <span style="background:rgba(192,57,43,0.2); color:#e74c3c;
                border:1px solid #c0392b; padding:6px 16px; border-radius:20px; font-size:0.9em;">
                📋 NLP Reports</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Row
    s1,s2,s3,s4 = st.columns(4)
    for col,(icon,val,label) in zip([s1,s2,s3,s4],[
        ("🖼️","10,015","Training Images"),
        ("🏷️","7","Disease Classes"),
        ("🎯","92.82%","CNN Accuracy"),
        ("⚡","< 5s","Report Time"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div style="font-size:2em; margin-bottom:6px;">{icon}</div>
                <h2 style="color:#e74c3c; margin:5px 0; font-size:1.9em; font-weight:800;
                           font-family:'Raleway',sans-serif;">{val}</h2>
                <p style="color:#ffcccc; margin:0; font-size:0.85em;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1,1])

    with left:
        # About
        st.markdown("""
        <div style="background:rgba(20,5,5,0.95); border:1px solid #c0392b;
            border-left:5px solid #e74c3c; border-radius:10px; padding:25px;
            margin-bottom:16px; animation:fadeInUp 0.6s ease forwards;">
            <h3 style="color:#e74c3c; margin-top:0; font-family:'Raleway',sans-serif;">
                🔬 About This System</h3>
            <p style="color:#ffe0e0; line-height:1.9; font-family:'Nunito',sans-serif;">
                This AI-powered system uses <b>Deep Learning</b> and
                <b>Natural Language Processing</b> to detect
                <b>7 types of skin lesions</b> and automatically generates
                complete <b>medical reports</b> with diagnosis, symptoms,
                treatment options, and personalized recommendations.
            </p>
            <p style="color:#ffcccc; line-height:1.9; font-family:'Nunito',sans-serif;">
                Built on the internationally recognized
                <b>HAM10000 dataset</b> from Vienna General Hospital.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Conditions
        st.markdown("""
        <div style="background:rgba(20,5,5,0.95); border:1px solid #c0392b;
            border-left:5px solid #27ae60; border-radius:10px; padding:25px;
            animation:fadeInUp 0.7s ease forwards;">
            <h3 style="color:#e74c3c; margin-top:0; font-family:'Raleway',sans-serif;">
                🏥 Detectable Conditions</h3>
        """, unsafe_allow_html=True)
        for name,desc,risk,color,bg in [
            ("Melanocytic Nevi",    "Benign Mole",           "LOW",       "#2ecc71","rgba(46,204,113,0.15)"),
            ("Benign Keratosis",    "Non-cancerous Growth",  "LOW",       "#2ecc71","rgba(46,204,113,0.15)"),
            ("Dermatofibroma",      "Benign Nodule",         "LOW",       "#2ecc71","rgba(46,204,113,0.15)"),
            ("Vascular Lesion",     "Blood Vessel Growth",   "LOW",       "#2ecc71","rgba(46,204,113,0.15)"),
            ("Actinic Keratosis",   "Pre-cancerous Patch",   "MODERATE",  "#f39c12","rgba(243,156,18,0.15)"),
            ("Basal Cell Carcinoma","Skin Cancer",           "HIGH",      "#e74c3c","rgba(231,76,60,0.15)"),
            ("Melanoma",            "Most Dangerous Cancer", "VERY HIGH", "#e74c3c","rgba(231,76,60,0.2)"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                padding:9px 0; border-bottom:1px solid rgba(192,57,43,0.2);
                font-family:'Nunito',sans-serif;">
                <div>
                    <span style="color:#ffe0e0; font-weight:600; font-size:0.9em;">{name}</span>
                    <span style="color:#ff9999; font-size:0.8em; margin-left:6px;">— {desc}</span>
                </div>
                <span style="background:{bg}; color:{color}; padding:3px 10px;
                    border-radius:12px; font-size:0.75em; font-weight:700;
                    white-space:nowrap; margin-left:8px;">{risk}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # How it works
        st.markdown("""
        <div style="background:rgba(20,5,5,0.95); border:1px solid #c0392b;
            border-left:5px solid #e74c3c; border-radius:10px; padding:25px;
            margin-bottom:16px; animation:fadeInUp 0.6s ease forwards;">
            <h3 style="color:#e74c3c; margin-top:0; font-family:'Raleway',sans-serif;">
                📋 How It Works</h3>
        """, unsafe_allow_html=True)
        for num,title,desc in [
            ("1","Upload Image","Upload a clear skin lesion photo (JPG/PNG)"),
            ("2","Enter Patient Info","Add name, age, sex, lesion location"),
            ("3","AI Analysis","CNN model analyzes the image in seconds"),
            ("4","Medical Report","Full NLP report with diagnosis & tips"),
            ("5","Download","Save report for doctor consultation"),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:14px; padding:12px 0;
                border-bottom:1px solid rgba(192,57,43,0.2); font-family:'Nunito',sans-serif;">
                <div style="background:#c0392b; color:white; border-radius:50%;
                    width:28px; height:28px; display:flex; align-items:center;
                    justify-content:center; font-weight:700; font-size:0.85em;
                    min-width:28px; box-shadow:0 0 10px rgba(192,57,43,0.5);">{num}</div>
                <div>
                    <p style="color:#ffe0e0; font-weight:700; margin:0; font-size:0.9em;">{title}</p>
                    <p style="color:#ff9999; font-size:0.82em; margin:3px 0 0;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Survival stats
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(26,5,5,0.98),rgba(44,10,10,0.98));
            border:1px solid #e74c3c; border-left:5px solid #e74c3c;
            border-radius:10px; padding:25px; margin-bottom:16px;
            box-shadow:0 4px 20px rgba(231,76,60,0.3);
            animation:fadeInUp 0.7s ease forwards;">
            <h3 style="color:#e74c3c; margin-top:0; text-align:center;
                       font-family:'Raleway',sans-serif;">
                ⚠️ Why Early Detection Matters</h3>
            <div style="display:flex; justify-content:space-around;
                align-items:center; margin:16px 0;">
                <div style="text-align:center;">
                    <div style="color:#2ecc71; font-size:3em; font-weight:800;
                                font-family:'Raleway',sans-serif;">98%</div>
                    <div style="color:#a0ffa0; font-size:0.85em;">Survival Rate</div>
                    <div style="color:#7acc7a; font-size:0.75em;">Early Detection</div>
                </div>
                <div style="color:#e74c3c; font-size:1.8em; font-weight:800;">VS</div>
                <div style="text-align:center;">
                    <div style="color:#e74c3c; font-size:3em; font-weight:800;
                                font-family:'Raleway',sans-serif;">23%</div>
                    <div style="color:#ffaaaa; font-size:0.85em;">Survival Rate</div>
                    <div style="color:#ff7777; font-size:0.75em;">Late Detection</div>
                </div>
            </div>
            <p style="color:#ffaaaa; text-align:center; font-size:0.9em; margin:0;
                      font-family:'Nunito',sans-serif;">
                🔬 Melanoma is <b>highly curable</b> when detected early!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Skin Foods
        st.markdown("""
        <div style="background:rgba(20,5,5,0.95); border:1px solid #c0392b;
            border-radius:10px; padding:24px; animation:fadeInUp 0.8s ease forwards;">
            <h3 style="color:#e74c3c; margin:0 0 14px; font-family:'Raleway',sans-serif;">
                🌿 Foods for Healthy Skin</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        """, unsafe_allow_html=True)
        for emoji,name,benefit in [
            ("🫐","Blueberries","Antioxidants"),
            ("🥑","Avocado","Healthy fats"),
            ("🐟","Salmon","Omega-3"),
            ("🍊","Oranges","Vitamin C"),
            ("🥦","Broccoli","Vitamins C & K"),
            ("🍵","Green Tea","Catechins"),
        ]:
            st.markdown(f"""
            <div style="background:rgba(192,57,43,0.1); border:1px solid rgba(192,57,43,0.3);
                border-radius:8px; padding:8px 10px; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.3em;">{emoji}</span>
                <div>
                    <div style="color:#ffe0e0; font-size:0.82em; font-weight:700;
                                font-family:'Nunito',sans-serif;">{name}</div>
                    <div style="color:#ff9999; font-size:0.75em;">{benefit}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ABCDE Rule
    st.markdown("""
    <div style="background:rgba(20,5,5,0.95); border:1px solid #c0392b;
        border-radius:14px; padding:24px; margin-bottom:16px;
        animation:fadeInUp 0.9s ease forwards;">
        <h3 style="color:#e74c3c; margin:0 0 16px; text-align:center;
                   font-family:'Raleway',sans-serif;">
            🔍 ABCDE Rule — How to Monitor Your Moles</h3>
        <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:10px;">
    """, unsafe_allow_html=True)
    for letter,word,desc,color in [
        ("A","Asymmetry","One half doesn't match other","#e74c3c"),
        ("B","Border","Edges irregular or blurred","#c0392b"),
        ("C","Color","Multiple colors in one spot","#a93226"),
        ("D","Diameter","Larger than 6mm in size","#922b21"),
        ("E","Evolution","Any change over time","#7b241c"),
    ]:
        st.markdown(f"""
        <div style="background:rgba(192,57,43,0.1); border:1px solid rgba(192,57,43,0.3);
            border-top:3px solid {color}; border-radius:10px; padding:14px; text-align:center;
            font-family:'Nunito',sans-serif;">
            <div style="color:{color}; font-size:2em; font-weight:800; margin-bottom:4px;
                        font-family:'Raleway',sans-serif;">{letter}</div>
            <div style="color:#ffe0e0; font-size:0.85em; font-weight:700; margin-bottom:4px;">{word}</div>
            <div style="color:#ff9999; font-size:0.75em; line-height:1.4;">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Team Section
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(92,28,28,0.9),rgba(146,43,33,0.9));
        border:1px solid #c0392b; border-radius:14px; padding:24px; text-align:center;
        margin-bottom:16px; animation:fadeInUp 1s ease forwards;">
        <h3 style="color:#ffffff; margin:0 0 20px; font-family:'Raleway',sans-serif;
                   font-size:1.3em; letter-spacing:1px;">👩‍💻 Our Team</h3>
        <div style="display:flex; justify-content:center; gap:15px; flex-wrap:wrap;">
    """, unsafe_allow_html=True)
    for name in ["Tejaswini","Sai Sreenidhi","Anusha","Sreekala","Poojitha"]:
        st.markdown(f"""
        <div style="background:rgba(20,5,5,0.8); border:1px solid #e74c3c;
            border-radius:50px; padding:10px 20px;
            animation:fadeIn 1s ease forwards;">
            <span style="color:#ffffff; font-weight:700; font-size:0.95em;
                         font-family:'Nunito',sans-serif;">👩‍🎓 {name}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="background:rgba(10,2,2,0.9); border:1px solid rgba(192,57,43,0.3);
        border-radius:10px; padding:18px; text-align:center;">
        <p style="color:#ff9999; font-size:0.85em; margin:0; line-height:1.8;
                  font-family:'Nunito',sans-serif;">
            ⚕️ <b style="color:#ffcccc;">Medical Disclaimer:</b>
            This system is for educational and research purposes only.
            It is <b>NOT</b> a substitute for professional medical diagnosis.
            Always consult a qualified dermatologist for proper diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2 - ANALYZE & DIAGNOSE
# ============================================================
with tab2:
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="section-header"><h4 style="color:#ffe0e0;margin:0;font-family:Nunito,sans-serif;">📤 Upload Skin Lesion Image</h4></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose image (JPG, JPEG, PNG)", type=['jpg','jpeg','png'])
        st.markdown('<div class="section-header"><h4 style="color:#ffe0e0;margin:0;font-family:Nunito,sans-serif;">👤 Patient Information</h4></div>', unsafe_allow_html=True)
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        patient_age  = st.number_input("Age", min_value=1, max_value=120, value=30)
        patient_sex  = st.selectbox("Sex", ["Select","Male","Female"])
        location     = st.selectbox("Lesion Location", ["Select","Face","Back","Arm","Leg","Chest","Scalp","Neck","Hand","Other"])
        analyze_btn  = st.button("🔬 ANALYZE IMAGE", type="primary", use_container_width=True)

    with col2:
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="📸 Uploaded Skin Lesion", use_column_width=True)
        else:
            st.markdown("""
            <div style="background:rgba(20,5,5,0.95); border:2px dashed #c0392b;
                border-radius:15px; padding:80px; text-align:center;">
                <div style="font-size:4em;">📸</div>
                <h3 style="color:#e74c3c; font-family:'Raleway',sans-serif;">Image Preview</h3>
                <p style="color:#ff9999; font-family:'Nunito',sans-serif;">Upload an image to see preview</p>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file and analyze_btn:
        with st.spinner("🤖 AI is analyzing your image..."):
            try:
                cnn_model, tl_model = load_models()
                image = Image.open(uploaded_file)
                predicted_class, confidence = predict_image(image, cnn_model, tl_model)
                info     = DISEASE_INFO[predicted_class]
                date     = datetime.datetime.now().strftime("%B %d, %Y")
                time_now = datetime.datetime.now().strftime("%I:%M %p")

                st.markdown("---")

                st.markdown(f"""
                <div class="{info['risk_class']}" style="animation:fadeIn 0.5s ease forwards;">
                    <div style="font-size:3em;">{info['icon']}</div>
                    <h1 style="color:white; margin:5px 0; font-family:'Raleway',sans-serif;">{info['full_name']}</h1>
                    <h2 style="color:white; margin:5px 0;">Risk: {info['risk_color']} {info['risk_level']}</h2>
                    <h3 style="color:white; margin:5px 0; opacity:0.9;">{info['type']}</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-header"><h3 style="color:#ffe0e0;margin:0;font-family:Nunito,sans-serif;">📋 AI-Generated Medical Report</h3></div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="report-card">
                    <table style="width:100%; color:#ffe0e0; border-collapse:collapse; font-family:'Nunito',sans-serif;">
                        <tr style="border-bottom:1px solid rgba(192,57,43,0.3);">
                            <td style="padding:8px;">📅 <b>Date</b></td><td style="padding:8px;">{date}</td>
                            <td style="padding:8px;">⏰ <b>Time</b></td><td style="padding:8px;">{time_now}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(192,57,43,0.3);">
                            <td style="padding:8px;">👤 <b>Patient</b></td><td style="padding:8px;">{patient_name if patient_name else 'Not specified'}</td>
                            <td style="padding:8px;">🎂 <b>Age</b></td><td style="padding:8px;">{patient_age} years</td>
                        </tr>
                        <tr>
                            <td style="padding:8px;">⚧ <b>Sex</b></td><td style="padding:8px;">{patient_sex if patient_sex!='Select' else 'Not specified'}</td>
                            <td style="padding:8px;">📍 <b>Location</b></td><td style="padding:8px;">{location if location!='Select' else 'Not specified'}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#ffe0e0;margin:0;">📖 Medical Description</h4></div><p style="color:#ffe0e0;line-height:1.8;margin-top:15px;font-family:Nunito,sans-serif;">{info["description"]}</p></div>', unsafe_allow_html=True)

                symp_html = "".join([f"<li style='color:#ffe0e0;margin:8px 0;font-family:Nunito,sans-serif;'>{s}</li>" for s in info['symptoms']])
                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#ffe0e0;margin:0;">⚠️ Common Symptoms</h4></div><ul style="margin-top:15px;">{symp_html}</ul></div>', unsafe_allow_html=True)

                treat_html = "".join([f"<li style='color:#ffe0e0;margin:8px 0;font-family:Nunito,sans-serif;'>{t}</li>" for t in info['treatment']])
                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#ffe0e0;margin:0;">💊 Treatment Options</h4></div><ul style="margin-top:15px;">{treat_html}</ul></div>', unsafe_allow_html=True)

                prev_html = "".join([f"<li style='color:#ffe0e0;margin:8px 0;font-family:Nunito,sans-serif;'>{p}</li>" for p in info['prevention']])
                st.markdown(f'<div class="report-card" style="border-color:#27ae60;"><div class="section-header" style="background:linear-gradient(90deg,#27ae60,#1a7a3c);"><h4 style="color:#ffe0e0;margin:0;">🛡️ Prevention Tips</h4></div><ul style="margin-top:15px;">{prev_html}</ul></div>', unsafe_allow_html=True)

                rec_color = "#e74c3c" if info['risk_level'] in ['HIGH','VERY HIGH'] else "#c0392b"
                st.markdown(f'<div class="report-card" style="border-color:{rec_color};"><div class="section-header" style="background:linear-gradient(90deg,{rec_color}88,{rec_color}44);"><h4 style="color:#ffe0e0;margin:0;">✅ Doctor\'s Recommendation</h4></div><p style="color:#ffe0e0;line-height:1.8;margin-top:15px;font-size:1.05em;font-family:Nunito,sans-serif;">{info["recommendation"]}</p></div>', unsafe_allow_html=True)

                st.markdown('<div class="report-card" style="border-color:#555;opacity:0.8;"><p style="color:#ff9999;font-size:0.85em;text-align:center;margin:0;font-family:Nunito,sans-serif;">⚕️ <b>Disclaimer:</b> AI-generated report for educational purposes only. Always consult a qualified dermatologist.</p></div>', unsafe_allow_html=True)

                report_text = f"""AI-POWERED SKIN LESION ANALYSIS REPORT
=======================================
Date     : {date}  |  Time: {time_now}
Patient  : {patient_name if patient_name else 'Not specified'}
Age      : {patient_age} years
Sex      : {patient_sex if patient_sex!='Select' else 'Not specified'}
Location : {location if location!='Select' else 'Not specified'}

DETECTION RESULT
----------------
Condition  : {info['full_name']}
Type       : {info['type']}
Risk Level : {info['risk_level']}

DESCRIPTION
-----------
{info['description']}

SYMPTOMS
--------
{chr(10).join(['• '+s.replace('🔸 ','') for s in info['symptoms']])}

TREATMENT OPTIONS
-----------------
{chr(10).join(['• '+t.replace('💊 ','') for t in info['treatment']])}

PREVENTION TIPS
---------------
{chr(10).join(['• '+p for p in info['prevention']])}

RECOMMENDATION
--------------
{info['recommendation']}

DISCLAIMER
----------
This report is AI-generated for educational purposes only.
Always consult a qualified dermatologist for proper diagnosis.

Generated by: Skin Cancer Detection System
Team: Tejaswini | Sai Sreenidhi | Anusha | Sreekala | Poojitha
"""
                st.download_button("📥 Download Full Medical Report",
                    data=report_text,
                    file_name=f"skin_report_{predicted_class}_{date}.txt",
                    mime="text/plain", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure cnn_model.h5 and transfer_learning_model.h5 are in project folder!")

# ============================================================
# TAB 3 - SKIN WELLNESS
# ============================================================
with tab3:
    st.markdown("""
    <div class="report-card" style="text-align:center; padding:30px;">
        <div style="font-size:3em;">💚</div>
        <h2 style="color:#e74c3c; font-family:'Raleway',sans-serif;">Skin Wellness Guide</h2>
        <p style="color:#ffcccc; font-size:1.1em; font-family:'Nunito',sans-serif;">
            Simple daily habits that keep your skin healthy and cancer-free!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌟 Daily Skin Care Habits")

    for i in range(0, len(SKIN_TIPS), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i+j < len(SKIN_TIPS):
                tip = SKIN_TIPS[i+j]
                with col:
                    st.markdown(f"""
                    <div class="tip-card">
                        <div style="font-size:2.5em;">{tip['icon']}</div>
                        <h4 style="color:#e74c3c; margin:5px 0; font-family:'Raleway',sans-serif;">{tip['title']}</h4>
                        <p style="color:#ffcccc; font-size:0.9em; line-height:1.5; font-family:'Nunito',sans-serif;">{tip['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 ABCDE Rule for Mole Monitoring")
    for col,(letter,word,desc) in zip(st.columns(5),[
        ("A","Asymmetry","One half doesn't match"),
        ("B","Border","Irregular or blurred edges"),
        ("C","Color","Multiple colors present"),
        ("D","Diameter","Larger than 6mm"),
        ("E","Evolution","Any change over time"),
    ]):
        with col:
            st.markdown(f"""
            <div class="report-card" style="text-align:center; padding:20px;">
                <h1 style="color:#e74c3c; font-size:3em; margin:0;
                           font-family:'Raleway',sans-serif;">{letter}</h1>
                <h4 style="color:#ffcccc; margin:5px 0; font-family:'Nunito',sans-serif;">{word}</h4>
                <p style="color:#ff9999; font-size:0.85em; font-family:'Nunito',sans-serif;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌿 Foods for Healthy Skin")
    foods = [
        ("🫐","Blueberries","Rich in antioxidants protecting skin from UV damage"),
        ("🥑","Avocado","Healthy fats keep skin moisturized and elastic"),
        ("🐟","Salmon","Omega-3 fatty acids reduce inflammation"),
        ("🍅","Tomatoes","Lycopene protects against sun damage"),
        ("🥦","Broccoli","Vitamins C and K boost collagen production"),
        ("🍊","Oranges","Vitamin C essential for collagen synthesis"),
        ("🌰","Walnuts","Zinc and selenium protect from oxidative damage"),
        ("🍵","Green Tea","Catechins reduce redness and improve hydration"),
    ]
    for i in range(0, len(foods), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i+j < len(foods):
                emoji, name, benefit = foods[i+j]
                with col:
                    st.markdown(f"""
                    <div class="tip-card">
                        <div style="font-size:2.5em;">{emoji}</div>
                        <h4 style="color:#e74c3c; margin:5px 0; font-family:'Raleway',sans-serif;">{name}</h4>
                        <p style="color:#ffcccc; font-size:0.85em; font-family:'Nunito',sans-serif;">{benefit}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# TAB 4 - AI PERFORMANCE
# ============================================================
with tab4:
    st.markdown("""
    <div class="report-card" style="text-align:center;">
        <h2 style="color:#e74c3c; font-family:'Raleway',sans-serif;">📊 AI Performance Analysis</h2>
        <p style="color:#ff9999; font-family:'Nunito',sans-serif;">For judges and technical evaluation</p>
    </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="section-header"><h4 style="color:#ffe0e0;margin:0;">🤖 Model Accuracies</h4></div>', unsafe_allow_html=True)
        st.metric("🧠 Custom CNN",       "92.82%", "Primary Model ✅")
        st.metric("🤖 MobileNetV2 (TL)", "73.53%", "Transfer Learning")
    with col6:
        st.markdown('<div class="section-header"><h4 style="color:#ffe0e0;margin:0;">📊 Dataset Info</h4></div>', unsafe_allow_html=True)
        st.metric("🖼️ Total Images","10,015")
        st.metric("🏷️ Classes","7 Types")
        st.metric("📁 Source","HAM10000")

    st.markdown("---")
    df = pd.DataFrame({
        "Code"      :['akiec','bcc','bkl','df','mel','nv','vasc'],
        "Full Name" :['Actinic Keratosis','Basal Cell Carcinoma','Benign Keratosis',
                      'Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular Lesion'],
        "Risk"      :['🟡 MODERATE','🔴 HIGH','🟢 LOW','🟢 LOW',
                      '🔴🔴 VERY HIGH','🟢 LOW','🟢 LOW'],
        "Type"      :['Pre-cancerous','Malignant','Benign','Benign',
                      'Malignant','Benign','Benign']
    })
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#e74c3c;font-family:Raleway,sans-serif;">🧠 Deep Learning</h3><p style="color:#ffe0e0;font-family:Nunito,sans-serif;">TensorFlow/Keras<br>Custom CNN<br>MobileNetV2<br>Transfer Learning</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#e74c3c;font-family:Raleway,sans-serif;">💬 NLP</h3><p style="color:#ffe0e0;font-family:Nunito,sans-serif;">Medical Report Generation<br>Risk Classification<br>Dynamic Text<br>Medical Terminology</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#e74c3c;font-family:Raleway,sans-serif;">🛠️ Tools</h3><p style="color:#ffe0e0;font-family:Nunito,sans-serif;">Python 3.10<br>Streamlit<br>OpenCV<br>NumPy / Pandas</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="report-card" style="border-color:#c0392b; margin-top:20px;">
        <h3 style="color:#e74c3c; text-align:center; font-family:'Raleway',sans-serif;">💡 Project Impact</h3>
        <p style="color:#ffe0e0; line-height:2.2; font-family:'Nunito',sans-serif;">
            ✅ Detects 7 types of skin cancer automatically from images<br>
            ✅ Generates complete NLP medical reports with prevention tips<br>
            ✅ Assists doctors in rural areas without dermatologists<br>
            ✅ Melanoma survival: 98% (early) vs 23% (late detection)<br>
            ✅ Results in seconds — faster than traditional lab reports<br>
            ✅ Built on internationally recognized HAM10000 dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Team Footer
    st.markdown("""
    <div style="text-align:center; padding:20px; margin-top:10px;">
        <p style="color:#ff9999; font-family:'Nunito',sans-serif; font-size:0.9em;">
            🔬 Skin Cancer Detection System | Built with ❤️ by
            <b style="color:#ffcccc;">Tejaswini • Sai Sreenidhi • Anusha • Sreekala • Poojitha</b>
        </p>
        <p style="color:#ff7777; font-family:'Nunito',sans-serif; font-size:0.85em;">
            📚 HAM10000 Dataset | CNN + MobileNetV2 | NLP Medical Reports | Project Expo 2026
        </p>
    </div>
    """, unsafe_allow_html=True)