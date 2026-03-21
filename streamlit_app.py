# ============================================================
# STREAMLIT WEB APP - Skin Cancer Detection System
# Theme: Dark Blue Ocean + Medical Pattern Background
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
# CSS - DARK BLUE + MEDICAL PATTERN BACKGROUND
# ============================================================
st.markdown("""
<style>
    /* Medical Pattern Background */
    .stApp {
        background-color: #0a1628;
        background-image:
            radial-gradient(circle at 10% 10%, rgba(41,128,185,0.18) 0%, transparent 45%),
            radial-gradient(circle at 90% 10%, rgba(41,128,185,0.12) 0%, transparent 45%),
            radial-gradient(circle at 10% 90%, rgba(41,128,185,0.12) 0%, transparent 45%),
            radial-gradient(circle at 90% 90%, rgba(41,128,185,0.18) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(26,74,122,0.15) 0%, transparent 60%),
            radial-gradient(circle at 30% 60%, rgba(52,152,219,0.08) 0%, transparent 35%),
            radial-gradient(circle at 70% 40%, rgba(52,152,219,0.08) 0%, transparent 35%),
            linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
        background-attachment: fixed;
        color: #e0f0ff;
    }

    /* All text */
    .stMarkdown, p, h1, h2, h3, h4, label {
        color: #e0f0ff !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #1a4a7a, #2980b9) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-size: 16px !important;
        font-weight: bold !important; padding: 12px !important;
        box-shadow: 0 4px 15px rgba(41,128,185,0.4) !important;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #2980b9, #1a4a7a) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(41,128,185,0.6) !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #0d2d5c, #1a4a7a) !important;
        color: white !important;
        border: 1px solid #2980b9 !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(7,16,32,0.8) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        color: #2980b9 !important;
        font-weight: bold !important;
        font-size: 15px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a4a7a !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* File uploader */
    .stFileUploader {
        background-color: rgba(13,33,55,0.8) !important;
        border: 2px dashed #2980b9 !important;
        border-radius: 12px !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background-color: rgba(13,33,55,0.9) !important;
        color: #e0f0ff !important;
        border: 1px solid #2980b9 !important;
        border-radius: 8px !important;
    }

    /* Metrics */
    .stMetric {
        background-color: rgba(13,33,55,0.8) !important;
        border: 1px solid #2980b9 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        backdrop-filter: blur(10px);
    }

    /* Dataframe */
    .stDataFrame {
        background-color: rgba(13,33,55,0.8) !important;
        border: 1px solid #2980b9 !important;
        border-radius: 10px !important;
    }

    /* Cards */
    .report-card {
        background: linear-gradient(135deg,
            rgba(7,16,32,0.9),
            rgba(13,33,55,0.9));
        border: 1px solid #2980b9;
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(41,128,185,0.2);
        backdrop-filter: blur(10px);
    }
    .section-header {
        background: linear-gradient(90deg, #1a4a7a, #0d2d5c);
        padding: 10px 20px;
        border-radius: 8px;
        margin: 15px 0 10px 0;
        border-left: 4px solid #2980b9;
    }
    .tip-card {
        background: linear-gradient(135deg,
            rgba(7,16,32,0.9),
            rgba(13,33,55,0.9));
        border: 1px solid #2980b9;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        text-align: center;
        box-shadow: 0 2px 10px rgba(41,128,185,0.2);
        backdrop-filter: blur(10px);
    }
    .stat-box {
        background: linear-gradient(135deg,
            rgba(13,45,92,0.9),
            rgba(26,74,122,0.9));
        border: 1px solid #2980b9;
        border-top: 4px solid #2980b9;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(41,128,185,0.3);
        backdrop-filter: blur(10px);
    }
    .home-feature {
        background: linear-gradient(135deg,
            rgba(13,45,92,0.9),
            rgba(13,33,55,0.9));
        border: 1px solid #2980b9;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(41,128,185,0.3);
        backdrop-filter: blur(10px);
    }

    /* Risk banners */
    .risk-low {
        background: linear-gradient(135deg, #1a7a3c, #27ae60);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #2ecc71;
        box-shadow: 0 0 25px rgba(46,204,113,0.5);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #7a5c00, #f39c12);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #f39c12;
        box-shadow: 0 0 25px rgba(243,156,18,0.5);
    }
    .risk-high {
        background: linear-gradient(135deg, #7a0d0d, #c0392b);
        padding: 25px; border-radius: 12px; text-align: center;
        border: 2px solid #e74c3c;
        box-shadow: 0 0 25px rgba(231,76,60,0.5);
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
        'description':"Actinic Keratosis is a rough, scaly patch caused by years of sun exposure. It is a pre-cancerous condition that can develop into skin cancer if left untreated. Commonly appears on sun-exposed areas like face, ears, scalp, and neck.",
        'symptoms':["🔸 Rough, dry, scaly patch","🔸 Flat to slightly raised bump","🔸 Color: pink, red, or brown","🔸 Itching, burning, or tenderness"],
        'treatment':["💊 Cryotherapy","💊 Topical medications","💊 Photodynamic therapy","💊 Laser therapy"],
        'prevention':["☀️ Apply sunscreen SPF 30+ daily","👒 Wear protective hats outdoors","👕 Cover skin with UV-protective clothing","🕶️ Wear UV-blocking sunglasses","🏠 Avoid peak sun hours (10am-4pm)"],
        'recommendation':"Consult a dermatologist soon. Early treatment prevents progression to cancer."
    },
    'bcc': {
        'full_name':'Basal Cell Carcinoma','risk_level':'HIGH',
        'risk_color':'🔴','risk_class':'risk-high','type':'Malignant (Cancer)','icon':'🚨',
        'description':"Basal Cell Carcinoma is the most common type of skin cancer. It begins in the basal cells and often appears as a transparent bump on sun-exposed skin.",
        'symptoms':["🔸 Pearly or waxy bump","🔸 Bleeding sore that returns","🔸 Pink growth with raised edges","🔸 Flat scar-like lesion"],
        'treatment':["💊 Surgical excision","💊 Mohs surgery","💊 Radiation therapy","💊 Targeted drug therapy"],
        'prevention':["☀️ Use broad-spectrum sunscreen daily","🚫 Avoid tanning beds completely","👒 Wear wide-brimmed hats","🔍 Do monthly self-skin checks","🏥 Annual dermatologist visit"],
        'recommendation':"URGENT: Seek immediate medical attention from a certified dermatologist."
    },
    'bkl': {
        'full_name':'Benign Keratosis','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Benign Keratosis includes seborrheic keratoses and solar lentigines. These are non-cancerous growths very common in older adults and generally harmless.",
        'symptoms':["🔸 Waxy, scaly raised growth","🔸 Color: light tan to black","🔸 Round or oval shape","🔸 Occasional itching"],
        'treatment':["💊 Usually no treatment needed","💊 Cryotherapy","💊 Curettage","💊 Laser treatment"],
        'prevention':["☀️ Sun protection from early age","💧 Keep skin moisturized daily","🥗 Eat antioxidant-rich foods","💧 Stay well hydrated","🔍 Monitor moles regularly"],
        'recommendation':"Generally benign. Monitor for changes in size, shape, or color."
    },
    'df': {
        'full_name':'Dermatofibroma','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Dermatofibroma is a common benign skin growth appearing on the legs. It is a harmless fibrous nodule in the deep skin layers.",
        'symptoms':["🔸 Small hard bump under skin","🔸 Brown, red, or purple color","🔸 Skin dimples when pinched","🔸 Mild tenderness"],
        'treatment':["💊 Usually no treatment needed","💊 Surgical removal","💊 Cryotherapy","💊 Steroid injections"],
        'prevention':["🛡️ Protect skin from minor injuries","🦟 Use insect repellent","🧴 Keep skin moisturized","🔍 Monitor any new skin bumps","💪 Maintain healthy immune system"],
        'recommendation':"Benign and rarely needs treatment. Monitor for any changes."
    },
    'mel': {
        'full_name':'Melanoma','risk_level':'VERY HIGH',
        'risk_color':'🔴🔴','risk_class':'risk-high','type':'Malignant (Most Dangerous)','icon':'🚨🚨',
        'description':"Melanoma is the most dangerous skin cancer. Early detection is CRITICAL — survival rate is 98% when caught early vs only 23% in advanced stages.",
        'symptoms':["🔸 Mole changing size/shape/color","🔸 Asymmetrical shape","🔸 Irregular borders","🔸 Multiple colors","🔸 Diameter > 6mm","🔸 Bleeding or itching"],
        'treatment':["💊 Surgical excision","💊 Immunotherapy","💊 Targeted therapy","💊 Radiation therapy"],
        'prevention':["☀️ Never skip sunscreen SPF 50+","🚫 Never use tanning beds","👒 Always cover skin in sun","🔍 Check moles monthly (ABCDE rule)","🏥 Annual skin cancer screening","👨‍👩‍👧 Know your family history"],
        'recommendation':"⚠️ URGENT: See an oncologist IMMEDIATELY. Do NOT delay!"
    },
    'nv': {
        'full_name':'Melanocytic Nevi (Mole)','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Melanocytic Nevi are common moles that are almost always harmless. Monitor using the ABCDE rule for suspicious changes.",
        'symptoms':["🔸 Small dark brown spot","🔸 Symmetric round shape","🔸 Smooth even border","🔸 Uniform color","🔸 Less than 6mm"],
        'treatment':["💊 Usually no treatment needed","💊 Surgical removal if suspicious","💊 Regular monitoring","💊 Dermoscopy"],
        'prevention':["☀️ Limit sun exposure as child","🧴 Use sunscreen from young age","🔍 Monitor moles with ABCDE rule","📸 Photograph moles to track changes","🏥 Regular dermatologist visits"],
        'recommendation':"Benign mole. Monitor using ABCDE: Asymmetry, Border, Color, Diameter, Evolution."
    },
    'vasc': {
        'full_name':'Vascular Lesion','risk_level':'LOW',
        'risk_color':'🟢','risk_class':'risk-low','type':'Benign (Non-cancerous)','icon':'✅',
        'description':"Vascular lesions are blood vessel abnormalities in the skin. Most are benign and caused by abnormal blood vessel growth near the skin surface.",
        'symptoms':["🔸 Red/purple/blue spot","🔸 Flat or slightly raised","🔸 May bleed easily","🔸 Spider-vein appearance"],
        'treatment':["💊 Usually no treatment needed","💊 Laser therapy","💊 Electrosurgery","💊 Cryotherapy"],
        'prevention':["🧴 Protect skin from trauma","☀️ Use sun protection","💧 Stay hydrated","🥗 Eat vitamin C rich foods","🏃 Maintain healthy blood circulation"],
        'recommendation':"Typically benign. See doctor if it bleeds or grows rapidly."
    }
}

SKIN_TIPS = [
    {"icon":"☀️","title":"Sun Protection","tip":"Apply SPF 30+ sunscreen every day, even on cloudy days. UV rays penetrate clouds!"},
    {"icon":"💧","title":"Stay Hydrated","tip":"Drink 8+ glasses of water daily. Hydrated skin is healthy skin that fights damage better."},
    {"icon":"🥗","title":"Eat Healthy","tip":"Foods rich in antioxidants (berries, nuts, green tea) protect skin from free radical damage."},
    {"icon":"😴","title":"Sleep Well","tip":"Get 7-8 hours of sleep. Skin repairs and regenerates itself during deep sleep cycles."},
    {"icon":"🚭","title":"Avoid Smoking","tip":"Smoking reduces blood flow to skin, causing premature aging and increasing cancer risk."},
    {"icon":"🔍","title":"Self Check Monthly","tip":"Do monthly skin checks. Use ABCDE rule to monitor moles for suspicious changes."},
    {"icon":"🏥","title":"Annual Checkup","tip":"Visit a dermatologist annually for professional skin cancer screening."},
    {"icon":"👒","title":"Wear Protection","tip":"Wear wide-brimmed hats, UV-protective clothing, and sunglasses when outdoors."},
    {"icon":"🧴","title":"Moisturize Daily","tip":"Apply moisturizer to damp skin after bathing. Healthy skin barrier prevents infections."},
    {"icon":"🚫","title":"No Tanning Beds","tip":"Tanning beds increase melanoma risk by 75%. Never use them under any circumstances!"},
    {"icon":"🍊","title":"Vitamin C","tip":"Vitamin C boosts collagen production and protects against UV-induced skin damage."},
    {"icon":"🏃","title":"Exercise Regularly","tip":"Exercise improves blood circulation, delivering nutrients and oxygen to skin cells."},
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
# HEADER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg,
        rgba(13,45,92,0.95) 0%,
        rgba(26,74,122,0.95) 100%);
    padding: 35px 40px; border-radius: 20px; text-align: center;
    border: 2px solid #2980b9; margin-bottom: 25px;
    box-shadow: 0 0 50px rgba(41,128,185,0.5);
    backdrop-filter: blur(10px);
">
    <div style="font-size:3.5em; margin-bottom:10px;">🔬</div>
    <h1 style="color:#ffffff; font-size:2.5em; margin:0; font-weight:800;
               text-shadow: 0 0 20px rgba(41,128,185,0.8);">
        Skin Cancer Detection System
    </h1>
    <div style="width:80px; height:4px; background:#2980b9;
                margin:15px auto; border-radius:2px;
                box-shadow: 0 0 10px rgba(41,128,185,0.8);"></div>
    <p style="color:#aaddff; font-size:1.1em; margin:5px 0;">
        AI-Powered Early Detection • Deep Learning • NLP Medical Reports
    </p>
    <p style="color:#7ab8d9; font-size:0.9em; margin:5px 0;">
        🏥 HAM10000 Dataset &nbsp;|&nbsp; 🧠 CNN Model &nbsp;|&nbsp;
        🤖 MobileNetV2 &nbsp;|&nbsp; 🔬 7 Class Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🔬 Detection & Report",
    "💙 Skin Health Tips",
    "📊 Model Analysis"
])

# ============================================================
# TAB 1 - HOME PAGE
# ============================================================
with tab1:

    # Hero Banner
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,
            rgba(7,16,32,0.95) 0%,
            rgba(13,33,55,0.95) 100%);
        border: 2px solid #2980b9; border-radius: 20px;
        padding: 50px 40px; text-align: center; margin-bottom: 30px;
        box-shadow: 0 0 50px rgba(41,128,185,0.3);
        backdrop-filter: blur(10px);
    ">
        <div style="font-size:4em; margin-bottom:10px;">🏥</div>
        <h1 style="color:#ffffff; font-size:2.8em; margin:0; font-weight:800;
                   text-shadow: 0 0 20px rgba(41,128,185,0.6);">
            Welcome to AI Skin Cancer Detection
        </h1>
        <div style="width:80px; height:4px; background:#2980b9;
                    margin:15px auto; border-radius:2px;
                    box-shadow: 0 0 10px rgba(41,128,185,0.8);"></div>
        <p style="color:#aaddff; font-size:1.2em; margin:10px 0; line-height:1.6;">
            AI-Powered Early Detection using Deep Learning &amp; NLP
        </p>
        <p style="color:#7ab8d9; font-size:0.95em; margin:5px 0;">
            🔬 HAM10000 Dataset &nbsp;|&nbsp;
            🧠 CNN Model &nbsp;|&nbsp;
            🤖 MobileNetV2 &nbsp;|&nbsp;
            📋 NLP Medical Reports
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats Row
    i1, i2, i3, i4 = st.columns(4)
    for col, (icon, val, label) in zip([i1,i2,i3,i4], [
        ("🖼️","10,015","Training Images"),
        ("🏷️","7","Disease Classes"),
        ("🎯","92.82%","CNN Accuracy"),
        ("⚡","< 5s","Report Time"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div style="font-size:2em;">{icon}</div>
                <h2 style="color:#2980b9; margin:5px 0; font-size:2em;
                           text-shadow: 0 0 10px rgba(41,128,185,0.6);">{val}</h2>
                <p style="color:#aaddff; margin:0; font-size:0.9em;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1,1])

    with left:
        st.markdown("""
        <div style="
            background:rgba(7,16,32,0.9);
            border:1px solid #2980b9; border-left:5px solid #2980b9;
            border-radius:10px; padding:25px; margin-bottom:20px;
            backdrop-filter:blur(10px);
            box-shadow: 0 4px 20px rgba(41,128,185,0.2);
        ">
            <h3 style="color:#2980b9; margin-top:0;">🔬 About This System</h3>
            <p style="color:#e0f0ff; line-height:1.9;">
                This AI-powered system uses <b>Deep Learning</b> and
                <b>Natural Language Processing</b> to detect
                <b>7 types of skin lesions</b> from dermatoscopic images.
                It automatically generates a complete <b>medical report</b>
                with diagnosis, symptoms, treatment options, and
                personalized recommendations.
            </p>
            <p style="color:#aaddff; line-height:1.9;">
                Built on the internationally recognized
                <b>HAM10000 dataset</b> from Vienna General Hospital,
                used in thousands of research publications worldwide.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:rgba(7,16,32,0.9);
            border:1px solid #2980b9; border-left:5px solid #27ae60;
            border-radius:10px; padding:25px;
            backdrop-filter:blur(10px);
            box-shadow: 0 4px 20px rgba(41,128,185,0.2);
        ">
            <h3 style="color:#2980b9; margin-top:0;">🏥 Detectable Conditions</h3>
        """, unsafe_allow_html=True)

        for risk, name, desc in [
            ("🟢","Melanocytic Nevi",    "Benign Mole"),
            ("🟢","Benign Keratosis",    "Non-cancerous Growth"),
            ("🟢","Dermatofibroma",      "Benign Nodule"),
            ("🟢","Vascular Lesion",     "Blood Vessel Growth"),
            ("🟡","Actinic Keratosis",   "Pre-cancerous Patch"),
            ("🔴","Basal Cell Carcinoma","Skin Cancer"),
            ("🔴","Melanoma",            "Most Dangerous Cancer"),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:8px 0;
                        border-bottom:1px solid #2980b922;">
                <span style="font-size:1.2em; margin-right:10px;">{risk}</span>
                <div>
                    <span style="color:#e0f0ff; font-weight:bold;">{name}</span>
                    <span style="color:#7ab8d9; font-size:0.85em; margin-left:8px;">— {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="
            background:rgba(7,16,32,0.9);
            border:1px solid #2980b9; border-left:5px solid #3498db;
            border-radius:10px; padding:25px; margin-bottom:20px;
            backdrop-filter:blur(10px);
            box-shadow: 0 4px 20px rgba(41,128,185,0.2);
        ">
            <h3 style="color:#2980b9; margin-top:0;">📋 How It Works</h3>
        """, unsafe_allow_html=True)

        for num, title, desc in [
            ("1","Upload Image","Upload a clear photo of the skin lesion (JPG/PNG)"),
            ("2","Enter Patient Info","Add patient name, age, sex and lesion location"),
            ("3","AI Analysis","CNN model analyzes the image in seconds"),
            ("4","Get Medical Report","Full NLP report with diagnosis & recommendations"),
            ("5","Download Report","Save report as text file for doctor consultation"),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; padding:12px 0;
                        border-bottom:1px solid #2980b922;">
                <div style="
                    background:#2980b9; color:white; border-radius:50%;
                    width:30px; height:30px; display:flex; align-items:center;
                    justify-content:center; font-weight:bold;
                    min-width:30px; margin-right:15px; margin-top:2px;
                    box-shadow: 0 0 10px rgba(41,128,185,0.6);
                ">{num}</div>
                <div>
                    <p style="color:#e0f0ff; font-weight:bold; margin:0;">{title}</p>
                    <p style="color:#7ab8d9; font-size:0.9em; margin:3px 0 0 0;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Survival stats
        st.markdown("""
        <div style="
            background:linear-gradient(135deg,rgba(26,5,5,0.95),rgba(44,10,10,0.95));
            border:1px solid #e74c3c; border-left:5px solid #e74c3c;
            border-radius:10px; padding:25px;
            backdrop-filter:blur(10px);
            box-shadow: 0 4px 20px rgba(231,76,60,0.3);
        ">
            <h3 style="color:#e74c3c; margin-top:0;">⚠️ Why Early Detection Matters</h3>
            <div style="display:flex; justify-content:space-around;
                        align-items:center; margin:20px 0;">
                <div style="text-align:center;">
                    <h2 style="color:#2ecc71; font-size:3em; margin:0;
                               text-shadow: 0 0 15px rgba(46,204,113,0.6);">98%</h2>
                    <p style="color:#a0ffa0; margin:5px 0;">Survival Rate</p>
                    <p style="color:#7acc7a; font-size:0.85em; margin:0;">Early Detection</p>
                </div>
                <div style="color:#e74c3c; font-size:2em; font-weight:bold;">VS</div>
                <div style="text-align:center;">
                    <h2 style="color:#e74c3c; font-size:3em; margin:0;
                               text-shadow: 0 0 15px rgba(231,76,60,0.6);">23%</h2>
                    <p style="color:#ffaaaa; margin:5px 0;">Survival Rate</p>
                    <p style="color:#ff7777; font-size:0.85em; margin:0;">Late Detection</p>
                </div>
            </div>
            <p style="color:#ffaaaa; text-align:center; font-size:0.95em; margin:0;">
                🔬 Melanoma is <b>highly curable</b> when detected early!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background:rgba(7,16,32,0.8); border:1px solid #555;
        border-radius:10px; padding:20px; text-align:center;
        backdrop-filter:blur(10px);
    ">
        <p style="color:#888; font-size:0.9em; margin:0; line-height:1.8;">
            ⚕️ <b style="color:#aaa;">Medical Disclaimer:</b>
            This system is developed for educational and research purposes only.
            It is <b>NOT</b> a substitute for professional medical diagnosis.
            Always consult a qualified dermatologist for proper diagnosis and treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2 - DETECTION & REPORT
# ============================================================
with tab2:
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="section-header"><h4 style="color:#e0f0ff;margin:0;">📤 Upload Skin Lesion Image</h4></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose image (JPG, JPEG, PNG)", type=['jpg','jpeg','png'])
        st.markdown('<div class="section-header"><h4 style="color:#e0f0ff;margin:0;">👤 Patient Information</h4></div>', unsafe_allow_html=True)
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
            <div style="background:rgba(7,16,32,0.9); border:2px dashed #2980b9;
                        border-radius:15px; padding:80px; text-align:center;
                        backdrop-filter:blur(10px);">
                <div style="font-size:4em;">📸</div>
                <h3 style="color:#2980b9;">Image Preview</h3>
                <p style="color:#7ab8d9;">Upload an image to see preview here</p>
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
                <div class="{info['risk_class']}">
                    <div style="font-size:3em;">{info['icon']}</div>
                    <h1 style="color:white; margin:5px 0;">{info['full_name']}</h1>
                    <h2 style="color:white; margin:5px 0;">Risk: {info['risk_color']} {info['risk_level']}</h2>
                    <h3 style="color:white; margin:5px 0; opacity:0.9;">{info['type']}</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-header"><h3 style="color:#e0f0ff;margin:0;">📋 AI-Generated Medical Report</h3></div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="report-card">
                    <table style="width:100%; color:#e0f0ff; border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #2980b944;">
                            <td style="padding:8px;">📅 <b>Date</b></td><td style="padding:8px;">{date}</td>
                            <td style="padding:8px;">⏰ <b>Time</b></td><td style="padding:8px;">{time_now}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #2980b944;">
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

                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#e0f0ff;margin:0;">📖 Medical Description</h4></div><p style="color:#e0f0ff;line-height:1.8;margin-top:15px;">{info["description"]}</p></div>', unsafe_allow_html=True)

                symp_html = "".join([f"<li style='color:#e0f0ff;margin:8px 0;'>{s}</li>" for s in info['symptoms']])
                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#e0f0ff;margin:0;">⚠️ Common Symptoms</h4></div><ul style="margin-top:15px;">{symp_html}</ul></div>', unsafe_allow_html=True)

                treat_html = "".join([f"<li style='color:#e0f0ff;margin:8px 0;'>{t}</li>" for t in info['treatment']])
                st.markdown(f'<div class="report-card"><div class="section-header"><h4 style="color:#e0f0ff;margin:0;">💊 Treatment Options</h4></div><ul style="margin-top:15px;">{treat_html}</ul></div>', unsafe_allow_html=True)

                prev_html = "".join([f"<li style='color:#e0f0ff;margin:8px 0;'>{p}</li>" for p in info['prevention']])
                st.markdown(f'<div class="report-card" style="border-color:#27ae60;"><div class="section-header" style="background:linear-gradient(90deg,#27ae60,#1a7a3c);"><h4 style="color:#e0f0ff;margin:0;">🛡️ Prevention Tips</h4></div><ul style="margin-top:15px;">{prev_html}</ul></div>', unsafe_allow_html=True)

                rec_color = "#e74c3c" if info['risk_level'] in ['HIGH','VERY HIGH'] else "#2980b9"
                st.markdown(f'<div class="report-card" style="border-color:{rec_color};"><div class="section-header" style="background:linear-gradient(90deg,{rec_color}88,{rec_color}44);"><h4 style="color:#e0f0ff;margin:0;">✅ Doctor\'s Recommendation</h4></div><p style="color:#e0f0ff;line-height:1.8;margin-top:15px;font-size:1.05em;">{info["recommendation"]}</p></div>', unsafe_allow_html=True)

                st.markdown('<div class="report-card" style="border-color:#555;opacity:0.8;"><p style="color:#aaa;font-size:0.85em;text-align:center;margin:0;">⚕️ <b>Disclaimer:</b> AI-generated report for educational purposes only. Always consult a qualified dermatologist for proper diagnosis.</p></div>', unsafe_allow_html=True)

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
"""
                st.download_button("📥 Download Full Medical Report",
                    data=report_text,
                    file_name=f"skin_report_{predicted_class}_{date}.txt",
                    mime="text/plain", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure cnn_model.h5 and transfer_learning_model.h5 are in project folder!")

# ============================================================
# TAB 3 - SKIN HEALTH TIPS
# ============================================================
with tab3:
    st.markdown('<div class="report-card" style="text-align:center;padding:30px;"><div style="font-size:3em;">💙</div><h2 style="color:#2980b9;">Healthy Skin Tips</h2><p style="color:#aaddff;font-size:1.1em;">Simple daily habits that keep your skin healthy and cancer-free!</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🌟 Daily Skin Care Habits")

    for i in range(0, len(SKIN_TIPS), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i+j < len(SKIN_TIPS):
                tip = SKIN_TIPS[i+j]
                with col:
                    st.markdown(f'<div class="tip-card"><div style="font-size:2.5em;">{tip["icon"]}</div><h4 style="color:#2980b9;margin:5px 0;">{tip["title"]}</h4><p style="color:#aaddff;font-size:0.9em;line-height:1.5;">{tip["tip"]}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 ABCDE Rule for Mole Monitoring")
    for col, (letter, word, desc) in zip(st.columns(5), [
        ("A","Asymmetry","One half doesn't match other half"),
        ("B","Border","Edges are irregular or blurred"),
        ("C","Color","Multiple colors in one lesion"),
        ("D","Diameter","Larger than 6mm"),
        ("E","Evolution","Any change over time"),
    ]):
        with col:
            st.markdown(f'<div class="report-card" style="text-align:center;padding:20px;"><h1 style="color:#2980b9;font-size:3em;margin:0;text-shadow:0 0 10px rgba(41,128,185,0.6);">{letter}</h1><h4 style="color:#aaddff;margin:5px 0;">{word}</h4><p style="color:#7ab8d9;font-size:0.85em;">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌿 Foods for Healthy Skin")
    foods = [
        ("🫐","Blueberries","Rich in antioxidants that protect skin from UV damage"),
        ("🥑","Avocado","Healthy fats keep skin moisturized and elastic"),
        ("🐟","Salmon","Omega-3 fatty acids reduce inflammation"),
        ("🍅","Tomatoes","Lycopene protects against sun damage"),
        ("🥦","Broccoli","Vitamins C and K boost collagen"),
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
                    st.markdown(f'<div class="tip-card"><div style="font-size:2.5em;">{emoji}</div><h4 style="color:#2980b9;margin:5px 0;">{name}</h4><p style="color:#aaddff;font-size:0.85em;">{benefit}</p></div>', unsafe_allow_html=True)

# ============================================================
# TAB 4 - MODEL ANALYSIS
# ============================================================
with tab4:
    st.markdown('<div class="report-card" style="text-align:center;"><h2 style="color:#2980b9;">📊 Technical Model Analysis</h2><p style="color:#7ab8d9;">For judges and technical evaluation</p></div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="section-header"><h4 style="color:#e0f0ff;margin:0;">🤖 Model Accuracies</h4></div>', unsafe_allow_html=True)
        st.metric("🧠 Custom CNN",       "92.82%", "Primary Model ✅")
        st.metric("🤖 MobileNetV2 (TL)", "73.53%", "Transfer Learning")
    with col6:
        st.markdown('<div class="section-header"><h4 style="color:#e0f0ff;margin:0;">📊 Dataset Info</h4></div>', unsafe_allow_html=True)
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
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#2980b9;">🧠 Deep Learning</h3><p style="color:#e0f0ff;">TensorFlow/Keras<br>Custom CNN<br>MobileNetV2<br>Transfer Learning</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#3498db;">💬 NLP</h3><p style="color:#e0f0ff;">Medical Report Generation<br>Risk Classification<br>Dynamic Text<br>Medical Terminology</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="report-card" style="text-align:center;"><h3 style="color:#e74c3c;">🛠️ Tools</h3><p style="color:#e0f0ff;">Python 3.10<br>Streamlit<br>OpenCV<br>NumPy / Pandas</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="report-card" style="border-color:#2980b9;margin-top:20px;"><h3 style="color:#2980b9;text-align:center;">💡 Project Impact</h3><p style="color:#e0f0ff;line-height:2.2;">✅ Detects 7 types of skin cancer automatically<br>✅ Generates NLP medical reports with prevention tips<br>✅ Assists doctors in rural areas without dermatologists<br>✅ Melanoma survival: 98% (early) vs 23% (late detection)<br>✅ Results in seconds — faster than lab reports<br>✅ Built on internationally recognized HAM10000 dataset</p></div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#7ab8d9;padding:20px;"><p>🔬 Skin Cancer Detection System | Built with ❤️ using Python + Streamlit + TensorFlow</p><p>📚 Dataset: HAM10000 | Models: CNN + MobileNetV2 | NLP: Medical Report Generation</p></div>', unsafe_allow_html=True)