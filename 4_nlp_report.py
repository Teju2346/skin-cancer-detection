# ============================================================
# PART 4: NLP MEDICAL REPORT GENERATOR
# Skin Cancer HAM10000
# Run: python 4_nlp_report.py
# ============================================================

import json
import datetime

# ============================================================
# MEDICAL KNOWLEDGE BASE (NLP Data Source)
# ============================================================

DISEASE_INFO = {
    'akiec': {
        'full_name'   : 'Actinic Keratosis / Intraepithelial Carcinoma',
        'risk_level'  : 'MODERATE',
        'risk_color'  : '🟡',
        'type'        : 'Pre-cancerous',
        'description' : (
            "Actinic Keratosis is a rough, scaly patch on the skin caused by years "
            "of sun exposure. It is considered a pre-cancerous condition, meaning it "
            "has the potential to develop into skin cancer if left untreated. "
            "It commonly appears on sun-exposed areas such as the face, lips, ears, "
            "back of hands, forearms, scalp, and neck."
        ),
        'symptoms'    : [
            "Rough, dry, scaly patch of skin",
            "Flat to slightly raised patch or bump",
            "Hard, wart-like surface in some cases",
            "Color variations: pink, red, or brown",
            "Itching, burning, or tenderness"
        ],
        'treatment'   : [
            "Cryotherapy (freezing with liquid nitrogen)",
            "Topical medications (5-fluorouracil, imiquimod)",
            "Photodynamic therapy",
            "Laser therapy",
            "Chemical peeling"
        ],
        'recommendation': (
            "Consult a dermatologist as soon as possible. "
            "While not immediately dangerous, untreated Actinic Keratosis "
            "can progress to Squamous Cell Carcinoma. Early treatment is advised."
        )
    },
    'bcc': {
        'full_name'   : 'Basal Cell Carcinoma',
        'risk_level'  : 'HIGH',
        'risk_color'  : '🔴',
        'type'        : 'Malignant (Cancer)',
        'description' : (
            "Basal Cell Carcinoma (BCC) is the most common type of skin cancer. "
            "It begins in the basal cells and often appears as a slightly transparent "
            "bump on the skin. It occurs most often on areas exposed to the sun."
        ),
        'symptoms'    : [
            "Pearly or waxy bump on the skin",
            "Flat, flesh-colored or brown scar-like lesion",
            "Bleeding or scabbing sore that heals and returns",
            "Pink growth with raised edges",
            "White, waxy scar-like lesion"
        ],
        'treatment'   : [
            "Surgical excision",
            "Mohs surgery (for facial BCCs)",
            "Radiation therapy",
            "Topical chemotherapy",
            "Targeted drug therapy"
        ],
        'recommendation': (
            "URGENT: Seek immediate medical attention from a certified dermatologist "
            "or oncologist. Basal Cell Carcinoma is a form of skin cancer and requires "
            "prompt professional evaluation and treatment planning."
        )
    },
    'bkl': {
        'full_name'   : 'Benign Keratosis-like Lesions',
        'risk_level'  : 'LOW',
        'risk_color'  : '🟢',
        'type'        : 'Benign (Non-cancerous)',
        'description' : (
            "Benign Keratosis-like Lesions include seborrheic keratoses and solar "
            "lentigines. These are non-cancerous skin growths that often appear as "
            "brown, black, or pale growths. They are very common in older adults "
            "and are generally harmless."
        ),
        'symptoms'    : [
            "Waxy, scaly, slightly raised growth",
            "Color range: light tan to black",
            "Round or oval shape",
            "Size from very small to more than 1 inch",
            "Occasional itching"
        ],
        'treatment'   : [
            "Usually no treatment needed",
            "Cryotherapy if cosmetically undesirable",
            "Curettage (scraping off)",
            "Electrosurgery",
            "Laser treatment"
        ],
        'recommendation': (
            "This condition is generally benign and not life-threatening. "
            "However, if the lesion changes in size, shape, or color, "
            "a dermatologist consultation is recommended for confirmation."
        )
    },
    'df': {
        'full_name'   : 'Dermatofibroma',
        'risk_level'  : 'LOW',
        'risk_color'  : '🟢',
        'type'        : 'Benign (Non-cancerous)',
        'description' : (
            "Dermatofibroma is a common benign skin growth that most often appears "
            "on the legs. It is a harmless fibrous nodule that develops in the deep "
            "layers of the skin. These growths are typically round and brownish."
        ),
        'symptoms'    : [
            "Small, hard bump under the skin",
            "Brown, red, or purple coloring",
            "Skin dimples when pinched",
            "Usually less than 1cm in diameter",
            "Mild tenderness or itching occasionally"
        ],
        'treatment'   : [
            "Usually no treatment necessary",
            "Surgical removal if bothersome",
            "Cryotherapy",
            "Steroid injections to flatten the lesion"
        ],
        'recommendation': (
            "Dermatofibroma is benign and rarely requires treatment. "
            "If the lesion grows rapidly, bleeds, or changes appearance, "
            "please consult a dermatologist for further evaluation."
        )
    },
    'mel': {
        'full_name'   : 'Melanoma',
        'risk_level'  : 'VERY HIGH',
        'risk_color'  : '🔴🔴',
        'type'        : 'Malignant (Most Dangerous Skin Cancer)',
        'description' : (
            "Melanoma is the most serious and dangerous type of skin cancer. "
            "It develops in the melanocytes that produce melanin. The exact cause "
            "is unclear, but exposure to ultraviolet radiation increases the risk "
            "significantly. Early detection is critical for survival."
        ),
        'symptoms'    : [
            "A mole that changes in size, shape, or color",
            "Asymmetrical shape with irregular borders",
            "Multiple colors in a single lesion",
            "Diameter larger than 6mm",
            "Evolution or change over time",
            "Possible bleeding or itching"
        ],
        'treatment'   : [
            "Surgical excision (primary treatment)",
            "Immunotherapy",
            "Targeted therapy (BRAF inhibitors)",
            "Radiation therapy",
            "Chemotherapy (advanced cases)"
        ],
        'recommendation': (
            "⚠️ URGENT: Immediate consultation with an oncologist or dermatologist "
            "is STRONGLY advised. Melanoma is the deadliest form of skin cancer. "
            "Early detection dramatically improves survival rates from 98% when "
            "caught early to only 23% in advanced stages. Do NOT delay."
        )
    },
    'nv': {
        'full_name'   : 'Melanocytic Nevi (Mole)',
        'risk_level'  : 'LOW',
        'risk_color'  : '🟢',
        'type'        : 'Benign (Non-cancerous)',
        'description' : (
            "Melanocytic Nevi, commonly known as moles, are benign growths on the "
            "skin formed due to clusters of pigment-forming cells. Most people have "
            "between 10 and 40 moles and they are almost always harmless. However, "
            "moles should be monitored for changes that could indicate melanoma."
        ),
        'symptoms'    : [
            "Small, dark brown spot",
            "Symmetric, round or oval shape",
            "Smooth, even border",
            "Uniform single color",
            "Generally less than 6mm in diameter",
            "Flat or slightly raised"
        ],
        'treatment'   : [
            "Usually no treatment required",
            "Surgical removal if suspicious",
            "Regular monitoring recommended",
            "Dermoscopy for evaluation"
        ],
        'recommendation': (
            "This appears to be a benign mole. No immediate action is required. "
            "However, use the ABCDE rule to monitor it: Asymmetry, Border, Color, "
            "Diameter, Evolution. If any changes occur, consult a dermatologist."
        )
    },
    'vasc': {
        'full_name'   : 'Vascular Lesion',
        'risk_level'  : 'LOW',
        'risk_color'  : '🟢',
        'type'        : 'Benign (Non-cancerous)',
        'description' : (
            "Vascular lesions are abnormalities of blood vessels in or under the "
            "skin, including cherry angiomas and pyogenic granulomas. Most vascular "
            "lesions are benign and caused by abnormal growth of blood vessels "
            "near the skin surface."
        ),
        'symptoms'    : [
            "Red, purple, or blue-colored spot",
            "Flat or slightly raised appearance",
            "May bleed easily when injured",
            "Bright red cherry-like spots",
            "Spider-vein like appearance in some cases"
        ],
        'treatment'   : [
            "Usually no treatment needed",
            "Laser therapy for cosmetic removal",
            "Electrosurgery",
            "Cryotherapy",
            "Surgical excision if necessary"
        ],
        'recommendation': (
            "Vascular lesions are typically benign and not dangerous. "
            "If the lesion bleeds frequently, grows rapidly, or causes discomfort, "
            "please consult a dermatologist for proper evaluation and treatment."
        )
    }
}

# ============================================================
# NLP REPORT GENERATOR FUNCTION
# ============================================================

def generate_medical_report(predicted_class, confidence,
                             patient_age=None, patient_sex=None,
                             location=None):
    info = DISEASE_INFO[predicted_class]
    date = datetime.datetime.now().strftime("%B %d, %Y")
    time = datetime.datetime.now().strftime("%I:%M %p")

    # NLP: Risk-level to natural language
    if info['risk_level'] == 'LOW':
        risk_sentence = "The detected condition is LOW RISK and is generally benign."
    elif info['risk_level'] == 'MODERATE':
        risk_sentence = "The detected condition is MODERATE RISK and requires attention."
    elif info['risk_level'] == 'HIGH':
        risk_sentence = "The detected condition is HIGH RISK. Immediate attention advised."
    else:
        risk_sentence = "The detected condition is VERY HIGH RISK. URGENT attention required."

    # NLP: Confidence score to natural language
    if confidence >= 90:
        conf_sentence = f"The AI model predicts this with HIGH confidence of {confidence:.1f}%."
    elif confidence >= 75:
        conf_sentence = f"The AI model predicts this with MODERATE confidence of {confidence:.1f}%."
    else:
        conf_sentence = f"The AI model predicts this with {confidence:.1f}% confidence. Professional confirmation advised."

    # NLP: Patient entity text
    patient_info = ""
    if patient_age:
        patient_info += f"  Patient Age    : {patient_age} years\n"
    if patient_sex:
        patient_info += f"  Patient Sex    : {patient_sex.capitalize()}\n"
    if location:
        patient_info += f"  Lesion Location: {location.capitalize()}\n"

    symptoms_text  = "\n".join([f"  • {s}" for s in info['symptoms']])
    treatment_text = "\n".join([f"  • {t}" for t in info['treatment']])

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         AI-POWERED SKIN LESION ANALYSIS REPORT              ║
║        Powered by Deep Learning + NLP Generation            ║
╚══════════════════════════════════════════════════════════════╝

  📅 Report Date : {date}
  ⏰ Report Time : {time}
  🤖 AI Model    : Custom CNN + MobileNetV2 Transfer Learning
{patient_info}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 PREDICTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Detected Condition : {info['full_name']}
  Condition Code     : {predicted_class.upper()}
  Condition Type     : {info['type']}
  Risk Level         : {info['risk_color']} {info['risk_level']}
  AI Confidence      : {confidence:.1f}%

  {conf_sentence}
  {risk_sentence}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 MEDICAL DESCRIPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {info['description']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  OBSERVED SYMPTOMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{symptoms_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💊 POSSIBLE TREATMENT OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{treatment_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {info['recommendation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚕️  DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  This report is generated by an AI system for educational
  and assistive purposes only. It is NOT a substitute for
  professional medical diagnosis. Always consult a qualified
  dermatologist or medical professional for proper diagnosis.

╔══════════════════════════════════════════════════════════════╗
║           END OF AI-GENERATED MEDICAL REPORT                ║
╚══════════════════════════════════════════════════════════════╝
"""
    return report

# ============================================================
# TEST ALL 7 CLASSES
# ============================================================
print("="*60)
print("TESTING NLP MEDICAL REPORT GENERATOR")
print("="*60)

test_cases = [
    ('mel',   94.3, 45, 'male',   'back'),
    ('bcc',   88.7, 62, 'female', 'face'),
    ('nv',    91.2, 28, 'female', 'arm'),
    ('akiec', 79.5, 55, 'male',   'scalp'),
    ('bkl',   85.0, 50, 'female', 'chest'),
    ('df',    76.3, 35, 'male',   'leg'),
    ('vasc',  82.1, 40, 'female', 'neck'),
]

for pred_class, confidence, age, sex, loc in test_cases:
    report = generate_medical_report(pred_class, confidence, age, sex, loc)
    print(report)
    filename = f'report_{pred_class}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Saved: {filename}")

# Save the generator as importable module
with open('nlp_report_generator.py', 'w', encoding='utf-8') as f:
    f.write(open(__file__, encoding='utf-8').read())
print("\nSaved: nlp_report_generator.py (for Streamlit import)")

print("\n" + "="*60)
print("NLP REPORT GENERATOR COMPLETE! 🎉")
print("="*60)
print("Generated 7 sample reports:")
for pred_class, _, _, _, _ in test_cases:
    print(f"  report_{pred_class}.txt")
print("\nNext Step: Run streamlit run streamlit_app.py")
