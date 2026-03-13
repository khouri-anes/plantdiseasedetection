import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AI Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide"
)



# Custom CSS for UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp > header {
        background-color: transparent;
    }
    .header-text {
        text-align: center; 
        color: #2E7D32; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-text {
        text-align: center; 
        color: #555; 
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #e9ecef;
    }
    .severity-mild { color: #28a745; font-weight: bold; }
    .severity-moderate { color: #ffc107; font-weight: bold; }
    .severity-severe { color: #fd7e14; font-weight: bold; }
    .severity-critical { color: #dc3545; font-weight: bold; }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2E7D32;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='header-text'>🌿 AI Plant Disease Detection & Severity Estimation</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Real-time agricultural diagnostic system for pepper, potato, and tomato crops.</p>",
            unsafe_allow_html=True)


# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/segment/runs_segmentation/plant_disease_seg_m/weights/best.pt")

model = load_model()

# --------------------------------
# TREATMENT DATABASE
# --------------------------------
TREATMENT_DB = {
    "Bacterial_spot": "Apply copper-based bactericide. Remove infected leaves.",
    "Early_blight": "Use fungicide (chlorothalonil). Improve air circulation.",
    "Late_blight": "Apply systemic fungicide immediately. Avoid leaf wetness.",
    "Leaf_Mold": "Reduce humidity. Apply sulfur-based fungicide.",
    "Septoria_leaf_spot": "Remove affected leaves. Use fungicide spray.",
    "Spider_mites_Two_spotted_spider_mite": "Use miticide or neem oil.",
    "Target_Spot": "Apply fungicide and avoid overhead irrigation.",
    "Tomato_YellowLeaf_Curl_Virus": "Remove infected plants. Control whiteflies.",
    "Tomato_mosaic_virus": "Remove infected plants. Sanitize tools.",
}

# --------------------------------
# LEAF AREA ESTIMATION
# --------------------------------
def estimate_leaf_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    area = np.count_nonzero(mask)
    return max(area, 1)

# --------------------------------
# SEVERITY CLASSIFICATION
# --------------------------------
def classify_severity(percentage, class_name):

    if "Virus" in class_name:
        if percentage <= 5:
            return "Mild"
        elif percentage <= 15:
            return "Moderate"
        elif percentage <= 30:
            return "Severe"
        else:
            return "Critical"
    else:
        if  percentage <= 5:
            return "Mild"
        elif percentage <= 10:
            return "Moderate"
        elif percentage <= 25:
            return "Severe"
        else:
            return "Critical"

# --------------------------------
# PDF GENERATOR
# --------------------------------
def generate_pdf(report_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Plant Disease Diagnostic Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))

    for item in report_data:
        text = f"""
        Disease: {item['name']} <br/>
        Severity: {item['severity']} <br/>
        Infected Area: {item['percentage']:.2f}% <br/>
        Treatment: {item['treatment']} <br/><br/>
        """
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --------------------------------
# MAIN LAYOUT
# --------------------------------
def get_severity_class(severity):
    if severity == "Mild": return "severity-mild"
    if severity == "Moderate": return "severity-moderate"
    if severity == "Severe": return "severity-severe"
    if severity == "Critical": return "severity-critical"
    return "severity-mild"

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📸 Image Upload")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"], help="Supported formats: JPG, PNG, JPEG")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Original Plant Leaf", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown("<div class='card' style='text-align: center; color: #666; padding: 40px;'>", unsafe_allow_html=True)
        st.write("👈 Please upload an image of a plant leaf from the left panel to begin the AI diagnosis.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔬 Diagnostic Results")
        
        with st.spinner("Analyzing image with AI model..."):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = model.predict(img, conf=0.4, device=device)
            result = results[0]
            leaf_area = estimate_leaf_area(img)

            if result.masks is None:
                st.success("✅ Analysis Complete! No disease detected. The plant appears perfectly healthy.")
            else:
                st.success("✅ Analysis Complete! Diseases detected from the image.")
                
                with st.expander("�️ View AI Segmentation Map", expanded=True):
                    plotted = result.plot(labels=False)
                    st.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Detailed Segmentation Output")

                st.divider()
                st.markdown("### 📊 Severity Analysis")
                
                disease_masks = {}
                report_data = []

                for mask, box in zip(result.masks.data, result.boxes):
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    if "healthy" in class_name.lower():
                        continue

                    m = mask.cpu().numpy()
                    m_resized = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                    if class_name not in disease_masks:
                        disease_masks[class_name] = m_resized > 0.5
                    else:
                        disease_masks[class_name] |= m_resized > 0.5

                for class_name, mask in disease_masks.items():
                    infected_area = np.count_nonzero(mask)
                    percentage = (infected_area / leaf_area) * 100
                    percentage = min(percentage, 100)
                    severity = classify_severity(percentage, class_name)
                    
                    formated_class_name = class_name.replace('_', ' ')
                    
                    st.markdown(f"#### 🦠 Condition: **{formated_class_name}**")
                    
                    col_met1, col_met2 = st.columns(2)
                    col_met1.metric(label="Infected Area", value=f"{percentage:.2f}%")
                    col_met2.markdown(f"<div style='margin-bottom: 5px; font-size: 14px; color: #555;'>Severity Level</div><div class='{get_severity_class(severity)}' style='font-size: 1.5rem;'>{severity}</div>", unsafe_allow_html=True)
                    
                    st.progress(int(percentage))

                    treatment = "No specific recommendation."
                    for key in TREATMENT_DB:
                        if key in class_name:
                            treatment = TREATMENT_DB[key]

                    st.info(f"🧪 **Treatment Recommendation:** {treatment}")

                    report_data.append({
                        "name": formated_class_name,
                        "severity": severity,
                        "percentage": percentage,
                        "treatment": treatment
                    })
                    st.write("") 

                # --------------------------------
                # DOWNLOAD PDF
                # --------------------------------
                if report_data:
                    st.divider()
                    st.markdown("### 📥 Export Report")
                    pdf_file = generate_pdf(report_data)
                    st.download_button(
                        label="📄 Download Full Diagnostic Report (PDF)",
                        data=pdf_file,
                        file_name="plant_diagnosis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)