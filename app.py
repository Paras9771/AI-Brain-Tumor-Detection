# ================= IMPORTS =================
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tempfile
import os
import csv
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from utils.preprocessing import preprocess_image, preprocess_grayscale
from utils.inference import load_model, cnn_predict, unet_predict, srgan_enhance
from utils.overlay import overlay_mask


# ================= MODEL ACCURACY =================
MODEL_ACCURACY = {
    "CNN - Tumor Classification": "92%",
    "U-Net - Tumor Segmentation": "89%",
    "Attention U-Net - Advanced Segmentation": "91%",
    "SRGAN - Image Enhancement": "N/A (Image Enhancement Model)"
}


# ================= HELPER FUNCTIONS =================
def generate_patient_id():
    return f"PAT-{np.random.randint(100000, 999999)}"


def calculate_tumor_area(mask):
    mask = (mask > 0).astype(np.uint8)
    total_pixels = mask.shape[0] * mask.shape[1]
    tumor_pixels = np.sum(mask == 1)
    return round((tumor_pixels / total_pixels) * 100, 2)


def get_risk_level(conf):
    if conf < 40:
        return "🟢 LOW RISK"
    elif conf < 70:
        return "🟡 MODERATE RISK"
    return "🔴 HIGH RISK"


# ================= SAVE PATIENT HISTORY =================
def save_patient_history(row):
    file_exists = os.path.exists("patient_history.csv")
    with open("patient_history.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Patient ID", "Date", "Time",
                "Patient Name", "Age", "Gender",
                "Doctor Name", "Hospital Name",
                "Model Used", "Accuracy",
                "Confidence", "Risk", "Tumor Area"
            ])
        writer.writerow(row)


# ================= PDF REPORT =================
def generate_pdf_report(patient_id, date_time,
                        patient_name, age, gender,
                        doctor_name, hospital_name,
                        model_choice,
                        mri_image=None,
                        confidence=None, risk=None,
                        tumor_area=None):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    heading_color = colors.green
    if risk:
        if "HIGH" in risk:
            heading_color = colors.red
        elif "MODERATE" in risk:
            heading_color = colors.orange

    styles["Title"].textColor = heading_color

    elements.append(Paragraph("AI BRAIN TUMOR DETECTION REPORT", styles["Title"]))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date & Time:</b> {date_time}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Age:</b> {age}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Doctor Name:</b> {doctor_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Hospital Name:</b> {hospital_name}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Model Used:</b> {model_choice}", styles["Normal"]))
    elements.append(
        Paragraph(f"<b>Model Accuracy:</b> {MODEL_ACCURACY.get(model_choice)}", styles["Normal"])
    )
    elements.append(Spacer(1, 12))

    tmp_path = None
    if mri_image is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()
        mri_image.convert("RGB").save(tmp_path)

        elements.append(Paragraph("<b>Uploaded MRI Image:</b>", styles["Normal"]))
        elements.append(Spacer(1, 8))
        elements.append(PDFImage(tmp_path, width=260, height=260))
        elements.append(Spacer(1, 14))

    if confidence is not None:
        elements.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles["Normal"]))
        elements.append(Paragraph(f"<b>Risk Level:</b> {risk}", styles["Normal"]))

    if tumor_area is not None:
        elements.append(Paragraph(f"<b>Tumor Area:</b> {tumor_area}%", styles["Normal"]))

    elements.append(Spacer(1, 18))
    elements.append(Paragraph(
        "<i>This AI-generated report is intended for academic and research purposes only.</i>",
        styles["Italic"]
    ))

    doc.build(elements)
    buffer.seek(0)

    try:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass

    return buffer


# ================= PAGE CONFIG =================
st.set_page_config("AI Brain Tumor Detection", "🧠", layout="wide")


# ================= HEADER =================
st.markdown("""
<div style="background:linear-gradient(90deg,#2563EB,#14B8A6);
padding:24px;border-radius:16px;color:white;">
<h2>🧠 AI-Based Brain Tumor Detection System</h2>
<p>Deep Learning Based Medical Imaging</p>
</div>
""", unsafe_allow_html=True)


# ================= SESSION =================
if "patient_id" not in st.session_state:
    st.session_state.patient_id = generate_patient_id()

if "date_time" not in st.session_state:
    st.session_state.date_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")


# ================= SIDEBAR =================
st.sidebar.markdown("### Patient Details")
patient_name = st.sidebar.text_input("Patient Name *")
age = st.sidebar.number_input("Age *", min_value=1, max_value=120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.markdown("### Doctor Details")
doctor_name = st.sidebar.text_input("Doctor Name *")

st.sidebar.markdown("### Hospital Details")
hospital_name = st.sidebar.text_input("Hospital Name *")

st.sidebar.markdown("### AI Controls")
model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_ACCURACY.keys())
)

file = st.sidebar.file_uploader("Upload MRI Image *", ["jpg", "png", "jpeg"])
analyze = st.sidebar.button("Analyze MRI")


# ================= MAIN =================
if analyze:
    if not patient_name or not doctor_name or not hospital_name or file is None:
        st.warning("⚠️ Please fill all required fields")
    else:
        image = Image.open(file)

        st.success(f"""
**Patient ID:** {st.session_state.patient_id}  
**Date & Time:** {st.session_state.date_time}  
**Patient Name:** {patient_name}  
**Age:** {age}  
**Gender:** {gender}  
**Doctor:** {doctor_name}  
**Hospital:** {hospital_name}  
**Model:** {model_choice}  
**Model Accuracy:** {MODEL_ACCURACY[model_choice]}
""")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input MRI", use_column_width=True)

        with col2:
            confidence = risk = area = None

            if "CNN" in model_choice:
                model = load_model("models/cnn_model.h5")
                confidence = round(cnn_predict(model, preprocess_image(image, (224, 224))) * 100, 2)
                risk = get_risk_level(confidence)
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Risk Level:** {risk}")

            elif "SRGAN" in model_choice:
                model = load_model("models/srgan_model.h5")
                enhanced = srgan_enhance(model, preprocess_image(image, (64, 64)))
                srgan_img = (np.squeeze(enhanced) * 255).astype(np.uint8)
                srgan_img = Image.fromarray(srgan_img)
                srgan_img = srgan_img.resize(image.size)
                st.image(srgan_img, caption="SRGAN Enhanced Image", use_column_width=True)

            else:
                path = "models/unet_model.h5" if model_choice == "U-Net - Tumor Segmentation" else "models/attention_unet_model.h5"
                model = load_model(path)
                raw = np.squeeze(unet_predict(model, preprocess_grayscale(image, (128, 128))))
                overlay, resized = overlay_mask(image, (raw > 0.5).astype(np.uint8))
                area = calculate_tumor_area(resized)
                st.image(overlay)
                st.success(f"Tumor Area: {area}%")

            save_patient_history([
                st.session_state.patient_id,
                st.session_state.date_time.split()[0],
                st.session_state.date_time.split()[1],
                patient_name, age, gender,
                doctor_name, hospital_name,
                model_choice,
                MODEL_ACCURACY[model_choice],
                confidence, risk, area
            ])

            pdf = generate_pdf_report(
                st.session_state.patient_id,
                st.session_state.date_time,
                patient_name, age, gender,
                doctor_name, hospital_name,
                model_choice,
                image, confidence, risk, area
            )

            st.download_button(
                "⬇️ Download PDF Report",
                pdf,
                "Brain_Tumor_Report.pdf",
                "application/pdf"
            )

st.markdown("---")
st.info("For academic and research purposes only")
