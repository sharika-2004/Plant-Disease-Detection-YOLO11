import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import cv2
from ultralytics import YOLO
import time
import numpy as np

# Load the model
model = YOLO("Plant_Disease_Classification.pt")

# Define full class names list
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Sidebar
st.sidebar.title("🌱 Plant Disease Classifier")
st.sidebar.markdown("Upload an image of a plant leaf to get disease classification results!")

# Page title
st.title("AI-Powered Plant Disease Detection 🌿")

components.html("""
    <style>
        .upload-box {
            border: 2px dashed #aaa;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            background-color: #f5f5f5;
            position: relative;
        }
        .upload-box:hover {
            background-color: #eee;
        }
        .upload-icon {
            font-size: 48px;
        }
        .upload-instruction {
            font-size: 18px;
            color: #444;
        }
    </style>
""", height=0)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing... Please wait."):
        time.sleep(1)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(img_cv)
        probs = results[0].probs

        if probs:
            top_idx = probs.top1
            confidence = probs.data[top_idx].item()
            class_name = class_names[top_idx]

            st.success(f"🧠 **Prediction**: {class_name}")
            st.markdown(f"🎯 **Confidence**: {confidence:.2%}")

            # Disease recommendations
            disease_recommendations = {
                "Apple___Apple_scab": "Apply a 5% urea solution to fallen leaves in autumn to accelerate decomposition and reduce overwintering spores. Maintain orchard sanitation by removing fallen leaves and pruning infected branches.",
                "Apple___Black_rot": "Implement sanitation by removing mummified fruits and pruning infected branches. Apply fungicides like captan or sulfur during the growing season as needed.",
                "Apple___Cedar_apple_rust": "Use sterol-inhibiting fungicides preventatively. Remove nearby juniper hosts to disrupt the disease cycle.",
                "Apple___healthy": "Maintain proper pruning, sanitation, and monitor regularly to prevent disease onset.",
                "Blueberry___healthy": "Ensure good air circulation, proper spacing, and regular monitoring to maintain plant health.",
                "Cherry_(including_sour)___Powdery_mildew": "Spray fungicides soon after petal fall and again 2–3 weeks later if needed. Rotate fungicides to prevent resistance development.",
                "Cherry_(including_sour)___healthy": "Prune for air circulation, avoid overhead watering, and monitor for early signs of disease.",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply foliar fungicides during the growing season, considering economic thresholds. Use resistant hybrids and manage crop residue.",
                "Corn_(maize)___Common_rust_": "Monitor fields regularly; fungicide application is rarely needed as most hybrids are resistant.",
                "Corn_(maize)___Northern_Leaf_Blight": "Plant resistant hybrids, practice crop rotation, and apply fungicides during early disease stages if necessary.",
                "Corn_(maize)___healthy": "Employ crop rotation, use resistant varieties, and monitor for early disease symptoms.",
                "Grape___Black_rot": "Apply fungicides starting at bud break and continue at 7–10 day intervals during wet weather. Ensure good air circulation and sunlight penetration.",
                "Grape___Esca_(Black_Measles)": "No effective chemical control; manage by removing infected wood and improving vineyard sanitation.",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides as needed and ensure proper canopy management to reduce humidity.",
                "Grape___healthy": "Maintain good air circulation, proper pruning, and regular monitoring to prevent disease.",
                "Orange___Haunglongbing_(Citrus_greening)": "There is no cure; manage by controlling the Asian citrus psyllid vector and removing infected trees promptly.",
                "Peach___Bacterial_spot": "Apply copper-based bactericides and oxytetracycline preventatively. Implement cultural practices to reduce disease spread.",
                "Peach___healthy": "Ensure proper pruning, avoid overhead irrigation, and monitor for early signs of disease.",
                "Pepper,_bell___Bacterial_spot": "Use disease-free seeds, apply copper-based bactericides, and avoid overhead watering to minimize spread.",
                "Pepper,_bell___healthy": "Practice crop rotation, maintain field sanitation, and monitor plants regularly.",
                "Potato___Early_blight": "Apply foliar fungicides containing fluopyram or similar active ingredients at the first sign of disease.",
                "Potato___Late_blight": "Use foliar fungicides on a regular schedule during conducive weather conditions. Eliminate cull piles and volunteer potatoes.",
                "Potato___healthy": "Plant certified disease-free seed potatoes, practice crop rotation, and monitor for early disease symptoms.",
                "Raspberry___healthy": "Ensure good air circulation, proper pruning, and monitor for common raspberry diseases.",
                "Soybean___healthy": "Rotate crops, use resistant varieties, and monitor fields regularly for disease symptoms.",
                "Squash___Powdery_mildew": "Apply fungicides like sulfur or neem oil at the first sign of disease. Ensure good air circulation and avoid overhead watering.",
                "Strawberry___Leaf_scorch": "Remove infected leaves, apply appropriate fungicides, and ensure proper spacing for air circulation.",
                "Strawberry___healthy": "Use disease-free plants, practice crop rotation, and maintain proper field sanitation.",
                "Tomato___Bacterial_spot": "Use disease-free seeds, apply copper-based bactericides, and avoid overhead irrigation.",
                "Tomato___Early_blight": "Apply fungicides containing chlorothalonil or copper at the first sign of disease. Remove infected leaves and ensure proper spacing.",
                "Tomato___Late_blight": "Apply fungicides regularly during cool, wet conditions. Remove and destroy infected plants promptly.",
                "Tomato___Leaf_Mold": "Improve air circulation through pruning, avoid overhead watering, and apply appropriate fungicides as needed.",
                "Tomato___Septoria_leaf_spot": "Remove infected leaves, apply fungicides, and practice crop rotation to prevent recurrence.",
                "Tomato___Spider_mites Two-spotted_spider_mite": "Introduce natural predators like ladybugs, apply miticides if necessary, and maintain adequate humidity levels.",
                "Tomato___Target_Spot": "Apply recommended fungicides, remove infected plant debris, and ensure proper spacing for air circulation.",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly populations, use resistant varieties, and remove infected plants promptly.",
                "Tomato___Tomato_mosaic_virus": "Use virus-free seeds, disinfect tools regularly, and remove infected plants to prevent spread.",
                "Tomato___healthy": "Practice crop rotation, use resistant varieties, and monitor plants regularly for early disease detection."
            }
            st.markdown("🔧 **Recommended actions:**")
            if 'healthy' not in class_name:
                recommendation = disease_recommendations.get(class_name, "No specific recommendation available.")
                st.info(f"🌾 {recommendation}")
            else:
                st.info("✅ No action needed. Plant is healthy.")
        else:
            st.error("⚠️ Unable to classify this image.")
else:
    st.info("Please upload a plant leaf image to get started.")
