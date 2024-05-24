import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import io
from io import BytesIO
from PIL import Image
from openai import OpenAI
import base64
import time

def generate_content(img=None):
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
    try:
        response = model.generate_content([img])
        
        # Add a delay of 500ms
        time.sleep(0.5)
        
        # Analysis model
        model_ana = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt_ana = f'''if this text is a medical report, X-Ray, MRI (Magnetic Resonance Imaging), CT (Computed Tomography), Ultrasound, PET (Positron Emission Tomography), SPECT (Single Photon Emission Computed Tomography), Mammography, Fluoroscopy, DEXA (Dual-Energy X-ray Absorptiometry) analysis: \n"{response.text}" \n\n give me more explain for the results and sure to say "الرجاء التواصل مع طبيب لمعلومات أكثر" Answer ONLY in Arabic.
        else (The given Text not related to any medical information) then reponse "لا يمكنني المساعدة بذلك"'''
        response_ana = model_ana.generate_content([prompt_ana])
        return response_ana.text, response.text
    except Exception as e:
        st.error("Failed to generate content: {}".format(e))
        return None, None

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_gpt_response(image):
    try:
        base64_image = encode_image(image)
        openai_api = os.getenv('OPENAI_API')
        openai_client = OpenAI(api_key=openai_api)
        MODEL = 'gpt-4o'

        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant in healthcare and medical reports and images"},
                {"role": "user", "content": [
                    {"type": "text", "text": '''If this image is a medical report, X-ray, MRI (Magnetic Resonance Imaging), CT (Computed Tomography), Ultrasound, PET (Positron Emission Tomography), SPECT (Single Photon Emission Computed Tomography), Mammography, Fluoroscopy, or DEXA (Dual-Energy X-ray Absorptiometry), or ECG,  please analyze it and provide a detailed explanation of whether the results are good or not. Explain the reasons behind your assessment, including any specific findings such as "1) 2) etc" and provide more detailed results. Additionally, offer recommendations based on your analysis, Answer ONLY in Arabic.
                else (The given Text not related to any medical information) then reponse "لا يمكنني المساعدة بذلك'''},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
    temperature=0.0,
)
        return response.choices[0].message.content
    except Exception as e:
        st.error("Failed to generate GPT response: {}".format(e))
        return None

def main():
    st.title("🔬🧪MedA👩🏻‍🔬🗜️")
    st.markdown("##### Skip the Wait, Not the Detail: Fast AI Lab Analysis")
    
    # Sidebar with larger font size
    st.sidebar.header("Imaging Types 📋")
    st.sidebar.markdown("""
    <style>
    .font-large {font-size:18px;}
    </style>
    <div class='font-large'>
    - Medical report 📄<br>
    - X-Ray ☠️<br>
    - MRI (Magnetic Resonance Imaging) 🧲<br>
    - CT (Computed Tomography) 🌀<br>
    - Ultrasound 🔊<br>
    - PET (Positron Emission Tomography) 💠<br>
    - SPECT (Single Photon Emission Computed Tomography) 🔅<br>
    - Mammography 🎀<br>
    - Fluoroscopy 💡<br>
    - DEXA (Dual-Energy X-ray Absorptiometry) ⚖️
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### Note")
    st.sidebar.info("Please note that this model's accuracy depends on the accuracy of the given image, estimated at 90%. Always consult a doctor for more detailed and recommended medical advice.")

    img_file_buffer = st.file_uploader("Upload an image (jpg, png, jpeg):", type=["jpg", "png", "jpeg"])
    img = None
    if img_file_buffer is not None:
        img = Image.open(io.BytesIO(img_file_buffer.getvalue()))

    if st.button("Analysis"):
        if img:
            processed_text_ahmad, raw_response_ahmad = generate_content(img)
            processed_text_mohammad = generate_gpt_response(img)
            st.markdown("### Doctor Ahmad")
            st.markdown(f"<div style='direction: rtl; text-align: right;'> {processed_text_ahmad}</div>", unsafe_allow_html=True)
            st.markdown("### Doctor Mohammad")
            st.markdown(f"<div style='direction: rtl; text-align: right;'> {processed_text_mohammad}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
