# Лескін Максим

import os
import urllib.parse
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
HF_TOKEN = os.getenv("HF_TOKEN")

def generate_image(prompt):
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed=42"

        response = requests.get(url, timeout=45)

        if response.status_code == 200:
            return response.content, None
        else:
            return None, f"Сервіс тимчасово недоступний: ({response.status_code})"
    except requests.exceptions.Timeout:
        return None,"Time-Out"
    except Exception as e:
        return None, f"Помилка підключення: {e}"


st.title("AI Creative Studio")
st.write("Опиши чітко, яке саме зображення ти хочеш отримати, обери, за бажанням, " \
"стиль у якому штучний інтелект сторить фото. Натисни кнопку 'Завантажити', щоб отримати згенероване фото на свій пристрій.")

prompt = st.text_area("Ваш промпт:")

STYLES = {
    "Без стилю": "",
    "Реалістичний": "high detail, photorealistic, natural lighting",
    "Акварель": "watercolor painting style, soft edges, pastel colors",
    "Мультяшний": "3D animated movie style, vibrant colors, friendly look",
    "Піксель-Арт": "pixel art style, 16-bit retro game aesthetic"
}

styles_choice = st.selectbox("Стилі:", list(STYLES.keys()))

if st.button("Сгенерувати зображення") and prompt:
    final_prompt = prompt + STYLES[styles_choice]

    with st.spinner("AI створює зображення"):
        image_bytes, error = generate_image(final_prompt)

    if error:
        st.error(f"Не вдалося сгенерувати зображення: {error}")
    elif image_bytes:
        st.image(image_bytes, caption=f"Стиль: {styles_choice}", use_container_width=True)
        st.session_state["last_image"] = image_bytes

if "last_image" in st.session_state:
    st.download_button(
        label="Завантажити згенероване фото",
        data=st.session_state["last_image"],
        file_name="ai_result.png",
        mime="image/png"
    )

st.title("Photo Editor Studio")
st.write("Обери будь-який із запропонованих у сайд барі фільтрів і застосуй їх на згенерованому зображенні." \
" Натиснувши кнопку 'Завантажити', ви можете отримати відредаговане фото на свій пристрій.")

filter_option = st.sidebar.selectbox("Оберіть потрібний ефект:", ["Оригінал",
                                                                    "Віддзеркалення",
                                                                    "Налаштування яскравості",
                                                                    "Налаштування контрасту",
                                                                    "Колірний сплеск"])

if "last_image" in st.session_state:
    image_created = Image.open(BytesIO(st.session_state["last_image"]))
    img_array = np.array(image_created)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Згенероване фото:")
        st.image(img_array, use_container_width=True)

    edited_img = img_array.copy()
    
    if filter_option == "Віддзеркалення":
        edited_img = cv2.flip(img_array, 1)
    elif filter_option == "Налаштування яскравості":
        brightness_val = st.sidebar.slider("Рівень яскравості", -100, 100, 0)
        edited_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=brightness_val)
    elif filter_option == "Налаштування контрасту":
        contrast_val = st.sidebar.slider("Рівень контрасту", 0.5, 3.0, 1.0, step=0.1)
        edited_img = cv2.convertScaleAbs(img_array, alpha=contrast_val, beta=0)
    elif filter_option == "Колірний сплеск":
        chosen_color = st.sidebar.selectbox("Оберіть колір сплеску:", ["Червоний", "Зелений", "Синій"])

        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if chosen_color == "Червоний":
            low_red1 = np.array([0, 50, 50])
            high_red1 = np.array([10, 255, 255])
            low_red2 = np.array([170, 50, 50])
            high_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv_img, low_red1, high_red1)
            mask2 = cv2.inRange(hsv_img, low_red2, high_red2)
            mask = np.where(mask1 > 0, mask1, mask2)

            edited_img = np.where(mask[:, :, None] > 0, img_array, gray)
        elif chosen_color == "Зелений":
            low_green = np.array([35, 50, 50])
            high_green = np.array([85, 255, 255])

            mask = cv2.inRange(hsv_img, low_green, high_green)

            edited_img = np.where(mask[:, :, None] > 0, img_array, gray)
        elif chosen_color == "Синій":
            low_blue = np.array([100, 50, 50])
            high_blue = np.array([140, 255, 255])

            mask = cv2.inRange(hsv_img, low_blue, high_blue)

            edited_img = np.where(mask[:, :, None] > 0, img_array, gray)
 
    with col2: 
        st.subheader("Відредаговане фото:")
        st.image(edited_img, use_container_width=True)
        
        edited_pil = Image.fromarray(edited_img) # Перетворюємо масив у зображення PIL

        output = BytesIO() # Створюємо буфер для збереження зображення
        edited_pil.save(output, format="PNG") # Зберігаємо зображення у форматі PNG
        output.seek(0) # Повертаємося на початок буферу

        st.download_button(
            label="Завантажити відредаговане фото",
            data=output,
            file_name="ai_result_edited.png",
            mime="image/png"
        )
