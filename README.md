# Interfaz-Proyecto-TFM
Codido de desarrollo de la interfaz
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import requests
from streamlit_lottie import st_lottie

#Configuración de la interfaz
st.set_page_config(page_title = "Clasificador de imágenes de lesiones cutáneas", page_icon= "U+1FA7A", layout="wide" )  # Título de la interfaz

#Dirección del correo
dir_email= "contactomail@gmail.com"

#Lottie Animacion
archivo_lottie ="https://lottie.host/02ae12d6-2d0d-4c07-9e83-09185ced47ff/XvZmjg3hjo.json"

#Funcion para cargar lottie. 
def cargalottie(url):
    r = requests.get(url)
    if r.status_code !=200: #si el request devuelve 200 es que todo ha ido bien. 
        return None
    return r.json()
    
#definimos variable lottie para cargar la funcion
lottie=cargalottie(archivo_lottie)

#introducción

with st.container():
    st.header("¿Quieres conocer si tienes posibilidades de padecer una patología cutánea?")
    st.title("Esto es un clasificador que te ayudará a identificar si una lesión cutánea es de tipo melanoma (mel), carcinoma basocelular (bcc) o queratosis benigna (bkl) basándose en la imagen que subas.")
    st.write("Aquí podrás realizar una primera toma de contacto, así como guiarte en un posible diagnóstico")
    st.warning("Recuerda que esto es un prototipo. Acude siempre a tu médico de confianza")

# Qué es la app?
with st.container():
    columna_texto, columna_animacion=st.columns(2)
    with columna_texto:
        st.header("¿Qué hace realmente la inferfaz?")
        st.write(
            """
            Esta infertaz muestra los resultados de un entrenamiento realizado durante el Proyecto de Fin de Máster, 

            Se ha realizado un entrenamiento para predecir 3 clases de lesiones cutáneas: 

            - Melanoma
            - Carcinoma basocelular
            - Queratosis benigna

            ***Recuerda siempre recurrir a atención sanitaria, ya que esto es un prototipo y tiene mucho potencial de mejoría***. 
            """
        )
    with columna_animacion: 
        st_lottie(lottie, height = 400)

#Funcionalidad de la app 
with st.container():
    # Cargar el modelo
    model = load_model('mejor_modelo_entrenamiento_vgg16.keras')

    # Subir la imagen
    uploaded_file = st.file_uploader("Seleccione la imagen a predecir (Formato: JPG)", type=["jpg"], )
    
    if uploaded_file is not None:
        # Mostrar la imagen cargada
        st.balloons()
        st.image(uploaded_file, caption='Imagen subida.', use_column_width=True)
        st.write("")
        st.write("Clasificando...")
    
        # Cargar y preprocesar la imagen
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Normalización adecuada
    
        # Realizar la predicción
        prediction = model.predict(img_array)
    
        # Obtener la clase con mayor probabilidad
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
    
        # Crear diccionario de clases
        class_labels = {0: 'mel', 1: 'bcc', 2: 'bkl'}  # Ejemplo de nombres de clases
    
        # Verificar si el índice predicho está en el diccionario de etiquetas
        if predicted_class_idx in class_labels:
            predicted_class_label = class_labels[predicted_class_idx]
            predicted_probability = prediction[0][predicted_class_idx] * 100  # Convertir a porcentaje
    
            # Mostrar el resultado en Streamlit
            st.write(f"**Clase predicha:** {predicted_class_label}")
            st.write(f"**Confianza:** {predicted_probability:.2f}%")
            st.warning("Recuerda siempre asistir al centro sanitario en cualquier circunstancia de duas en el diagnóstico")
        else:
            st.write(f"Índice predicho {predicted_class_idx} no está en class_labels.")

#Contacto 

with st.container():
    st.write("---")
    st.header("Ponte en contacto con nosotros!")
    st.write("##")
    forma_contacto = f"""
    <form action="https://formsubmit.co/{dir_email}" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Tu nombre" required>
        <input type="email" name="email" placeholder="Tu email" required>
        <textarea name="message" placeholder="Tu mensaje aquí" required></textarea>
        <button type="submit">Enviar</button>
    </form>
    """
    st.markdown(forma_contacto, unsafe_allow_html=True)

#estilo interfaz

# CSS personalizado para agregar fondo, colores y estilos
st.markdown(
    """
    <style>
        /* Fondo azul */
        .stApp {
            background-color: #f0f0f0;  /* Color azul de fondo */
        }

        /* Colores y estilo de texto */
        h1, h2, h3 {
            color: #FF5733;
            font-family: Arial, sans-serif;
        }

        /* Texto justificado y color */
        p, li, label, .stMarkdown p {
            color: #1C2833;
            text-align: justify; /* Justificar texto */
        }

        /* Botones */
        .stButton button {
            color: white;
            background-color: #2980B9;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)
