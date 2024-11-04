#se cargan las librerías necesarias
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
import gdown
import os

# Configuración de la interfaz
st.set_page_config(page_title = "Clasificador de imágenes de lesiones cutáneas", page_icon= "U+1FA7A", layout="wide" )  # Título de la interfaz

#Dirección del correo
dir_email= "contactomail@gmail.com"

# URLs de Google Drive para descargar los modelos
url_vgg16 = 'https://drive.google.com/file/d/1F61T6iVumbcvzOvSsBegvSi3PSztTJZR/view?usp=sharing'
url_resnet = 'https://drive.google.com/file/d/1EkACKYU-FpVGvIapjrYi46Pgm0UOIofJ/view?usp=sharing' 

# Rutas locales para guardar los modelos descargados
model_path_vgg16 = 'Modelo_VGG16_VBM.keras'
model_path_resnet = 'Modelo_ResNet_VBM.keras'
# Función para descargar modelo desde Google Drive si no está localmente
def download_model(url, model_path):
    if not os.path.exists(model_path):
        st.write(f"Descargando el modelo {model_path}...")
        gdown.download(url, model_path, quiet=False)
        st.write(f"Modelo {model_path} descargado correctamente.")

# Descargar los modelos si no existen localmente
download_model(url_vgg16, model_path_vgg16)
download_model(url_resnet, model_path_resnet)

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
    st.title("Clasificador de patologías cutáneas")
    st.write("Aquí podrás realizar una primera toma de contacto, así como guiarte en un posible diagnóstico")
    st.warning("Recuerda que esto es un prototipo. Acude siempre a tu médico de confianza")

# Qué es la app?
with st.container():
    columna_texto, columna_animacion=st.columns(2) #se crean dos columnas para la sección
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

# Funcionalidad de la app 
    
#Este apartado es el corazón de la interfaz. Se centra en ofrecer una herramienta accesible y efectiva para la clasificación de lesiones cutáneas. Esta funcionalidad permite a los usuarios seleccionar entre diferentes modelos de aprendizaje automático para obtener diagnósticos preliminares sobre imágenes de lesiones. El uso de múltiples modelos, como 'Modelo_VGG16_VBM.keras' y 'Modelo_ResNet50_VBM.keras', brinda a los usuarios la oportunidad de comparar los resultados y elegir el que mejor se ajuste a sus necesidades.
            
#La aplicación facilita la carga de varias imágenes a la vez, lo que optimiza el flujo de trabajo y mejora la experiencia del usuario. Una vez que se suben las imágenes, el proceso se lleva a cabo el proceso de predicción. Las imágenes son preprocesadas para cumplir con los requisitos del modelo, incluyendo la redimensión y la normalización. Esto asegura que las predicciones sean precisas y consistentes.
            
#Al mostrar los resultados, la app organiza las imágenes y sus respectivas clasificaciones en dos columnas, lo que maximiza el uso del espacio en pantalla. Cada imagen es acompañada por su clase predicha y la confianza del modelo en su evaluación, presentada como un porcentaje. Esto proporciona al usuario una comprensión clara y rápida de los resultados.
            
#La funcionalidad enfatiza la importancia de la atención médica profesional, recordando que las predicciones son solo un punto de partida para un diagnóstico más exhaustivo.
    
with st.container():
    
    
    modelo_opciones = ['VGG16', 'ResNet']
    modelo_seleccionado = st.selectbox("Selecciona el modelo para la clasificación:", modelo_opciones)
    
    # Cargar el modelo seleccionado
    if modelo_seleccionado == 'VGG16':
        model = load_model(model_path_vgg16)
    else:
        model = load_model(model_path_resnet)
    
    # Diccionario de clases
    class_labels = {0: 'Melanoma', 1: 'Carcinoma basocelular', 2: 'Queratosis benigna'}
    
    # Subir y clasificar imágenes
    uploaded_files = st.file_uploader("Seleccione las imágenes a predecir (Formato: JPG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Clasificando...")
    
        cols = st.columns(2)  # Para organizar visualmente las imágenes
        for i, uploaded_file in enumerate(uploaded_files):
            # Cargar y preprocesar la imagen
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
    
            # Realizar la predicción
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_class_label = class_labels.get(predicted_class_idx, "Clase desconocida")
            predicted_probability = prediction[0][predicted_class_idx] * 100
    
            # Mostrar imagen y predicción
            cols[i % 2].image(uploaded_file, caption='Imagen subida.', use_column_width=True)
            cols[i % 2].write(f"**Clase predicha:** {predicted_class_label}")
            cols[i % 2].write(f"**Confianza:** {predicted_probability:.2f}%")



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
    
# Estilo interfaz
# CSS personalizado para agregar fondo, colores y estilos
st.markdown(
    """
    <style>
        /* Fondo gris claro */
        .stApp {
            background-color: #f0f0f0;  /* Color gris claro de fondo */
        }

        /* Colores y estilo de texto */
        h1, h2, h3 {
            color: #333;
            font-family: Arial, sans-serif;
            font-size: 1.5rem;
        }

        /* Texto justificado y color */
        p, li, label, .stMarkdown p {
            color: #555;
            text-align: justify; /* Justificar texto */
        }

        /* Botones */
        .stButton button {
            color: white;
            background-color: #007BFF;
            font-weight: bold;
            border-radius: 5px; /* Esquinas redondeadas */
            padding: 10px 20px; /* Espaciado interno */
        }
    </style>
    """,
    unsafe_allow_html=True
)
