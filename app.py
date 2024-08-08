import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


def get_gradcam_heatmap(model, img_array, class_index):
    last_conv_layer = model.get_layer('block5_conv3')
    heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def get_img_array(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_top_predictions(preds):
    previsoes = decode_predictions(preds, top=5)[0]
    previsao = [f"Previsão: {i[1].ljust(20)} Prob: {round(i[2] * 100, ndigits=2)}%" for i in previsoes]
    return previsao

model = VGG16(weights='imagenet')

st.title("VGG16 e Grad-Cam - Dogs vs Cats")

st.sidebar.header("Escolha a Imagem")
animal = st.sidebar.selectbox('Escolha o tipo de animal:', ['Cachorro', 'Gato'])
valor = st.sidebar.slider('Escolha uma imagem:', min_value=1, max_value=78, value=1)

dic_animal = {"Cachorro":"dog","Gato":"cat"}
an_path = dic_animal[animal]

img_path = f'{an_path}s/{an_path}.{valor}.jpg'



img = image.load_img(img_path, target_size=(224, 224))
img_array = get_img_array(img)
preds = model.predict(img_array)
class_index = np.argmax(preds[0])

heatmap = get_gradcam_heatmap(model, img_array, class_index)

st.subheader('Imagem e Mapa de Calor Grad-Cam')

col1, col2 = st.columns(2)

with col1:
    st.image(img, caption='Imagem Original', use_column_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_array[0] / 255.)
    ax.imshow(heatmap, cmap='viridis', alpha=0.5)
    st.pyplot(fig)

st.subheader('Previsão')
top_prevs = get_top_predictions(preds)
st.write(top_prevs)
