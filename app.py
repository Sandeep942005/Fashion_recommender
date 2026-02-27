import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential

# GPU safety
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

st.set_page_config(page_title="AI Fashion Recommender", layout="wide")

features = np.load("features.npy")
filenames = pickle.load(open("filenames.pkl", "rb"))
categories = pickle.load(open("categories.pkl", "rb"))
knn = pickle.load(open("knn_model.pkl", "rb"))
classifier = load_model("category_model.h5")

feature_model = Sequential([
    ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3)),
    GlobalMaxPooling2D()
])
feature_model.trainable = False

def extract_features(img):
    img = img.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img, verbose=0)
    return features[0] / np.linalg.norm(features)

st.title("👗 AI Fashion Recommendation System")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    # Category prediction
    img = image.resize((224,224))
    img = np.expand_dims(np.array(img), axis=0)
    prediction = classifier.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_category = list(set(categories))[predicted_index]

    st.write("Predicted Category:", predicted_category)

    # Similarity search
    query_features = extract_features(image)
    distances, indices = knn.kneighbors([query_features])

    cols = st.columns(5)

    count = 0
    for i, idx in enumerate(indices[0]):
        if categories[idx] == predicted_category and count < 5:
            with cols[count]:
                st.image(Image.open(filenames[idx]))
                similarity = (1 - distances[0][i]) * 100
                st.write(f"{similarity:.2f}% match")
                count += 1