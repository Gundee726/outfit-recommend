import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import tensorflow as tf

# Load the ResNet50 model (pretrained on ImageNet, without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build the final model by adding a global max pooling layer
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features (same as your existing function)
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Load embeddings and filenames
embeddings = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Precision at k
def precision_at_k(query_image_path, model, embeddings, filenames, top_k=5):
    query_features = extract_features(query_image_path, model)
    similarities = cosine_similarity([query_features], embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_count = 0
    for i, index in enumerate(top_k_indices):
        retrieved_image_path = filenames[index]
        if is_relevant(query_image_path, retrieved_image_path):  # Custom relevance check
            relevant_count += 1

    precision = relevant_count / top_k
    print(f"Precision at {top_k}: {precision}")
    return precision

# Example relevance checking function (based on category or folder structure)
def is_relevant(query_image_path, retrieved_image_path):
    query_label = query_image_path.split('/')[-2]  # Example: Get folder/category name
    retrieved_label = retrieved_image_path.split('/')[-2]
    return query_label == retrieved_label

# Mean Average Precision (mAP)
def mean_average_precision(query_images, model, embeddings, filenames, top_k=5):
    average_precisions = []
    for query_image in query_images:
        precision = precision_at_k(query_image, model, embeddings, filenames, top_k)
        average_precisions.append(precision)
    
    mAP = np.mean(average_precisions)
    print(f"Mean Average Precision: {mAP}")
    return mAP

if __name__ == '__main__':
    # List of query images (add actual image paths from your dataset)
    query_images = ['./images/1526.jpg', './images/1555.jpg']

    # Compute and print Mean Average Precision
    mean_average_precision(query_images, model, embeddings, filenames, top_k=5)
