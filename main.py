#Data preprocessing
import os

import cv2
import numpy as np
from scipy import fftpack
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import SVC

# Path to dataset
train_dir = './dataset/chest_xray/train/'
test_dir = './dataset/chest_xray/test/'

# Parameters
IMG_SIZE = 128  # Image size to resize

# Function to load and preprocess images
def load_images_from_folder(folder, img_size):
    images = []
    labels = []
    for label in ['PNEUMONIA', 'NORMAL']:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(0 if label == 'NORMAL' else 1)  # 0 = Normal, 1 = Pneumonia
    return np.array(images), np.array(labels)

# Load train and test data
X_train, y_train = load_images_from_folder(train_dir, IMG_SIZE)
X_test, y_test = load_images_from_folder(test_dir, IMG_SIZE)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0


# Example function to extract intensity and texture features
def extract_features(images):
    features = []
    for img in images:
        # Convert the image to uint8 before calculating GLCM
        img_uint8 = (img * 255).astype(np.uint8)

        # Intensity-based features (mean, variance)
        mean_intensity = np.mean(img)
        variance_intensity = np.var(img)

        # Texture features using GLCM (Gray Level Co-occurrence Matrix)
        # Update the function name here
        glcm = graycomatrix(img_uint8, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Frequency domain features (Fourier Transform)
        f_transform = fftpack.fft2(img)
        f_transform_magnitude = np.abs(f_transform)
        freq_mean = np.mean(f_transform_magnitude)
        freq_var = np.var(f_transform_magnitude)

        # Shape features can be added as needed (e.g., using contours)

        # Combine all features
        feature_vector = [mean_intensity, variance_intensity, contrast, correlation, energy, homogeneity, freq_mean, freq_var]
        features.append(feature_vector)

    return np.array(features)

# Extract features from both training and test sets
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)


# Feature selection: Filter-based (e.g., ANOVA F-test)
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_features, y_train)
X_test_selected = selector.transform(X_test_features)

# Feature selection: Wrapper-based (e.g., RFE with Random Forest)
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train_features, y_train)
X_test_rfe = rfe.transform(X_test_features)

# Example: SVM classifier
svm = SVC()
svm.fit(X_train_selected, y_train)
y_pred_svm = svm.predict(X_test_selected)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Example: Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train_selected, y_train)
y_pred_rf = rf.predict(X_test_selected)
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:\n", cm)

# AUC score
auc = roc_auc_score(y_test, y_pred_svm)
print("ROC-AUC Score:", auc)



