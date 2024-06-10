from matplotlib import pyplot as plt
from skimage import feature
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
from skimage.util import random_noise


def directory_check():
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    target_dir = '../resized_dataset/train'
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} found.")
    else:
        print(f"Directory {target_dir} does not exist. Please check the path.")

#directory_checks()

# Function to load images from a given directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    images.append(img)
                    labels.append(label)
    return images, labels


resized_dataset_folder = '../resized_dataset/'


# Loading training data
training_images_list, training_labels = load_images_from_folder(os.path.join(resized_dataset_folder, 'train'))
training_images = np.array(training_images_list)

# Path to save cache
features_path = 'features.npy'
labels_path = 'labels.npy'
rf_model_path = 'random_forest_model.joblib'
test_normalise_path = 'test_normalise.npy'

def normalising_image(inp_image):
    print("Normalising Training Images...")
    return inp_image/255.0

def extract_features(normalised_inp_images):
    print("Extracting Training Features...")
    hog_features = []
    for image in normalised_inp_images:
        hog_feat = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=False, channel_axis=-1)
        hog_features.append(hog_feat)
    return np.array(hog_features)

if os.path.exists(features_path) and os.path.exists(labels_path):
    features = np.load(features_path)
    training_labels = np.load(labels_path)
else:
    # Extract features and labels
    normalised_training_images = normalising_image(training_images)
    features = extract_features(normalised_training_images)
    np.save(features_path, features)
    np.save(labels_path, training_labels)


#Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(features, training_labels, test_size=0.2, random_state=42)
print("train-val split completed")


def training_random_forest(n_estimators, random_state = 42):
    rf_classifier = RandomForestClassifier(n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    predictions = rf_classifier.predict(X_val)
    print("Accuracy validation:", accuracy_score(y_val, predictions)*100)
    print("Classification Report:", classification_report(y_val, predictions))
    
    joblib.dump(rf_classifier, rf_model_path)
    print(f"Model saved to {rf_model_path}")
    
if os.path.exists((rf_model_path)):
    rf_classifier = joblib.load(rf_model_path)
    print("Random forest model loaded")
else:
    print("Training random forest model")
    rf_classifier=training_random_forest(500)
    
#Loading Test images 
test_images_list, test_labels= load_images_from_folder(os.path.join(resized_dataset_folder, 'test'))
test_images_unNormalised = np.array(test_images_list)
test_image = test_images_unNormalised

    
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16

def apply_gaussian_blur(image, num_convolutions):
    blurred_image = image.copy()
    for _ in range(num_convolutions):
        blurred_image = cv2.filter2D(blurred_image, -1, gaussian_kernel)
    return blurred_image


#Perbutation 1 
noise = int(input("noise?"))
if noise ==1:
    def add_gaussian_noise_to_color_images(images, std_dev):
        perturbed_images = []
        for image in images:
            noise = np.random.normal(0, std_dev, image.shape)
            perturbed_image = image + noise
            perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
            perturbed_images.append(perturbed_image)
        return np.array(perturbed_images)
    
    standard_deviations = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    accuracy_scores_perb_one = []

    for std_dev in standard_deviations:
        # Apply Gaussian noise to test images
        noisy_final_images_test = add_gaussian_noise_to_color_images(test_image, std_dev)
        
        # Extract features from noisy test images
        noisy_features_test = extract_features(noisy_final_images_test)
        
        # Predict on the noisy test set
        predictions_noisy_test = rf_classifier.predict(noisy_features_test)
        
        # Calculate and store accuracy
        accuracy = accuracy_score(test_labels, predictions_noisy_test)*100
        accuracy_scores_perb_one.append(accuracy)
        print(f"Standard Deviation: {std_dev}, Accuracy: {accuracy}")


#Perbutation 2
blur = input("Perbutation 2 - Test Gaussian Blurring? (1 or 0): ")
blur = int(blur)
# blurred_images_test = [apply_gaussian_blur(image, num_times) for image in test_image for num_times in range(10)]
if blur == 1:
    accuracy_scores_blur = []

    for num_convolutions in range(10):
        blurred_final_images_test = [apply_gaussian_blur(image, num_convolutions) for image in test_image]
        print("Image Blurred")
        blurred_features_test = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in blurred_final_images_test])
        print("Test features Extracted")
        predictions_blurred_test = rf_classifier.predict(blurred_features_test)
    
        
        
    num_convolutions_list = list(range(10)) 

    plt.figure(figsize=(10, 6))
    plt.plot(num_convolutions_list, accuracy_scores_blur, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy - Random Forest vs. Number of Gaussian Blurring Convolutions')
    plt.xlabel('Number of Convolutions')
    plt.ylabel('Classification Accuracy - Random Forest')
    plt.xticks(num_convolutions_list)
    plt.ylim(0, 50)  

    plt.grid(True)
    plt.show()
    
#Perbutation 3
#Image contrast increase
contrast_increase = input("Perbutation 3 - Do you want to implement image contrast increase?(1/0): ")
contrast_increase = int(contrast_increase)

if contrast_increase == 1:
    contrast_multipliers = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
    accuracy_scores_contrast_increase = []
    for multiplier in contrast_multipliers:
        adjusted_images = [np.clip(image * multiplier, 0, 255).astype(np.uint8) for image in test_image]
        print("Contrast Increased")
        blurred_features_test_image_contrast_increase = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in adjusted_images])
        print("Image Blurred")
        predictions_image_contrast_increase = rf_classifier.predict(blurred_features_test_image_contrast_increase)
        accuracy = accuracy_score(test_labels, predictions_image_contrast_increase)*100
        accuracy_scores_contrast_increase.append(accuracy)
        print(f"Contrast Multiplier: {multiplier}, Accuracy: {accuracy}")

    #plotting
    num_contrasts = list(range(len(contrast_multipliers)))

    plt.figure(figsize=(10, 6))
    plt.plot(num_contrasts, accuracy_scores_contrast_increase, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy - Random Forest vs. Contrast Multipliers')
    plt.xlabel('Contrast Multiplier Index')
    plt.ylabel('Classification Accuracy - Random Forest')
    plt.xticks(num_contrasts, labels=[str(m) for m in contrast_multipliers])
    plt.ylim(0, 50)  
    plt.grid(True)
    plt.show()
       

#Perbutation 4     
#Image contrast decrease
contrast_decrease = input("Perbutation 4 - Do you want to implement image contrast decrease?(1/0): ")
contrast_decrease = int(contrast_decrease)

if contrast_decrease == 1:
    contrast_multipliers = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    accuracy_scores_contrast_decrease = []
    for multiplier in contrast_multipliers:
        adjusted_images = [np.clip(image * multiplier, 0, 255).astype(np.uint8) for image in test_image]
        print("Contrast Decreased")
        blurred_features_test_image_contrast_decrease = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in adjusted_images])
        print("Image Blurred")
        predictions_image_contrast_decrease = rf_classifier.predict(blurred_features_test_image_contrast_decrease)
        accuracy = accuracy_score(test_labels, predictions_image_contrast_decrease)*100
        accuracy_scores_contrast_decrease.append(accuracy)
        print(f"Contrast Multiplier: {multiplier}, Accuracy: {accuracy}")

    #plotting
    num_contrasts = list(range(len(contrast_multipliers)))

    plt.figure(figsize=(10, 6))
    plt.plot(num_contrasts, accuracy_scores_contrast_decrease, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy - Random Forest vs. Contrast Multipliers(decrease)')
    plt.xlabel('Contrast Multiplier Index')
    plt.ylabel('Classification Accuracy - Random Forest')
    plt.xticks(num_contrasts, labels=[str(m) for m in contrast_multipliers])
    plt.ylim(0, 50)  

    plt.grid(True)
    plt.show()
    
    
#PERBUTATION 5
brightness_increase = input("Perbutation 5 - Do you want to implement image brightness increase?(1/0): ")
brightness_increase = int(brightness_increase)

if brightness_increase == 1:
    brightness_additions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    accuracy_scores_brightness_increase = []
    for addition in brightness_additions:
        adjusted_images = [np.clip(image + addition, 0, 255).astype(np.uint8) for image in test_image]
        print("Brightness Increased")
        
        blurred_features_test_image_brightness_increase = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in adjusted_images])
        print("Feature Extraction Completed")  
        predictions_image_brightness_increase = rf_classifier.predict(blurred_features_test_image_brightness_increase)
        accuracy = accuracy_score(test_labels, predictions_image_brightness_increase) * 100
        accuracy_scores_brightness_increase.append(accuracy)
        print(f"Brightness Increase Value: {addition}, Accuracy: {accuracy}")

    # Plotting
    num_brightness_levels = list(range(len(brightness_additions)))

    plt.figure(figsize=(10, 6))
    plt.plot(num_brightness_levels, accuracy_scores_brightness_increase, marker='o', linestyle='-', color='r')
    plt.title('Classification Accuracy - Random Forest vs. Brightness Increase Levels')
    plt.xlabel('Brightness Increase Value Index')
    plt.ylabel('Classification Accuracy - Random Forest')
    plt.xticks(num_brightness_levels, labels=[str(a) for a in brightness_additions])
    plt.ylim(0, 50)  
    plt.grid(True)
    plt.show()
        
        
#Perbutation 6
brightness_decrease = input("Permutation 6 - Do you want to implement image brightness decrease?(1/0): ")
brightness_decrease = int(brightness_decrease)

if brightness_decrease == 1:
    brightness_subtractions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    accuracy_scores_brightness_decrease = []
    for subtraction in brightness_subtractions:
        adjusted_images = [np.clip(image - subtraction, 0, 255).astype(np.uint8) for image in test_image]
        print("Brightness Decreased")
        
        # Proceding with feature extraction and prediction
        features_test_image_brightness_decrease = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in adjusted_images])
        print("Feature Extraction Completed")  
        predictions_image_brightness_decrease = rf_classifier.predict(features_test_image_brightness_decrease)
        accuracy = accuracy_score(test_labels, predictions_image_brightness_decrease) * 100
        accuracy_scores_brightness_decrease.append(accuracy)
        print(f"Brightness Decrease Value: {subtraction}, Accuracy: {accuracy}")

    num_brightness_levels = list(range(len(brightness_subtractions)))

    plt.figure(figsize=(10, 6))
    plt.plot(num_brightness_levels, accuracy_scores_brightness_decrease, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy - Random Forest vs. Brightness Decrease Levels')
    plt.xlabel('Brightness Decrease Value Index')
    plt.ylabel('Classification Accuracy - Random Forest')
    plt.xticks(num_brightness_levels, labels=[str(a) for a in brightness_subtractions])
    plt.ylim(0, 50)  
    plt.grid(True)
    plt.show()


#Perbutation 7 

occulsion_increase = input("Permutation 7 - Do you want to implement Occulsion?(1/0): ")
occulsion_increase = int(occulsion_increase)

if occulsion_increase == 1:
    occlusion_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    accuracy_scores_occlusion = []

    for size in occlusion_sizes:
        occluded_images = []
        for image in test_image:
            img = image.copy()
            if size > 0:
                # Randomly selecting the top-left corner of the square
                x = np.random.randint(0, img.shape[0] - size)
                y = np.random.randint(0, img.shape[1] - size)
                # Setting the square area to black
                img[x:x+size, y:y+size] = 0
            occluded_images.append(img)

        print(f"Occlusion with square edge length of {size}")

        # Feature extraction and classification 
        features_test_image_occlusion = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in occluded_images])
        predictions_image_occlusion = rf_classifier.predict(features_test_image_occlusion)
        accuracy = accuracy_score(test_labels, predictions_image_occlusion) * 100
        accuracy_scores_occlusion.append(accuracy)
        print(f"Occlusion Square Edge Length: {size}, Accuracy: {accuracy}")


    plt.figure(figsize=(10, 6))
    plt.plot(occlusion_sizes, accuracy_scores_occlusion, marker='o', linestyle='-', color='g')
    plt.title('Classification Accuracy - Random forest vs. Occlusion Square Edge Length')
    plt.xlabel('Occlusion Square Edge Length')
    plt.ylabel('Classification Accuracy (%)')
    plt.xticks(occlusion_sizes)
    plt.ylim(0, 50)  

    plt.grid(True)
    plt.show()
    
    
#Perbutation 8
salt_pepper = input("Permutation 8 - Do you want to implement Salt & Pepper Noise?(1/0): ")
salt_pepper= int(salt_pepper)

if salt_pepper == 1:
    noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    accuracy_scores_salt_pepper = []

    for noise_level in noise_levels:
        noised_images = [random_noise(image, mode='s&p', amount=noise_level) for image in test_image]
        noised_images = np.array([np.clip(image*255, 0, 255).astype(np.uint8) for image in noised_images])  # Adjust values back to 0-255 range and convert to uint8
        
        print(f"Adding salt and pepper noise: {noise_level}")

        # Feature extraction and classification 
        features_test_image_salt_pepper = np.array([hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1) for image in noised_images])
        predictions_image_salt_pepper = rf_classifier.predict(features_test_image_salt_pepper)
        accuracy = accuracy_score(test_labels, predictions_image_salt_pepper) * 100
        accuracy_scores_salt_pepper.append(accuracy)
        print(f"Noise Level: {noise_level}, Accuracy: {accuracy}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracy_scores_salt_pepper, marker='o', linestyle='-', color='blue')
    plt.title('Classification Accuracy vs. Salt and Pepper Noise Levels')
    plt.xlabel('Salt and Pepper Noise Level')
    plt.ylabel('Classification Accuracy (%)')
    plt.xticks(noise_levels)
    plt.grid(True)
    plt.show()