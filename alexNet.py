import os
import cv2
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage import util as skimage_util
from sklearn.metrics import accuracy_score


transform_training_data=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

transform_val_test_data = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading training data
train_dataset = datasets.ImageFolder(root="../dataset_original/train/", transform=transform_training_data)


torch.manual_seed(10)


val_size = len(train_dataset) // 5  # 20% for validation
train_size = len(train_dataset) - val_size  # Remaining 80% for training

train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)  

test_ds = datasets.ImageFolder(root="../dataset_original/test/", transform=transform_val_test_data)
test_loader = DataLoader(test_ds, batch_size=32)

# Printing dataset sizes
print(f"Training dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")
print(f"Test dataset size: {len(test_ds)}")

full_dataset = datasets.ImageFolder(root="../dataset_original/train/", transform=None)
class_names = full_dataset.classes


for i,_ in val_loader:
    print(f"Minimum value in tensor: {i.min()}")
    print(f"Maximum value in tensor: {i.max()}")
    break



for images, labels in train_loader:
    break
#print the labels
# print('Label:', labels.numpy())
# print('Class:', *np.array([class_names[i] for i in labels]))

im=make_grid(images,nrow=16)

alex= models.alexnet(pretrained=True)

for param in alex.parameters():
    param.requires_grad=False
    
torch.manual_seed(42)

alex.classifier=nn.Sequential(  nn.Linear(9216,1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(1024,len(class_names)),
                                nn.LogSoftmax(dim=1))

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(alex.classifier.parameters(),lr=0.001)
epochs = 12
batch_size = 32
train_losses = []
test_losses = []
train_correct = []
test_correct = []


model_path = 'alexnet_model.pth' # Define the path to save or load the model

# save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval() 
    print(f'Model loaded from {path}')

start_time = time.time()


if os.path.exists(model_path):
    load_model(alex, model_path) 
    print("Model Loaded..")
else:
    # Training process
    print("Training Model..")
    for i in range(epochs):
        trn_corr = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            y_pred = alex(X_train)
            loss = criterion(y_pred, y_train)
            
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (b+1) % 30 == 0:
                print(f'Epoch: {i+1:2}/{epochs} | Batch: {b+1:4} | Loss: {loss.item():.4f} | Accuracy: {trn_corr*100/((b+1)*batch_size):.2f}%')

        epoch_accuracy = trn_corr * 100 / (len(train_loader.dataset))
        detached_loss = loss.item()
        
        # Logging
        train_losses.append(detached_loss)
        train_correct.append(trn_corr)
        
        print(f'Epoch: {i+1:2}/{epochs} | Loss: {detached_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%')

    save_model(alex, model_path)

duration = time.time() - start_time
print(f'Training complete in {duration//60:.0f}m {duration%60:.0f}s')


val = input("Do you wanna check your models performance on the validation set?(0/1) : ")
val = int(val)
if val==1:
    alex.eval()
    print("Model in evaluation mode")

    with torch.no_grad():
        predictions = []
        true_labels = []
        
        for inputs, labels in val_loader:
            outputs = alex(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.view(-1).cpu().numpy())
            true_labels.extend(labels.view(-1).cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")




#Perbutation 1 Gaussian Pixel Noise 
perb1 = int(input("Test perb 1 : "))
if perb1==1:
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    # Base transform without noise
    transform_val_test_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def evaluate_model_with_noise(model, std_dev):
        transform_val_test_data_with_noise = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AddGaussianNoise(0., std_dev / 255.0),  # Apply Gaussian noise
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load the test dataset with the updated transformation
        test_ds_with_noise = datasets.ImageFolder(root="../dataset_original/test/", transform=transform_val_test_data_with_noise)
        test_loader_with_noise = DataLoader(test_ds_with_noise, batch_size=32)

        # Evaluate the model
        alex.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader_with_noise:
                outputs = alex(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    # Main evaluation loop
    standard_deviations = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    accuracies_noise_dl = []

    for std_dev in standard_deviations:
        accuracy = evaluate_model_with_noise(alex, std_dev)
        accuracies_noise_dl.append(accuracy)
        print(f'Std Dev: {std_dev}, Accuracy: {accuracy}%')




#Perbutation 2 Gaussian blur
perb2 = int(input("Check perbutation 2? (0/1) : "))
if perb2 == 1:
    # Defineing kernel
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32) / 16

    def apply_gaussian_blur(image_np, n):
        """Apply Gaussian blur n times."""
        blurred_image = image_np.copy()
        for _ in range(n):
            blurred_image = cv2.filter2D(blurred_image, -1, gaussian_kernel)
        return blurred_image
    
    def apply_gaussian_blur_to_tensor(image_tensor, n_blur):
        # Convert tensor to numpy array
        image_np = image_tensor.numpy().transpose((1, 2, 0))
        blurred_image_np = apply_gaussian_blur(image_np, n_blur)
        # Convert to tensor
        blurred_image_tensor = torch.tensor(blurred_image_np.transpose((2, 0, 1)))
        return blurred_image_tensor
    

    def evaluate_model_with_blur(n_blur):
        """Evaluate the model with images blurred n times."""
        alex.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs_blurred = torch.stack([apply_gaussian_blur_to_tensor(img, n_blur) for img in inputs])
                outputs = alex(inputs_blurred)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.view(-1).cpu().numpy())
                true_labels.extend(labels.view(-1).cpu().numpy())
                
        print(f"Blur number : {n_blur}, Accuracy : ", accuracy_score(true_labels, predictions))
        return accuracy_score(true_labels, predictions)*100

    blur_levels = range(10)  # 0 to 9 times
    accuracies_blur_dl = [evaluate_model_with_blur(n) for n in blur_levels]

    # Plotting the accuracies against the blur levels
    plt.figure(figsize=(10, 6))
    plt.plot(blur_levels, accuracies_blur_dl, marker='o', linestyle='-', color='b')
    plt.title('AlexNet Model Accuracy vs. Gaussian Blur Levels')
    plt.xlabel('Number of Times Gaussian Blur Applied')
    plt.ylabel('Accuracy - Alex Net Model')
    plt.ylim(0,100)
    plt.grid(True)
    plt.show()
    
    
    # # Plotting the accuracies against the blur levels
    # plt.figure(figsize=(10, 6))
    # plt.plot(blur_levels, accuracies, marker='o', linestyle='-', color='b')
    # plt.title('AlexNet Model Accuracy vs. Gaussian Blur Levels')
    # plt.xlabel('Number of Times Gaussian Blur Applied')
    # plt.ylabel('Accuracy - Alex Net Model')
    # plt.grid(True)
    # plt.show()
    



def denormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return images * std + mean

def renormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std

def adjust_contrast(images, multiplier):
    #denormalise images
    images = denormalize(images)
    
    images = torch.clamp(images, 0, 1)
    
    # Adjust contrast
    images = images * multiplier
    images = torch.clamp(images, 0, 1)  
    
    # Renormalize images
    images = renormalize(images)
    
    return images



perb3 = int(input("Check Perbutation 3 : "))
if perb3 == 1:
    factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
    alex.eval()
    accuracies_contrast_increase_dl = []

    for factor in factors:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            # Adjust contrast
            adjusted_images = adjust_contrast(images.float(), factor)
            
            adjusted_images = adjusted_images.to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(adjusted_images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        accuracies_contrast_increase_dl.append(accuracy)
        print(f"Factor : {factor} and Accuracy : ", accuracy)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies_contrast_increase_dl, marker='o', linestyle='-', color='b')
    plt.title('Factor vs. Accuracy')
    plt.xlabel('Contrast Factor')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(factors)
    plt.show()
    
    
    plt.style.use('ggplot') 

    plt.figure(figsize=(12, 7))  
    plt.plot(factors, accuracies_contrast_increase_dl, marker='o', markersize=8, linestyle='-', linewidth=2, color='b')
    plt.title('Contrast Factor vs. Model Accuracy', fontsize=18)
    plt.xlabel('Contrast Factor', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.axis([min(factors), max(factors), min(accuracies_contrast_increase_dl) - 0.05, max(accuracies_contrast_increase_dl) + 0.05])
    plt.xticks(factors, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,100)

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.show()
    
    
    
    

    
    
    

perb4 = int(input("Check Perbutation 4 : "))
if perb4 == 1:
    factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    alex.eval()
    accuracies_contrast_decrease_dl = []

    for factor in factors:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            # Adjust contrast
            adjusted_images = adjust_contrast(images.float(), factor)
            
            adjusted_images = adjusted_images.to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(adjusted_images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        accuracies_contrast_decrease_dl.append(accuracy)
        print(f"Factor : {factor} and Accuracy : ", accuracy)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies_contrast_decrease_dl, marker='o', linestyle='-', color='b')
    plt.title('Factor vs. Accuracy')
    plt.xlabel('Contrast Factor')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(factors)
    plt.show()
    
    
    plt.style.use('ggplot') 

    plt.figure(figsize=(12, 7))  
    plt.plot(factors, accuracies_contrast_decrease_dl, marker='o', markersize=8, linestyle='-', linewidth=2, color='b')
    plt.title('Contrast Factor vs. Model Accuracy', fontsize=18)
    plt.xlabel('Contrast Factor', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.axis([min(factors), max(factors), min(accuracies_contrast_decrease_dl) - 0.05, max(accuracies_contrast_decrease_dl) + 0.05])
    plt.xticks(factors, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0,100)

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.xticks(rotation=45)
    plt.tight_layout() 
    plt.show()
  
  
    
    
def increase_brightness(images, brightness):
    images += brightness
    images = torch.clamp(images, 0, 255)
    return images
    
    
perb5 = int(input("Check Perbutation 5 : "))
if perb5 == 1:      
    factors = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Brightness factors
    alex.eval()
    accuracies_brightness_increase_dl = []

    for factor in factors:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            # Increase brightness
            brightened_images = increase_brightness(images.float(), factor)
            
            brightened_images = brightened_images.to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(brightened_images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        accuracies_brightness_increase_dl.append(accuracy)
        print(f"Factor : {factor} and Accuracy : ", accuracy)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies_brightness_increase_dl, marker='o', linestyle='-', color='b')
    plt.title('Factor vs. Accuracy')
    plt.xlabel('Brightness Factor')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(factors)
    plt.show()
    
def decrease_brightness(images, brightness):
    images -= brightness
    # images = torch.clamp(images, 0, 255)
    return images


perb6 = int(input("Check Perturbation 6: "))   
if perb6 == 1: 
    factors = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    alex.eval()
    accuracies_brightness_decrease_dl = []

    for factor in factors:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            # Decrease brightness
            darkened_images = decrease_brightness(images.float(), factor)
            
            darkened_images = darkened_images.to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(darkened_images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        accuracies_brightness_decrease_dl.append(accuracy)
        print(f"Factor : {factor} and Accuracy : ", accuracy)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies_brightness_decrease_dl, marker='o', linestyle='-', color='b')
    plt.title('Factor vs. Accuracy')
    plt.xlabel('Brightness Factor')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(factors)
    plt.show()
    
    
perb7 = int(input("Check Perturbation 7: "))   
def apply_occlusion(images, edge_length):
        # edge_length: Size of the square occlusion
        if edge_length == 0:
            return images  # No occlusion
        batch_size, _, height, width = images.shape
        occluded_images = images.clone()
        
        for i in range(batch_size):
            # Randomly choose the top-left corner of the square
            x = np.random.randint(0, height - edge_length)
            y = np.random.randint(0, width - edge_length)
            occluded_images[i, :, x:x+edge_length, y:y+edge_length] = 0  # Apply occlusion
        return occluded_images
 
if perb7 == 1: 
    factors = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    alex.eval()
    accuracies_occlusion_dl = []

    for factor in factors:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            # Apply occlusion
            occluded_images = apply_occlusion(images.float(), factor)
            
            occluded_images = occluded_images.to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(occluded_images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        accuracies_occlusion_dl.append(accuracy)
        print(f"Occlusion Edge Length: {factor}, Accuracy: {accuracy}")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(factors, accuracies_occlusion_dl, marker='o', linestyle='-', color='b')
    plt.title('Occlusion Edge Length vs. Accuracy')
    plt.xlabel('Occlusion Edge Length')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(factors)
    plt.show()
    
    
perb8 = int(input("Check Perturbation 8 : "))
if perb8 == 1:
    noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    alex.eval()  
    accuracies_salt_pepper_dl = []

    for noise_level in noise_levels:
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            images_np = images.numpy()
            noisy_images = np.zeros_like(images_np)
            
            for i in range(images_np.shape[0]):
                # Apply salt and pepper noise
                noisy_images[i] = skimage_util.random_noise(images_np[i], mode='s&p', amount=noise_level)
            
            noisy_images_torch = torch.tensor(noisy_images, dtype=torch.float).to(next(alex.parameters()).device)
            labels = labels.to(next(alex.parameters()).device)
            
            with torch.no_grad():
                outputs = alex(noisy_images_torch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        accuracies_salt_pepper_dl.append(accuracy)
        print(f"Noise Level: {noise_level}, Accuracy: {accuracy}")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies_salt_pepper_dl, marker='o', linestyle='-', color='b')
    plt.title('Salt and Pepper Noise Level vs. Accuracy')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(noise_levels)
    plt.show()