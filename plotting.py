from alexNet import accuracies_noise_dl, accuracies_blur_dl, accuracies_contrast_increase_dl, accuracies_brightness_decrease_dl, accuracies_contrast_decrease_dl, accuracies_brightness_increase_dl, accuracies_occlusion_dl, accuracies_salt_pepper_dl
from randomForest import accuracy_scores_perb_one, accuracy_scores_blur, accuracy_scores_contrast_increase, accuracy_scores_contrast_decrease, accuracy_scores_brightness_increase, accuracy_scores_brightness_decrease, accuracy_scores_occlusion, accuracy_scores_salt_pepper
import matplotlib.pyplot as plt




#For perbutation 1 
standard_deviations = [1,2,3,4,5,6,7,8,9,10]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(standard_deviations, accuracy_scores_perb_one, label='Classical ML Model - Random Forest', marker='o')
plt.plot(standard_deviations, accuracies_noise_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

plt.title('Accuracy vs. Standard Deviation for Different Noises')
plt.xlabel('Standard Deviation')
plt.ylabel('Accuracy')
plt.legend()

plt.grid(True)
plt.show()

#For Perbutation 2 
num_conv = [1,2,3,4,5,6,7,8,9,10]
# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(num_conv, accuracy_scores_blur, label='Classical ML Model - Random Forest', marker='o')
plt.plot(num_conv, accuracies_blur_dl, label='Deep Learning Model - AlexNet Model ', marker='x')


plt.title('Accuracy vs. Number of Times Gaussian Blurring Applied')
plt.xlabel('Number of Convolutions')
plt.ylabel('Accuracy')
plt.legend()

plt.grid(True)
plt.show()



#For Perbutation 3
# Plotting the data
plt.figure(figsize=(10, 6))
contrast_increase_factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]

plt.plot(contrast_increase_factors, accuracy_scores_contrast_increase, label='Classical ML Model - Random Forest', marker='o')
plt.plot(contrast_increase_factors, accuracies_contrast_increase_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

plt.title('Accuracy vs. Contrast Increase')
plt.xlabel('Contrast Multipliers')
plt.ylabel('Accuracy')
plt.legend()

plt.grid(True)
plt.show()



#Perbutation 4 
contrast_decrease_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(contrast_decrease_factors, accuracy_scores_contrast_decrease, label='Classical ML Model - Random Forest', marker='o')
plt.plot(contrast_decrease_factors, accuracies_contrast_decrease_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

# Adding title and labels
plt.title('Accuracy vs. Contrast Decrease')
plt.xlabel('Contrast Multipliers')
plt.ylabel('Accuracy')
plt.legend()
plt.xlim(max(contrast_decrease_factors), min(contrast_decrease_factors))

# Show the plot
plt.grid(True)
plt.show()


#Perbutation 5 
brightness_increase_factors = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(brightness_increase_factors, accuracy_scores_brightness_increase, label='Classical ML Model - Random Forest', marker='o')
plt.plot(brightness_increase_factors, accuracies_brightness_increase_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

# Adding title and labels
plt.title('Accuracy vs. Brightness Increase')
plt.xlabel('Brightness Factor')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


#Perbutation 6
brightness_decrease_factors = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(brightness_decrease_factors, accuracy_scores_brightness_decrease, label='Classical ML Model - Random Forest', marker='o')
plt.plot(brightness_decrease_factors, accuracies_brightness_decrease_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

# Adding title and labels
plt.title('Accuracy vs. Brightness Decrease')
plt.xlabel('Brightness Factor')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

#Perbutation 7 
edge_length = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
plt.figure(figsize=(10, 6))
plt.plot(edge_length, accuracy_scores_occlusion, label='Classical ML Model - Random Forest', marker='o')
plt.plot(edge_length, accuracies_occlusion_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

# Adding title and labels
plt.title('Accuracy vs. Occlusion')
plt.xlabel('Occlusion Square Edge Length')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0,100)
# Show the plot
plt.grid(True)
plt.show()


#Perbutation 8

noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]


plt.figure(figsize=(10, 6))
plt.plot(noise_levels, accuracy_scores_salt_pepper, label='Classical ML Model - Random Forest', marker='o')
plt.plot(noise_levels, accuracies_salt_pepper_dl, label='Deep Learning Model - AlexNet Model ', marker='x')

# Adding title and labels
plt.title('Accuracy vs. Salt and Pepper Noise')
plt.xlabel('Salt and Pepper Noise Levels')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0,100)
# Show the plot
plt.grid(True)
plt.show()
