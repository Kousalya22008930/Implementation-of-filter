# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: 
Import necessary libraries (cv2, NumPy, Matplotlib) for image loading, filtering, and visualization.

### Step 2: 
Load the image using cv2.imread() and convert it to RGB format using cv2.cvtColor() for proper display in Matplotlib.

### Step 3: 
Apply different filters:
1. Averaging Filter: Define an averaging kernel using np.ones() and apply it to the image using cv2.filter2D().
2. Weighted Averaging Filter: Define a weighted kernel (e.g., 3x3 Gaussian-like) and apply it with cv2.filter2D().
3. Gaussian Filter: Use cv2.GaussianBlur() to apply Gaussian blur.
4. Median Filter: Use cv2.medianBlur() to reduce noise.
5. Laplacian Operator: Use cv2.Laplacian() to apply edge detection.
    

### Step 4: 
Display each filtered image using plt.subplot() and plt.imshow() for side-by-side comparison of the original and processed images.

### Step 5: 
Save or show the images using plt.show() after applying each filter to visualize the effects of smoothing and sharpening.

```
Program:
Developed By   : KOUSALYA A.
Register Number: 212222230068
```
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image (convert to grayscale for simplicity)
image = cv2.imread('ex_5.tif', cv2.IMREAD_GRAYSCALE)
# Display original image
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/bfaec339-067f-42b1-83c2-cfcd932896a0)

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel = np.ones((4, 4), np.float32) / 9  # Simple 3x3 averaging kernel
averaged_image = cv2.filter2D(image, -1, kernel)  # Apply the filter
plt.imshow(averaged_image, cmap='gray')
plt.title("Averaging Filter")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/984cf000-f563-4f8a-a8d9-27b4dde7c1fd)

ii) Using Weighted Averaging Filter
```Python
weighted_kernel = np.array([[1, 2, 1], 
                            [2, 4, 2], 
                            [1, 2, 1]], np.float32)

weighted_kernel = weighted_kernel / weighted_kernel.sum()  # Normalize the weights
weighted_image = cv2.filter2D(image, -1, weighted_kernel)  # Apply the filter
plt.imshow(weighted_image, cmap='gray')
plt.title("Weighted Averaging Filter")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/c5901495-8f20-4c90-88e1-0487f0dfb7c0)

iii) Using Gaussian Filter
```Python
gaussian_image = cv2.GaussianBlur(image, (3, 3), 1)  # Apply Gaussian Blur
plt.imshow(gaussian_image, cmap='gray')
plt.title("Gaussian Filter")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/5bba550e-19e4-48fa-88f8-74ac4916cdba)

iv)Using Median Filter
```Python
median_image = cv2.medianBlur(image, 3)  # Apply Median Blur
plt.imshow(median_image, cmap='gray')
plt.title("Median Filter")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/96ee7a12-2c97-44ff-8b0a-77715bc521b8)


### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)  # Laplacian kernel
laplacian_image = cv2.filter2D(image, -1, laplacian_kernel)  # Apply the Laplacian
sharpened_laplacian_image = cv2.add(image, laplacian_image)  # Sharpen the image
plt.imshow(sharpened_laplacian_image, cmap='gray')
plt.title("Laplacian Kernel")
plt.axis('off')

```
![image](https://github.com/user-attachments/assets/c7b4f580-9c4a-4075-828f-55ef7950cee5)

ii) Using Laplacian Operator
```Python
laplacian_operator_image = cv2.Laplacian(image, cv2.CV_64F)  # Laplacian operator
laplacian_operator_image = cv2.convertScaleAbs(laplacian_operator_image)  # Convert to absolute values
sharpened_operator_image = cv2.add(image, laplacian_operator_image)  # Sharpen the image
plt.imshow(sharpened_operator_image, cmap='gray')
plt.title("Laplacian Operator")
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/38374bfe-8846-46ce-a9b4-9dce2914c335)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
