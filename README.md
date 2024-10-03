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
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("spidy.jpg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7, 3))
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
```
![1](https://github.com/user-attachments/assets/ba5f3682-5028-44f0-b828-3fdf1eac1fb7)

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(7, 3))
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()
```
![2](https://github.com/user-attachments/assets/40b1ac52-ea39-46e7-b457-f9227f6fd5c4)

ii) Using Weighted Averaging Filter
```Python
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(7, 3))
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
![3](https://github.com/user-attachments/assets/bf59b172-56d5-49c8-af00-33e3d5705d7c)

iii) Using Gaussian Filter
```Python
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)

plt.figure(figsize=(7, 3))
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
![4](https://github.com/user-attachments/assets/84422100-ccf4-4b58-9401-54b2cc4e074d)

iv)Using Median Filter
```Python
median_blur = cv2.medianBlur(image2, 11)

plt.figure(figsize=(7, 3))
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()

```
![5](https://github.com/user-attachments/assets/02fabe3f-5959-4ebb-aa40-fb0c02a2569a)

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
sharpened_image = cv2.filter2D(image2, -1, kernel1)

plt.figure(figsize=(7, 3))
plt.imshow(sharpened_image)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()
```
![6](https://github.com/user-attachments/assets/63c2808b-bf88-4795-a20c-2acd75952850)

ii) Using Laplacian Operator
```Python
laplacian = cv2.Laplacian(image2, cv2.CV_64F)

plt.figure(figsize=(7, 3))
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()

```
![7](https://github.com/user-attachments/assets/bc35065c-7180-455e-b0f0-a0038caeb75c)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
