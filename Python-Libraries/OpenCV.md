# OpenCV

## 1. What is OpenCV?
OpenCV (Open Source Computer Vision Library) is used for image and video processing, computer vision, and real-time visual applications. Its Python package is commonly installed as `opencv-python`.

## 2. Installation and import

```bash
pip install opencv-python
```

```python
import cv2
```

For headless servers without GUI display support, `opencv-python-headless` may be more appropriate. Do not install both variants in the same environment.

## 3. Read and inspect an image

```python
import cv2

image = cv2.imread("photo.jpg")
if image is None:
    raise FileNotFoundError("Could not read photo.jpg")

print(image.shape)  # height, width, channels
print(image.dtype)
```

OpenCV reads color images in **BGR** channel order by default, not RGB.

## 4. Display and color conversion

```python
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

In notebooks, Matplotlib is often more convenient for display:

```python
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
```

## 5. Resize, crop, and save

```python
small = cv2.resize(image, (300, 200))
crop = image[50:200, 100:250]  # rows first, then columns
cv2.imwrite("resized.jpg", small)
```

## 6. Basic image processing

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 200)

cv2.imwrite("edges.jpg", edges)
```

A threshold can convert a grayscale image into a binary image:

```python
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

## 7. Video capture

```python
cap = cv2.VideoCapture(0)  # default camera

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Camera access may not work in every notebook or remote environment.

## 8. Why OpenCV matters in Machine Learning
- Loading and preparing image datasets.
- Resizing images to a model's expected dimensions.
- Converting color spaces and reducing noise.
- Detecting edges, shapes, and visual features.
- Processing video frames before inference.

## 9. Practice
Read an image, print its shape, convert it to grayscale, resize it, detect edges, and save each result.

## 10. Common mistakes
- Confusing BGR and RGB.
- Not checking whether `imread` successfully loaded the file.
- Reversing row/column order when cropping.
- Using GUI display functions in an environment that has no display.
