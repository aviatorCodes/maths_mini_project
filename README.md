# 🧮 Matrix-Based Image Processing

Welcome to the **Matrix-Based Image Processing** project! This is an interactive web application built with Python and Streamlit. It demonstrates how standard image filters (like blur, sharpen, and edge detection) are actually just linear algebra concepts and matrix math operating under the hood.

Whether you are a beginner learning about arrays or a computer vision enthusiast, this project provides a hands-on way to visualize the math behind the magic.

---

## 🌟 What Does This Project Do?

In digital computing, an image is simply a grid of numbers (a matrix), where each number represents a pixel's color or brightness. 

This application allows you to:
1. **Upload an image** and see its exact dimensions and matrix properties.
2. **Apply mathematical filters** using a process called *convolution*.
3. **Tweak dynamic parameters** (like the size of a blur or the intensity of an emboss effect) in real-time.
4. **Compare the results** using data science metrics like Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
5. **View processing times** to understand the performance impact of different mathematical operations.

---

## 🧠 The Concepts: How It Works (For Beginners)

### 1. What is a Kernel?
A kernel (or filter) is a tiny matrix—usually 3x3, 5x5, or larger. Think of it as a small "window" or "magnifying glass" that slides across the original image. Different arrangements of numbers inside this tiny matrix produce different visual effects.
* A kernel full of equal fractions averages the pixels together, creating a **Box Blur**.
* A kernel that highlights differences between a pixel and its neighbors creates **Edge Detection** or **Sharpening**.

### 2. What is Convolution?
Convolution is the mathematical process of sliding that small kernel over the large image matrix. 
For every stop the kernel makes, it multiplies its own numbers by the image's numbers directly beneath it, adds them all up, and places the final sum into a brand-new image. We implemented this using **NumPy Tensor Dot Products**, which performs this math incredibly fast without using slow Python loops!

### 3. What is Padding?
When the kernel slides to the very edge of an image, it "hangs off" the side. To fix this, we add a fake border around the image before doing the math. This is called padding (e.g., reflecting the edge pixels).

---

## 📁 Project Structure

Here is a quick tour of the codebase:

```text
img_processing/
├── app.py                   # The main Streamlit web application UI
├── req.txt                  # Python dependencies (NumPy, Streamlit, Pillow, etc.)
│
├── core/                    # The "Brain" of the application
│   ├── convolution.py       # Contains the vectorized matrix math engine
│   ├── filters.py           # Logic for handling standard filters vs. Sobel
│   ├── img_utils.py         # Helpers to convert images to matrices and back
│   └── kernels.py           # The mathematical definitions for our filter matrices
│
├── analysis/                # The "Evaluator" of the application
│   ├── benchmark.py         # Measures how fast the filters run
│   └── metrics.py           # Calculates image difference (MSE, PSNR)
│
└── tests/                   # Automated tests to ensure our math is correct
    └── test_convolution.py
```

---

## 🚀 Getting Started

Follow these instructions to run the project on your local machine.

### Prerequisites
Make sure you have [Python 3.8+](https://www.python.org/downloads/) installed on your computer.

### 1. Clone the repository
Download this project to your local machine:
```bash
git clone <your-repository-url>
cd img_processing
```

### 2. Install the required libraries
It is highly recommended to use a virtual environment. Install the necessary Python packages using your requirements file:
```bash
pip install -r req.txt
```
*(Note: If you don't have a `req.txt` yet, you will need `streamlit`, `numpy`, `Pillow`, and `matplotlib`)*

### 3. Run the Application
Start the Streamlit server with this command:
```bash
streamlit run app.py
```

A browser window should automatically open pointing to `http://localhost:8501`. 

### 4. Play with it!
* Expand the sidebar on the left.
* Upload a `.jpg` or `.png` file.
* Select a filter like **Gaussian Blur**.
* Adjust the sliders to see how changing the underlying math dynamically alters the visual output!

---

## 🛠️ Built With

* **[Python](https://www.python.org/)** - The main programming language.
* **[NumPy](https://numpy.org/)** - For high-performance, vectorized matrix mathematics.
* **[Streamlit](https://streamlit.io/)** - For instantly turning Python scripts into interactive web apps.
* **[Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)** - For reading and saving image files.