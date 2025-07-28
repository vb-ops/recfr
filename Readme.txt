Readme
README: Image DEColorization using Deep Learning
----------------------------------------------

📌 PROJECT OVERVIEW:
This project uses a Convolutional Neural Network (CNN) to colorize grayscale images. The model is trained on a dataset of colored images and learns to predict the 'ab' chrominance channels from the Luminance channel (L) in LAB color space.

🛠️ LIBRARIES REQUIRED:
Make sure the following Python libraries are installed. You can install them using pip:
    pip install numpy matplotlib scikit-image keras tensorflow tqdm

Imports used:
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import rgb2lab, lab2rgb
    from skimage.io import imread
    from skimage.transform import resize
    from keras.models import Sequential
    from keras.layers import Conv2D, UpSampling2D, InputLayer
    import tensorflow as tf
    from tqdm.keras import TqdmCallback

⚙️ SYSTEM REQUIREMENTS:
- Python 3.7+
- GPU recommended (TensorFlow GPU support enabled)
- Internet connection (for downloading dependencies)

📁 FOLDER STRUCTURE:
Place your training and test images as follows:

    project_root/
    ├── images/
    │   ├── train/          --> All training images (jpg/jpeg/png)
    │   └── test/           --> Images to be colorized (grayscale)
    ├── results/            --> Will be auto-created to store output images
    ├── colorizeer.py           --> This is your main Python script
    └── README.txt          --> This file

🔄 PATHS TO SET:
- Training Images: `"images/train"` (in `load_images()`)
- Testing Images: `"images/test"` (in `colorize_test_images()`)
- Output Directory: `"results"` (auto-created)

📈 OUTPUT:
- During training, the model saves a plot of the loss curve: `results/loss_curve.png`
- For each test image, a comparison of grayscale and colorized versions is saved in `results/` with the format `comparison_<filename>.png`

🚀 USAGE:
1. Ensure required libraries are installed.
2. Place your training and test images in the correct folders.
3. Run the script:
       python script.py
4. Check the `results/` folder for outputs.


-43611106_Prathimani
