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

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Load images with progress bar
def load_images(path, size=(256, 256), max_images=None):
    images = []
    print("📸 Loading dataset...")

    extensions = ['jpg', 'jpeg', 'png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{path}/**/*.{ext}", recursive=True))

    if not files:
        raise ValueError("❌ No images found. Please add images in the path provided.")

    for i, file in enumerate(files):
        try:
            img = imread(file)
            if img.shape[-1] != 3:
                continue
            img = resize(img, size, anti_aliasing=True)
            images.append(img)
            if i % 10 == 0:
                print(f"🔄 Loaded {i+1}/{len(files)} images...")
            if max_images and len(images) >= max_images:
                break
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")
    print(f"✅ Loaded {len(images)} images\n")
    return np.array(images)

# Preprocess for L and ab channels
def preprocess(images):
    lab = rgb2lab(images)
    L = lab[:,:,:,0:1] / 50.0 - 1  # Normalize to [-1,1]
    ab = lab[:,:,:,1:] / 128.0     # Normalize to [-1,1]
    return L, ab

# Reconstruct color image
def postprocess(L, ab):
    L = (L + 1) * 50
    ab = ab * 128
    lab = np.concatenate((L, ab), axis=-1)
    return lab2rgb(lab)

# Build model
def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))  # 2 channels for ab
    model.compile(optimizer='adam', loss='mse')
    return model

# Visualize training loss
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_curve.png")
    plt.close()
    print("📉 Saved loss curve to results/loss_curve.png")

# Colorize test images
def colorize_test_images(model, test_path="images/test", output_path="results"):
    print("🎯 Loading test images...")
    os.makedirs(output_path, exist_ok=True)
    
    test_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        test_files.extend(glob.glob(f"{test_path}/**/*.{ext}", recursive=True))
    
    if not test_files:
        print("❌ No test images found.")
        return

    for i, file in enumerate(test_files):
        try:
            img = imread(file)
            if img.ndim == 3 and img.shape[2] == 3:
                gray = rgb2lab(resize(img, (256, 256)))[:, :, 0]
            elif img.ndim == 2:
                gray = resize(img, (256, 256))
            else:
                print(f"⚠️ Skipped invalid format: {file}")
                continue

            L_test = gray / 50.0 - 1
            L_input = np.expand_dims(L_test, axis=(0, -1))

            pred_ab = model.predict(L_input)
            result_rgb = postprocess(L_input, pred_ab)

            # Side-by-side comparison
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(gray, cmap='gray')
            axs[0].set_title("Grayscale")
            axs[1].imshow(result_rgb[0])
            axs[1].set_title("Colorized")
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            filename = os.path.basename(file).split('.')[0]
            plt.savefig(f"{output_path}/comparison_{filename}.png")
            plt.close()
            print(f"✅ Saved: comparison_{filename}.png")
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

# ----------- Main Script -----------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Load dataset
    images = load_images("images/train", max_images=200)
    L, ab = preprocess(images)

    # Split for validation
    split = int(0.9 * len(L))
    L_train, L_val = L[:split], L[split:]
    ab_train, ab_val = ab[:split], ab[split:]

    # Build and train model
    print("🧠 Building model...")
    model = build_model()
    print("🏃 Training...")
    history = model.fit(
        L_train, ab_train,
        validation_data=(L_val, ab_val),
        batch_size=16,
        epochs=10,
        callbacks=[TqdmCallback(verbose=1)]
    )

    plot_loss(history)

    # Colorize test images
    colorize_test_images(model)
