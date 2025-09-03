# beginner_style_equalize_foreground_only.py
# Run: python beginner_style_equalize_foreground_only.py
# Make sure jeniffer.jpg is in the same folder.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Settings
# ------------------------
INPUT_IMAGE = "jeniffer.jpg"
PLANE = "V"   # choose "S" or "V"

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not read", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    # Convert to HSV
    print("Converting to HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Choose channel
    if PLANE.upper() == "S":
        plane = S
        print("Using Saturation (S) channel.")
    else:
        plane = V
        print("Using Value (V) channel.")

    # Build mask using Otsu
    print("Building foreground mask with Otsu threshold...")
    _, mask = cv2.threshold(
        plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Extract foreground pixel values
    print("Collecting foreground pixel values...")
    vals = plane[mask == 255]
    print("Foreground pixel count:", vals.size)

    # Histogram + CDF
    print("Building histogram and cumulative sum...")
    hist, _ = np.histogram(vals, bins=256, range=(0, 255))
    cdf = np.cumsum(hist)  # 👈 np.cumsum used here
    # find first nonzero value
    cdf_nonzero = cdf[np.nonzero(cdf)]
    if cdf_nonzero.size > 0:
        cdf_min = cdf_nonzero.min()
    else:
        cdf_min = 0
    N = vals.size

    # Build LUT for equalization
    print("Building equalization LUT...")
    # formula: (cdf - cdf_min) / (N - cdf_min) * 255
    lut = np.floor((cdf - cdf_min) / max(N - cdf_min, 1) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply LUT only on foreground
    print("Applying LUT to foreground pixels...")
    eq_plane = plane.copy()
    eq_plane[mask == 255] = lut[plane[mask == 255]]

    # ---- Show results ----
    print("Showing results...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(plane, cmap="gray", vmin=0, vmax=255)
    plt.title(PLANE + " (original)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(eq_plane, cmap="gray", vmin=0, vmax=255)
    plt.title(PLANE + " (equalized fg)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(lut)
    plt.title("Equalization mapping (LUT)")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.ylim(0, 255)

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
