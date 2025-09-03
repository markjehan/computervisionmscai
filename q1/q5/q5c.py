# beginner_style_foreground_and_hist.py
# Run: python beginner_style_foreground_and_hist.py
# Make sure jeniffer.jpg is in the same folder.

import cv2
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
    print("Converting BGR -> HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Choose plane
    if PLANE.upper() == "S":
        plane = S
        print("Using S (Saturation) channel.")
    else:
        plane = V
        print("Using V (Value) channel.")

    # Apply Otsu threshold to build mask
    print("Applying Otsu threshold...")
    thresh_val, mask = cv2.threshold(
        plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("Otsu threshold value =", thresh_val)

    # Extract foreground (bitwise AND with mask)
    print("Extracting foreground region...")
    foreground = cv2.bitwise_and(plane, mask)

    # Build histogram of foreground manually
    print("Building histogram of foreground pixels...")
    hist = [0] * 256
    h = foreground.shape[0]
    w = foreground.shape[1]
    y = 0
    while y < h:
        x = 0
        while x < w:
            if mask[y, x] == 255:
                val = int(foreground[y, x])
                if val < 0: val = 0
                if val > 255: val = 255
                hist[val] = hist[val] + 1
            x = x + 1
        y = y + 1

    # ---- Show results ----
    print("Showing results...")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(foreground, cmap="gray", vmin=0, vmax=255)
    plt.title("Foreground " + PLANE)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(range(256), hist, width=1.0)
    plt.title("Foreground histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
