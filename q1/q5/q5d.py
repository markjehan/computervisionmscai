import cv2
import matplotlib.pyplot as plt
import numpy as np

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

    # Select plane
    if PLANE.upper() == "S":
        plane = S
        print("Using Saturation (S) channel.")
    else:
        plane = V
        print("Using Value (V) channel.")

    # Create mask using Otsu threshold
    print("Applying Otsu threshold to build mask...")
    thresh_val, mask = cv2.threshold(
        plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("Otsu threshold value chosen:", thresh_val)

    # Collect values of pixels inside foreground
    print("Collecting foreground values...")
    vals = plane[mask == 255]   # take only the plane values where mask=255
    print("Total foreground pixels collected:", len(vals))

    # Build histogram using numpy
    print("Building histogram (256 bins)...")
    hist, _ = np.histogram(vals, bins=256, range=(0, 255))

    # Compute cumulative sum with numpy
    print("Computing cumulative sum using np.cumsum...")
    cdf = np.cumsum(hist)   # running total across intensity bins

    # Show last value for confirmation
    print("Final CDF value (should equal number of foreground pixels):", cdf[-1])

    # ---- Show histogram and CDF ----
    print("Displaying histogram and cumulative sum...")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(256), hist, width=1.0)
    plt.title("Histogram (foreground)")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.plot(range(256), cdf)
    plt.title("Cumulative sum (CDF)")
    plt.xlabel("Intensity")
    plt.ylabel("Cumulative Count")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
