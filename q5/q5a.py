# beginner_style_split_hsv_jennifer.py
# Run: python beginner_style_split_hsv_jennifer.py
# Make sure jeniffer.jpg is in the same folder.

import cv2
import matplotlib.pyplot as plt

# ---------------------
# Settings
# ---------------------
INPUT_IMAGE = "jeniffer.jpg"
OUT_H = "jeniffer_H.png"
OUT_S = "jeniffer_S.png"
OUT_V = "jeniffer_V.png"

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if bgr is None:
        print("Error: Could not read", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    # Convert BGR (OpenCV default) to RGB (for display with matplotlib)
    print("Converting BGR -> RGB...")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert BGR -> HSV
    print("Converting BGR -> HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Split channels
    print("Splitting HSV channels...")
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Show some stats
    print("H channel min =", H.min(), "max =", H.max(), "(expected 0..179)")
    print("S channel min =", S.min(), "max =", S.max(), "(expected 0..255)")
    print("V channel min =", V.min(), "max =", V.max(), "(expected 0..255)")

    # Save each channel as an image
    print("Saving H, S, V images...")
    ok_h = cv2.imwrite(OUT_H, H)
    ok_s = cv2.imwrite(OUT_S, S)
    ok_v = cv2.imwrite(OUT_V, V)
    if ok_h: print("Saved:", OUT_H)
    if ok_s: print("Saved:", OUT_S)
    if ok_v: print("Saved:", OUT_V)

    # Plot all results
    print("Showing 2x2 figure...")
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title("Original (RGB)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(H, cmap="gray")
    plt.title("Hue (H)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(S, cmap="gray", vmin=0, vmax=255)
    plt.title("Saturation (S)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(V, cmap="gray", vmin=0, vmax=255)
    plt.title("Value (V)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
