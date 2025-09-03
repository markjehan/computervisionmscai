import cv2
import matplotlib.pyplot as plt

# ---------------------
# Settings (beginner style)
# ---------------------
INPUT_IMAGE = "inputimg.jpg"
OUTPUT_IMAGE = "output_gamma.png"
GAMMA = 0.6  # <1 brightens, >1 darkens

def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def main():
    print("Opening image:", INPUT_IMAGE)
    img_bgr = cv2.imread(INPUT_IMAGE)
    if img_bgr is None:
        print("Error: Could not read the image file. Check the name/path.")
        return
    print("Image opened. Size (HxW):", img_bgr.shape[0], "x", img_bgr.shape[1])

    # Keep an RGB copy for display (OpenCV loads as BGR)
    rgb_before = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to Lab
    print("Converting to Lab color space...")
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # Apply gamma to L* channel using simple loops
    print("Applying gamma to L* (beginner pixel-by-pixel; may be slow)...")
    h = L.shape[0]
    w = L.shape[1]
    L_after = L.copy()

    # Gamma formula (on 0..1): out = in^gamma
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Read original L value (0..255)
            L_val = int(L[y, x])
            # Normalize to 0..1
            L_norm = L_val / 255.0
            # Apply gamma
            L_gamma = L_norm ** GAMMA
            # Scale back to 0..255 and clamp
            L_new = clamp_0_255(L_gamma * 255.0)
            # Write back
            L_after[y, x] = L_new
            x = x + 1
        y = y + 1

    # Merge channels and convert back to BGR then RGB for display
    print("Merging channels and converting back to RGB...")
    lab_out = cv2.merge([L_after, a, b])
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    rgb_after = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    # Save output (convert back to BGR for OpenCV saving)
    ok = cv2.imwrite(OUTPUT_IMAGE, cv2.cvtColor(rgb_after, cv2.COLOR_RGB2BGR))
    if ok:
        print("Saved corrected image to:", OUTPUT_IMAGE)
    else:
        print("Warning: Could not save output image.")
    print("Gamma used (gamma):", GAMMA)

    # ---- Show before/after images ----
    print("Showing before/after images...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_before)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_after)
    plt.title("Gamma-corrected L* (gamma=" + str(GAMMA) + ")")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Histograms for L channel (beginner style counts) ----
    print("Building histograms for L* channel...")
    hist_before = [0] * 256
    hist_after = [0] * 256

    y = 0
    while y < h:
        x = 0
        while x < w:
            v1 = int(L[y, x])
            v2 = int(L_after[y, x])
            if v1 < 0: v1 = 0
            if v1 > 255: v1 = 255
            if v2 < 0: v2 = 0
            if v2 > 255: v2 = 255
            hist_before[v1] = hist_before[v1] + 1
            hist_after[v2] = hist_after[v2] + 1
            x = x + 1
        y = y + 1

    # Plot histograms
    print("Showing histograms...")
    plt.figure(figsize=(7, 6))
    # Beginners often just plot with default settings twice
    plt.plot(range(256), hist_before, label="Original L*")
    plt.plot(range(256), hist_after, label="Gamma-corrected L*")
    plt.title("Histograms of L* Channel")
    plt.xlabel("L* intensity (0–255)")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
