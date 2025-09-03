# beginner_style_combine_background_with_equalized_foreground.py
# Run: python beginner_style_combine_background_with_equalized_foreground.py
# Make sure jeniffer.jpg is in the same folder.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Settings
# ------------------------
INPUT_IMAGE = "jeniffer.jpg"
PLANE = "V"   # choose "S" or "V"
OUT_IMAGE = "jeniffer_equalized_foreground.png"
OUT_MASK = "jeniffer_mask.png"

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not read", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    print("Converting BGR->RGB (for display) and BGR->HSV (for channels)...")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Select which plane to equalize
    if PLANE.upper() == "S":
        plane = S
        print("Using S (Saturation) channel.")
    else:
        plane = V
        print("Using V (Value) channel.")

    # 1) Mask (foreground via Otsu)
    print("Building foreground mask with Otsu threshold...")
    _, mask = cv2.threshold(
        plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("Mask built. Foreground pixel count:", int((mask == 255).sum()))

    # 2) Equalize foreground only (manual LUT like in part (e))
    print("Collecting foreground pixel values...")
    vals = plane[mask == 255]
    print("Foreground values collected:", vals.size)

    print("Building histogram and CDF (np.cumsum)...")
    hist, _ = np.histogram(vals, bins=256, range=(0, 255))
    cdf = np.cumsum(hist)  # <-- using np.cumsum here

    # Handle case where all-zeros could happen
    cdf_nonzero = cdf[np.nonzero(cdf)]
    if cdf_nonzero.size > 0:
        cdf_min = cdf_nonzero.min()
    else:
        cdf_min = 0
    N = vals.size if vals.size > 0 else 1  # avoid divide-by-zero

    print("Building equalization LUT...")
    # LUT formula: (cdf - cdf_min) / (N - cdf_min) * 255
    denom = max(N - cdf_min, 1)
    lut = np.floor((cdf - cdf_min) / denom * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    print("Applying LUT only on foreground pixels...")
    eq_plane = plane.copy()
    eq_plane[mask == 255] = lut[plane[mask == 255]]

    # 3) Recombine with background: replace only chosen plane
    print("Recombining channels and converting back to RGB...")
    H_out = H.copy()
    S_out = S.copy()
    V_out = V.copy()

    if PLANE.upper() == "S":
        S_out = eq_plane
    else:
        V_out = eq_plane

    hsv_out = cv2.merge([H_out, S_out, V_out])
    bgr_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    print("Saving outputs...")
    ok_img = cv2.imwrite(OUT_IMAGE, bgr_out)
    ok_mask = cv2.imwrite(OUT_MASK, mask)
    if ok_img: print("Saved:", OUT_IMAGE)
    else:      print("Warning: could not save", OUT_IMAGE)
    if ok_mask: print("Saved:", OUT_MASK)
    else:       print("Warning: could not save", OUT_MASK)

    # Display: H, S, V plane, mask, original, result
    print("Showing H, S, V planes, mask, original, and result...")
    plt.figure(figsize=(13, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(H, cmap="gray")
    plt.title("Hue (H)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(S, cmap="gray", vmin=0, vmax=255)
    plt.title("Saturation (S)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(V, cmap="gray", vmin=0, vmax=255)
    plt.title("Value (V)")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(mask, cmap="gray")
    plt.title("Foreground mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(rgb_out)
    plt.title("Result: fg equalized (" + PLANE + ")")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
