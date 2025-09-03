# beginner_style_apply_to_s.py
# Run: python beginner_style_apply_to_s.py
# Make sure spider.png is in the same folder.

import cv2
import matplotlib.pyplot as plt
import math

# ----------------------
# Settings (beginner style)
# ----------------------
INPUT_IMAGE = "spider.png"
OUTPUT_S = "spider_S_vibrance.png"

ALPHA = 0.8   # strength of bump
SIGMA = 70.0  # spread of bump

def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def vibrance_pixel(s_value, alpha, sigma):
    # Apply bump around 128
    diff = (s_value - 128.0)
    exp_term = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    bump = alpha * 128.0 * exp_term
    new_val = s_value + bump
    return clamp_0_255(new_val)

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not open", INPUT_IMAGE)
        return
    print("Image opened. Shape:", bgr.shape)

    # Convert to HSV
    print("Converting to HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Apply vibrance pixel-by-pixel to S
    print("Applying vibrance transform to S channel...")
    h = S.shape[0]
    w = S.shape[1]
    S_vib = S.copy()
    y = 0
    while y < h:
        x = 0
        while x < w:
            orig_val = int(S[y, x])
            new_val = vibrance_pixel(orig_val, ALPHA, SIGMA)
            S_vib[y, x] = new_val
            x = x + 1
        y = y + 1

    # Save the vibrance S channel image
    ok = cv2.imwrite(OUTPUT_S, S_vib)
    if ok:
        print("Saved vibrance S channel to:", OUTPUT_S)
    else:
        print("Warning: could not save output file.")

    # ---- Build transform curve f(x) for 0..255 ----
    print("Building transform curve f(x)...")
    x_vals = []
    fx_vals = []
    i = 0
    while i <= 255:
        fx_vals.append(vibrance_pixel(i, ALPHA, SIGMA))
        x_vals.append(i)
        i = i + 1

    # ---- Show before/after and curve ----
    print("Showing images and curve...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(S, cmap="gray", vmin=0, vmax=255)
    plt.title("S (original)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(S_vib, cmap="gray", vmin=0, vmax=255)
    plt.title("S after f(x) (alpha=" + str(ALPHA) + ")")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(x_vals, fx_vals, linewidth=2)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.title("Intensity transform f(x)")
    plt.xlabel("Input S")
    plt.ylabel("Output S")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
