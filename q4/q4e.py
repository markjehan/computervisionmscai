# beginner_style_display_all.py
# Run: python beginner_style_display_all.py
# Make sure spider.png is in the same folder.

import cv2
import matplotlib.pyplot as plt
import math

# ----------------------
# Settings (beginner style)
# ----------------------
INPUT_IMAGE = "spider.png"
ALPHA = 0.8   # strength of vibrance bump
SIGMA = 70.0  # spread of bump

def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def vibrance_pixel(s_value, alpha, sigma):
    # Gaussian-like bump centered at 128
    diff = (s_value - 128.0)
    exp_term = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    bump = alpha * 128.0 * exp_term
    new_val = s_value + bump
    return clamp_0_255(new_val)

def apply_vibrance_to_S(S, alpha, sigma):
    print("Applying vibrance to S channel (pixel-by-pixel)...")
    h = S.shape[0]
    w = S.shape[1]
    out = S.copy()
    y = 0
    while y < h:
        x = 0
        while x < w:
            s_val = int(S[y, x])
            out[y, x] = vibrance_pixel(s_val, alpha, sigma)
            x = x + 1
        y = y + 1
    return out

def build_transform_curve(alpha, sigma):
    print("Building transform curve f(x) for x in 0..255...")
    x_vals = []
    fx_vals = []
    i = 0
    while i <= 255:
        fx = vibrance_pixel(i, alpha, sigma)
        x_vals.append(i)
        fx_vals.append(fx)
        i = i + 1
    return x_vals, fx_vals

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not open", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    # Convert for display and processing
    print("Converting BGR->RGB for display and BGR->HSV for channels...")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Apply vibrance on S, then recombine and convert back to RGB
    S_v = apply_vibrance_to_S(S, ALPHA, SIGMA)
    print("Recombining H, S_v, V and converting HSV->BGR->RGB...")
    hsv_v = cv2.merge([H, S_v, V])
    bgr_v = cv2.cvtColor(hsv_v, cv2.COLOR_HSV2BGR)
    rgb_v = cv2.cvtColor(bgr_v, cv2.COLOR_BGR2RGB)

    # Build transform curve
    x_vals, fx_vals = build_transform_curve(ALPHA, SIGMA)

    # Display: Original | Vibrance | Curve
    print("Showing Original, Vibrance, and Transform Curve...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_v)
    plt.title("Vibrance (alpha=" + str(ALPHA) + ", sigma=" + str(SIGMA) + ")")
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
