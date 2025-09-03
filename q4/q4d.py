# beginner_style_recombine_with_alpha.py
# Run: python beginner_style_recombine_with_alpha.py
# Make sure spider.png is in the same folder.

import cv2
import math

# ---------------------
# Settings
# ---------------------
INPUT_IMAGE = "spider.png"
OUTPUT_IMAGE = "spider_vibrance.png"

ALPHA = 0.8   # chosen alpha from q4c
SIGMA = 70.0  # sigma for bump

# ---------------------
# Helpers
# ---------------------
def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def vibrance_pixel(s_value, alpha, sigma):
    # bump function centered at 128
    diff = (s_value - 128.0)
    exp_term = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    bump = alpha * 128.0 * exp_term
    new_val = s_value + bump
    return clamp_0_255(new_val)

def vibrance_channel(S, alpha, sigma):
    print("Applying vibrance to S channel (pixel-by-pixel)...")
    h = S.shape[0]
    w = S.shape[1]
    out = S.copy()
    y = 0
    while y < h:
        x = 0
        while x < w:
            val = int(S[y, x])
            new_val = vibrance_pixel(val, alpha, sigma)
            out[y, x] = new_val
            x = x + 1
        y = y + 1
    return out

# ---------------------
# Main
# ---------------------
def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not open", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    print("Converting to HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Apply vibrance to S channel
    S_v = vibrance_channel(S, ALPHA, SIGMA)

    # Recombine channels
    print("Recombining channels...")
    hsv_v = cv2.merge([H, S_v, V])

    # Convert back to BGR
    bgr_v = cv2.cvtColor(hsv_v, cv2.COLOR_HSV2BGR)

    # Save result
    ok = cv2.imwrite(OUTPUT_IMAGE, bgr_v)
    if ok:
        print("Saved recombined vibrance image ->", OUTPUT_IMAGE, "(alpha =", ALPHA, ")")
    else:
        print("Warning: could not save the output image.")

if __name__ == "__main__":
    main()
