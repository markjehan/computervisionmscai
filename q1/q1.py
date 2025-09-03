# beginner_style_piecewise.py

# NOTE: Put emma.jpg in the SAME folder before running this.
# Run with: python beginner_style_piecewise.py

import matplotlib.pyplot as plt
from PIL import Image

# ---------- helper to convert a single pixel ----------
def convert_intensity(v):
    # Piecewise rule:
    # 1) 0–49 -> same value
    # 2) exactly 50 -> jump to 100
    # 3) 51–150 -> linear from 100 to 255
    # 4) 151–255 -> identity
    if v < 50:
        s = v
    elif v == 50:
        s = 100
    elif v > 50 and v <= 150:
        # linear: 100 + ( (255-100)/(150-50) ) * (v-50)
        s = 100 + ((255 - 100) / (150 - 50)) * (v - 50)
    else:
        s = v

    # clip and turn into int
    if s < 0:
        s = 0
    if s > 255:
        s = 255
    return int(s)

# ---------- build LUT (0..255) so we can also plot it ----------
def build_lut_list():
    lut_list = []
    r_values = []
    s_values = []
    i = 0
    while i <= 255:
        out_v = convert_intensity(i)
        lut_list.append(out_v)
        r_values.append(i)
        s_values.append(out_v)
        i = i + 1
    return lut_list, r_values, s_values

# ---------- main ----------
def main():
    print("Opening image...")
    try:
        img_path = "emma.jpg"
        img = Image.open(img_path).convert("L")  # make sure it's grayscale
    except Exception as e:
        print("Could not open emma.jpg. Error:", e)
        return

    print("Image opened. Size:", img.size)

    # Build LUT for plotting and mapping
    print("Building LUT...")
    lut_list, r_vals, s_vals = build_lut_list()
    print("LUT built with", len(lut_list), "entries.")

    # Apply transform pixel-by-pixel (beginner style loops)
    print("Applying transform (this might be a bit slow)...")
    w, h = img.size
    in_pixels = img.load()
    out_img = Image.new("L", (w, h))
    out_pixels = out_img.load()

    y = 0
    while y < h:
        x = 0
        while x < w:
            original_v = in_pixels[x, y]
            # use the LUT list (equivalent to convert_intensity(original_v))
            new_v = lut_list[original_v]
            out_pixels[x, y] = new_v
            x = x + 1
        y = y + 1

    # Save
    out_name = "emma_piecewise.png"
    out_img.save(out_name)
    print("Saved output as", out_name)

    # --- Plot graph of the LUT ---
    print("Plotting LUT...")
    plt.figure(figsize=(6, 6))
    plt.plot(r_vals, s_vals, linewidth=2)
    plt.title("Piecewise Intensity Transformation with Vertical Jump")
    plt.xlabel("Input Intensity (r)")
    plt.ylabel("Output Intensity (s)")
    plt.grid(True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.show()

    # --- Show images side by side ---
    print("Showing original and transformed images...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original (emma.jpg)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(out_img, cmap="gray", vmin=0, vmax=255)
    plt.title("Transformed")
    plt.axis("off")

    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
