import os
import random
import colorsys
import cv2
import numpy as np
import pandas as pd

# Define file to store labeled colors
CSV_FILE = "labeled_colors.csv"

# Check if the CSV exists, if not, create it
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["R", "G", "B", "H", "S", "V", "HEX", "Color_Name"])
    df.to_csv(CSV_FILE, index=False)

def generate_random_color():
    """Generate a random RGB color."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b

def convert_to_hsv(r, g, b):
    """Convert RGB to HSV format."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return int(h * 360), int(s * 100), int(v * 100)  # Convert to standard HSV

def convert_to_hex(r, g, b):
    """Convert RGB to HEX format."""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def show_color(r, g, b):
    """Display the color on the screen using OpenCV and ensure proper closing."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (b, g, r)  # OpenCV uses BGR format
    
    cv2.imshow("What is this color?", img)

    while True:
        key = cv2.waitKey(1) & 0xFF  # Continuously check for key press
        if key == 27 or cv2.getWindowProperty("What is this color?", cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit loop when ESC is pressed or window is closed manually
    
    cv2.destroyAllWindows()

def save_color_data(r, g, b, h, s, v, hex_code, color_name):
    """Save the labeled color data to the CSV file."""
    df = pd.read_csv(CSV_FILE)
    new_data = pd.DataFrame([[r, g, b, h, s, v, hex_code, color_name]], 
                            columns=["R", "G", "B", "H", "S", "V", "HEX", "Color_Name"])
    df = pd.concat([df, new_data], ignore_index=True)  # Append new data
    df.to_csv(CSV_FILE, index=False)

while True:
    # Generate a random color
    r, g, b = generate_random_color()
    
    # Convert to different color formats
    h, s, v = convert_to_hsv(r, g, b)
    hex_code = convert_to_hex(r, g, b)

    # Show the color
    show_color(r, g, b)

    # Ask user for the color name
    color_name = input(f"What was this color? (RGB: {r},{g},{b} | HSV: {h},{s},{v} | HEX: {hex_code})\n> ")

    # Save data
    save_color_data(r, g, b, h, s, v, hex_code, color_name)

    print(f"✅ Saved: {color_name} (RGB: {r},{g},{b} | HSV: {h},{s},{v} | HEX: {hex_code})\n")
