import streamlit as st
from PIL import Image
import numpy as np
import colorsys
from scipy.ndimage import label, find_objects

# Function to convert the entire RGB image to a single value on the rainbow scale (hue)
def rgb_to_hue(image_array):
    # Normalize RGB values to [0, 1]
    normalized = image_array / 255.0
    r, g, b = normalized[..., 0], normalized[..., 1], normalized[..., 2]

    # Convert RGB to HSV (vectorized using numpy)
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val

    # Calculate hue
    hue = np.zeros_like(max_val)
    mask = diff > 0
    red_mask = (max_val == r) & mask
    green_mask = (max_val == g) & mask
    blue_mask = (max_val == b) & mask

    hue[red_mask] = ((g[red_mask] - b[red_mask]) / diff[red_mask]) % 6
    hue[green_mask] = ((b[green_mask] - r[green_mask]) / diff[green_mask]) + 2
    hue[blue_mask] = ((r[blue_mask] - g[blue_mask]) / diff[blue_mask]) + 4

    hue /= 6  # Normalize hue to range [0, 1]
    hue[hue < 0] += 1  # Ensure no negative values
    return hue

# Function to filter pixels based on the hue threshold and the largest square size around a chunk
def filter_pixels(image, threshold, max_square_size):
    img_array = np.array(image)
    hue_values = rgb_to_hue(img_array)  # Calculate hue values
    mask = hue_values < threshold  # Pixels below the threshold

    # Label connected regions
    labeled_array, num_features = label(mask)
    filtered_array = np.zeros_like(img_array)

    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        region_indices = np.argwhere(region)
        if region_indices.size > 0:
            y_min, x_min = region_indices.min(axis=0)
            y_max, x_max = region_indices.max(axis=0)
            square_size = max(y_max - y_min + 1, x_max - x_min + 1)
            if square_size <= max_square_size:  # Check if the square size meets the condition
                filtered_array[region] = img_array[region]

    return Image.fromarray(filtered_array)

# Streamlit app
st.title("Rainbow Pixel Filter")
st.write("Upload an image and use the sliders to dynamically filter pixels based on the rainbow color scale and maximum square size.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to hue array
    img_array = np.array(image)
    hue_values = rgb_to_hue(img_array)

    # Rainbow color scale slider
    threshold = st.slider(
        "Select a threshold on the rainbow scale (0 to 1):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Default value
        step=0.01,
    )

    # Maximum square size slider
    max_square_size = st.slider(
        "Select the maximum size of the largest square around connected pixel regions (in pixels):",
        min_value=1,
        max_value=1000,
        value=50,  # Default value
        step=5,
    )

    # Filter pixels based on the slider values
    filtered_image = filter_pixels(image, threshold, max_square_size)

    # Display the filtered image
    st.image(filtered_image, caption=f"Filtered Image (Threshold: {threshold}, Max Square Size: {max_square_size}px)", use_container_width=True)
