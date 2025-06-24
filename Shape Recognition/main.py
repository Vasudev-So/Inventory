import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import tempfile

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]

    denominator = math.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
    if denominator < 1e-5:
        return 1.0  # Avoid division by near-zero, assume sharp angle

    cosine_angle = (dx1 * dx2 + dy1 * dy2) / denominator
    return max(min(cosine_angle, 1.0), -1.0)  # Clamp to [-1, 1]



def classify_polygon(approx):
    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"

    elif num_vertices == 4:
        pts = approx.reshape(4, 2)
        dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]

        angles = []
        for i in range(4):
            cos_val = angle(pts[i - 1], pts[(i + 1) % 4], pts[i])
            # Clamp cos_val to valid range for acos
            cos_val = max(min(cos_val, 1.0), -1.0)
            try:
                angle_deg = math.degrees(math.acos(cos_val))
            except ValueError:
                angle_deg = 90  # fallback safe value
            angles.append(angle_deg)

        if all(abs(a - 90) < 10 for a in angles):
            if np.std(dists) < 10:
                return "Square"
            else:
                return "Rectangle"
        elif np.std(dists) < 10:
            return "Rhombus"
        elif any(abs(a - 90) > 15 for a in angles):
            return "Parallelogram"
        else:
            return "Trapezium"

    elif num_vertices == 5:
        return "Pentagon"
    elif num_vertices == 6:
        return "Hexagon"
    elif num_vertices > 6:
        return "Circle"
    else:
        return "Unknown"


def detect_shapes(image, filter_shape=None):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        shape = classify_polygon(approx)

        if filter_shape is None or shape.lower() == filter_shape.lower():
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            cv2.putText(img, shape, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

# ----- Streamlit UI -----

st.set_page_config(page_title="Shape Detection App", layout="centered")
st.title("🔺🔶 Shape Detection App")
st.markdown("Upload an image with shapes. It will detect common and complex polygons.")

uploaded_file = st.file_uploader("📤 Upload a shape image (e.g., PNG/JPG)", type=["png", "jpg", "jpeg"])
filter_shape = st.text_input("🔍 Enter shape to highlight only (optional)", placeholder="e.g., Triangle, Rhombus")

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    with st.spinner("Detecting shapes..."):
        result_img = detect_shapes(img, filter_shape if filter_shape else None)

    st.subheader("✅ Detected Shapes")
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")
