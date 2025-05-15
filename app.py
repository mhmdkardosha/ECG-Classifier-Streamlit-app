import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib


def extract_ecg_points_from_image(image_pil, debug_mode=False):
    """
    Extract ECG waveform points from an uploaded image with improved zero amplitude detection.

    Args:
        image_pil (PIL.Image): Image uploaded by the user.
        debug_mode (bool): If True, display intermediate processing steps.

    Returns:
        np.array: Array of 187 data points, or None if extraction fails.
    """
    # 1. Convert PIL Image to OpenCV format
    image_cv = np.array(image_pil.convert('RGB'))
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    # Save original dimensions for visualization
    original_height, original_width = gray_image.shape[:2]

    try:
        # 2. Enhanced preprocessing with better noise handling
        # Apply Gaussian blur with adaptive kernel size based on image dimensions
        kernel_size = max(
            3, min(5, int(min(original_width, original_height) / 100)))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + \
            1  # Ensure odd number

        blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

        # CLAHE for better contrast in low-amplitude areas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Better thresholding with Otsu's method if adaptive threshold is too strict
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Backup threshold using Otsu's method for comparison
        _, otsu_thresh = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Choose threshold method based on which one captures more signal
        if cv2.countNonZero(adaptive_thresh) > cv2.countNonZero(otsu_thresh):
            thresh = adaptive_thresh
        else:
            thresh = otsu_thresh

        if debug_mode:
            st.image([gray_image, enhanced, thresh],
                     caption=["Original", "Enhanced", "Threshold"],
                     width=200)

        # 3. Improved grid removal - adaptive kernel size based on image size
        h_size = max(15, int(original_width / 30))
        v_size = max(15, int(original_height / 30))

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (h_size, 1))
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, v_size))

        h_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        v_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        grid_mask = cv2.bitwise_or(h_lines, v_lines)
        no_grid = cv2.bitwise_and(thresh, cv2.bitwise_not(grid_mask))

        # 4. Better cleanup for preserving small signal variations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(
            no_grid, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Use smaller kernel for median blur to preserve details
        cleaned = cv2.medianBlur(cleaned, 3)

        if debug_mode:
            st.image([grid_mask, no_grid, cleaned],
                     caption=["Grid Mask", "No Grid", "Cleaned"],
                     width=200)

        # 5. Improved contour detection and selection
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            # Try again with less preprocessing if no contours found
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                st.warning(
                    "No ECG waveform detected. The image may not contain a clear waveform.")
                return None

        # Handle contour selection more intelligently
        # Instead of just largest contours, consider horizontal coverage
        x_coverage = {}
        for i, c in enumerate(contours):
            x, _, w, _ = cv2.boundingRect(c)
            x_coverage[i] = (x, x+w)  # Store x-range covered by contour

        # Sort contours by area first
        area_sorted = sorted([(i, cv2.contourArea(contours[i])) for i in range(len(contours))],
                             key=lambda x: x[1], reverse=True)

        # Get top contours but ensure good x-coverage
        selected_indices = []
        current_coverage = set()

        # First include the largest contours
        largest_indices = [idx for idx,
                           _ in area_sorted[:min(5, len(area_sorted))]]
        for idx in largest_indices:
            selected_indices.append(idx)
            x_min, x_max = x_coverage[idx]
            current_coverage.update(range(x_min, x_max+1))

        # Add more contours to improve coverage if needed
        remaining_indices = [idx for idx,
                             _ in area_sorted[5:min(15, len(area_sorted))]]
        for idx in remaining_indices:
            x_min, x_max = x_coverage[idx]
            new_points = set(range(x_min, x_max+1)) - current_coverage
            if len(new_points) > original_width / 20:  # Only add if significant new coverage
                selected_indices.append(idx)
                current_coverage.update(new_points)

        selected_contours = [contours[i] for i in selected_indices]

        if debug_mode:
            # Draw selected contours on image for visualization
            contour_img = cv2.cvtColor(cleaned.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(
                contour_img, selected_contours, -1, (0, 255, 0), 2)
            st.image(contour_img, caption="Selected Contours", width=400)

        # 6. Improved points extraction with better handling of sparse areas
        try:
            all_points = np.vstack([c.reshape(-1, 2)
                                   for c in selected_contours])
        except ValueError:
            # Fall back to all contours if selected contours are insufficient
            if contours:
                all_points = np.vstack([c.reshape(-1, 2) for c in contours])
            else:
                st.warning("Could not extract sufficient contour points.")
                return None

        # 7. Better sorting with handling of points at same x-coordinate
        # Group points by x-coordinate and select best y for each x
        points_by_x = {}
        for point in all_points:
            x, y = point
            if x not in points_by_x:
                points_by_x[x] = []
            points_by_x[x].append(y)

        # For each x with multiple y values, keep the one closest to median
        unique_points = []
        for x, y_values in points_by_x.items():
            if len(y_values) == 1:
                unique_points.append([x, y_values[0]])
            else:
                # For multiple y values at same x, use median approach for robustness
                median_y = np.median(y_values)
                best_y = min(y_values, key=lambda y: abs(y - median_y))
                unique_points.append([x, best_y])

        points_array = np.array(sorted(unique_points, key=lambda p: p[0]))

        if len(points_array) < 10:  # Reduced minimum requirement
            st.warning(
                "Too few unique points detected. The image may not contain a clear ECG waveform.")
            return None

        x_points = points_array[:, 0]
        y_points = points_array[:, 1]

        # 8. Better handling of horizontal spread
        x_min, x_max = np.min(x_points), np.max(x_points)
        if (x_max - x_min) <= max(5, original_width / 100):  # Adaptive threshold
            st.warning(
                "Insufficient horizontal spread in the detected waveform.")
            return None

        # 9. Generate evenly spaced sample points
        sampled_x = np.linspace(x_min, x_max, 187)
        sampled_y = np.zeros(187)

        # 10. Improved interpolation for better zero amplitude detection
        # Use scipy's interp1d for more accurate interpolation
        from scipy import interpolate

        # First, ensure x_points are strictly increasing (required for interp1d)
        unique_x_indices = np.unique(x_points, return_index=True)[1]
        unique_x_points = x_points[unique_x_indices]
        unique_y_points = y_points[unique_x_indices]

        if len(unique_x_points) >= 4:  # Need at least 4 points for cubic interpolation
            # Try cubic interpolation first
            try:
                interp_func = interpolate.interp1d(unique_x_points, unique_y_points,
                                                   kind='cubic', bounds_error=False,
                                                   fill_value=(unique_y_points[0], unique_y_points[-1]))
                sampled_y = interp_func(sampled_x)
            except:
                # Fallback to linear interpolation if cubic fails
                interp_func = interpolate.interp1d(unique_x_points, unique_y_points,
                                                   kind='linear', bounds_error=False,
                                                   fill_value=(unique_y_points[0], unique_y_points[-1]))
                sampled_y = interp_func(sampled_x)
        else:
            # Manual linear interpolation for very sparse points
            for i, target_x in enumerate(sampled_x):
                # Find closest points
                distances = np.abs(x_points - target_x)
                closest_idx = np.argmin(distances)

                if distances[closest_idx] < 1:  # If very close point exists
                    sampled_y[i] = y_points[closest_idx]
                else:
                    # Find points on either side for interpolation
                    left_indices = np.where(x_points < target_x)[0]
                    right_indices = np.where(x_points > target_x)[0]

                    if len(left_indices) == 0:
                        sampled_y[i] = y_points[right_indices[0]] if len(
                            right_indices) > 0 else y_points[closest_idx]
                    elif len(right_indices) == 0:
                        sampled_y[i] = y_points[left_indices[-1]
                                                ] if len(left_indices) > 0 else y_points[closest_idx]
                    else:
                        # Use closest points on either side
                        left_idx = left_indices[-1]
                        right_idx = right_indices[0]

                        # Linear interpolation
                        x_left, y_left = x_points[left_idx], y_points[left_idx]
                        x_right, y_right = x_points[right_idx], y_points[right_idx]

                        # Safe division
                        x_diff = x_right - x_left
                        if abs(x_diff) > 1e-10:  # Avoid division by very small numbers
                            weight = (target_x - x_left) / x_diff
                            sampled_y[i] = y_left * \
                                (1 - weight) + y_right * weight
                        else:
                            sampled_y[i] = y_left

        # 11. Improved normalization with better zero-amplitude preservation
        # In image coordinates, y increases downward, so invert
        sampled_y = np.max(sampled_y) - sampled_y

        # Normalize to [0, 1] range with better handling of flat signals
        y_min, y_max = np.min(sampled_y), np.max(sampled_y)
        y_range = y_max - y_min

        if y_range > 1e-6:  # Non-zero amplitude
            normalized_y = (sampled_y - y_min) / y_range
        else:
            # For nearly flat signals, center them at 0.5 instead of returning None
            st.warning(
                "Very low amplitude variation detected. Results may be unreliable.")
            if abs(y_min) < 1e-6:  # If already close to zero
                normalized_y = np.full(187, 0.5)  # Center flat line
            else:
                # Just normalize position but preserve flatness
                normalized_y = np.full(187, 0.5)

        # 12. Add signal quality check
        signal_quality = min(1.0, max(0.1, y_range / (original_height * 0.1)))
        if signal_quality < 0.3:
            st.warning(
                f"Low signal quality detected ({signal_quality:.2f}). Results may be unreliable.")

        if debug_mode:
            # Show the extracted signal
            fig, ax = plt.subplots()
            ax.plot(sampled_x, sampled_y)
            ax.set_title("Extracted Signal (Before Normalization)")
            st.pyplot(fig)

        return normalized_y

    except Exception as e:
        st.error(f"Error extracting ECG data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # More detailed error info
        return None


def load_my_model():
    """
    Placeholder for loading your pre-trained classification model.
    """
    model = joblib.load('../models/heartbeat_model.pkl')
    return model


model = load_my_model()
class_names = ["Normal beat (N)", "Supraventricular premature beat (S)",
               "Premature ventricular contraction (V)", "Fusion of ventricular and normal beat (F)",
               "Unclassifiable beat (Q)"]

# --- Streamlit App Interface ---
st.title("ECG Heart Rate Condition Predictor")

st.write("""
Upload an image of a single ECG period. The app will attempt to extract
the waveform data and predict the heart rate condition.
**Note:** The image processing part is highly complex and this is a conceptual demo.
""")

uploaded_file = st.file_uploader(
    "Choose an ECG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)

    if st.button("Analyze ECG"):
        with st.spinner("Processing image and extracting data..."):
            ecg_data_points = extract_ecg_points_from_image(image)

        if ecg_data_points is not None and len(ecg_data_points) == 187:
            st.success("ECG data extracted successfully!")
            st.subheader("Extracted ECG Data (Sampled Points)")
            st.line_chart(ecg_data_points)

            data_for_model = np.array(ecg_data_points).reshape(1, -1)

            with st.spinner("Predicting condition..."):
                probabilities = model.predict_proba(data_for_model)
                predicted_class_index = np.argmax(probabilities, axis=1)[0]
                predicted_class_name = class_names[predicted_class_index]
                st.subheader("Prediction")
                st.write(f"Predicted Condition: **{predicted_class_name}**")
                st.write("Probabilities:")
                for i, class_name in enumerate(class_names):
                    st.write(f"- {class_name}: {probabilities[0][i]:.4f}")

        else:
            st.error("Could not extract 187 data points from the image. Please ensure the image is clear and shows a single ECG period. The image processing part might need more advanced techniques.")

st.sidebar.header("About")
st.sidebar.info("""
This app is a conceptual demonstration for predicting heart rate conditions
from an ECG image. The core challenge lies in accurately converting the
ECG image into a numerical data series.
The five classes are:
- Normal beat (N)
- Supraventricular premature beat (S)
- Premature ventricular contraction (V)
- Fusion of ventricular and normal beat (F)
- Unclassifiable beat (Q)
""")
