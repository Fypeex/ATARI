import cv2

def overlay_images(img_paths, output_path="output.png"):
    # --- Load all images ---
    imgs = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in img_paths]

    # Ensure all images loaded
    if any(i is None for i in imgs):
        raise ValueError("One or more image paths are invalid.")

    # --- Resize all images to match the first one ---
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]

    # --- Initialize the base canvas ---
    result = imgs[0].copy()

    # --- Overlay each remaining image ---
    for img in imgs[1:]:
        # If image has alpha channel
        if img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                result[:, :, c] = (1 - alpha) * result[:, :, c] + alpha * img[:, :, c]
        else:
            # No alpha → simple weighted addition
            result = cv2.addWeighted(result, 0.7, img, 0.3, 0)

    # --- Save the final output ---
    cv2.imwrite(output_path, result)
    print("Saved:", output_path)


# Example usage:
image_files = [
    "./observations/episode_001_frame_00020.png",
    "./observations/episode_001_frame_00021.png",
    "./observations/episode_001_frame_00022.png",
    "./observations/episode_001_frame_00023.png",
]

overlay_images(image_files, "overlay_result.png")
