import cv2
import pytesseract
import os

# Ensure Tesseract is installed and set the path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract.exe' (for Windows)

def preprocess_image(image_path):
    """Preprocess the image for better text segmentation."""
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    return img, dilated

def segment_words(dilated, original_img, output_dir="words"):
    """Detect and save word segments."""
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word_count = 0
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter small regions (noise)
        if w > 30 and h > 30:
            word_count += 1
            # Crop and save each word
            word = original_img[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/word_{word_count}.png", word)

    print(f"Words saved in '{output_dir}' directory.")

def extract_text_from_words(output_dir="words"):
    """Extract text from saved word images."""
    words = []
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(".png"):
            word_path = os.path.join(output_dir, file)
            text = pytesseract.image_to_string(word_path, lang="eng", config="--psm 8")
            words.append(text.strip())
    return words

if __name__ == "__main__":
    image_path = "handwritten_text.jpg"  # Replace with your image path

    # Step 1: Preprocess the image
    original_img, dilated = preprocess_image(image_path)

    # Step 2: Segment words
    segment_words(dilated, original_img)

    # Step 3: Extract text from each word
    words = extract_text_from_words()
    print("Extracted Words:", words)
