import cv2
import pytesseract
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode

def detect_image_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    color_variance = np.var(image.reshape(-1, 3), axis=0).mean()
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_peaks = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    horizontal_lines = 0
    vertical_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):
                horizontal_lines += 1
            else:
                vertical_lines += 1
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_regions, _ = cv2.connectedComponents(thresh)
    
    screenshot_score = 0
    if edge_density < 0.1:
        screenshot_score += 1

    if color_variance < 1000:
        screenshot_score += 1
    
    if hist_peaks < 10:
        screenshot_score += 1
    
    if horizontal_lines + vertical_lines > 10 and (horizontal_lines > 5 or vertical_lines > 5):
        screenshot_score += 1
    if num_regions < 1000:
        screenshot_score += 1

    contrast = np.max(gray) - np.min(gray)
    if contrast > 200:
        screenshot_score += 1
    
    border_check = np.sum(gray[0,:]) + np.sum(gray[-1,:]) + np.sum(gray[:,0]) + np.sum(gray[:,-1])
    if border_check > gray.shape[0] * gray.shape[1] * 0.8:
        screenshot_score += 1
    
    print(f"Điểm số screenshot: {screenshot_score}/7")
    
    if screenshot_score >= 4:
        return "screenshot"
    else:
        return "photo"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh, kiểm tra lại đường dẫn!")
    
    image_type = detect_image_type(image)
    print(f"Đã phát hiện loại ảnh: {image_type}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.medianBlur(gray, 3)
    
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if image_type == "photo":
        gray = correct_skew(gray)
    
    if image_type == "screenshot":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        gray = binary
    else:

        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)
    
    cv2.imshow("Processed Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("processed_image.png", gray)
    
    return gray

def correct_skew(image, delta=0.5, limit=5):
    def determine_score(arr, angle):
        data = rotate(arr, angle, resize=True, preserve=False)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    if abs(best_angle) < 0.5:
        return image
    
    print(f"Đang xoay ảnh với góc: {best_angle} độ")
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=255)
    
    return rotated

def rotate(image, angle, resize=False, preserve=False):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if resize:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        if preserve:
            pass
        else:
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)
    else:
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)

def ocr_image(image_path, lang='eng+vie+equ'):
    gray = preprocess_image(image_path)

    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(gray, lang=lang, config=config)
    
    return text

def save_text_to_file(text, output_path="output.txt"):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Văn bản đã lưu vào {output_path}")

if __name__ == "__main__":
    image_path = "img/11_.png"
    extracted_text = ocr_image(image_path)
    save_text_to_file(extracted_text)