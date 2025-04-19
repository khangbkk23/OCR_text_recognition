# Advanced OCR Text Recognition System

## Overview
A sophisticated Python-based OCR tool that automatically detects image types (photos/screenshots) and applies optimal preprocessing before text extraction using Tesseract OCR. Supports multi-language text recognition including English and Vietnamese.

## Key Features
- Intelligent image type detection (photo vs screenshot)
- Adaptive preprocessing pipeline
- Automatic skew correction
- Multi-language support (English, Vietnamese, equations)
- Advanced image enhancement techniques:
  - CLAHE contrast enhancement
  - Adaptive thresholding
  - Noise reduction

## Requirements
- OpenCV (`pip install opencv-python`)
- Tesseract OCR (install system-wide)
- pytesseract (`pip install pytesseract`)
- scikit-image (`pip install scikit-image`)
- numpy (`pip install numpy`)
