# VideoDup AI: Multi-modal Video Duplicate Detection System 🎬

## 📝 Project Overview
This is a Final Year Project (FYP) developed to detect video plagiarism using a multi-modal approach. 
The system analyzes videos across three dimensions:
- **Visual**: Perceptual Hashing (pHash)
- **Textual**: Speech-to-Text using OpenAI Whisper (Tiny model)
- **Audio**: MFCC feature extraction & Cosine Similarity

## ✨ Key Features
- **Modern UI**: Built with Streamlit for a sleek user experience.
- **Batch Processing**: Compare a base video against multiple samples at once.
- **Adjustable Weights**: Customizable sensitivity for Visual, Text, and Audio components.

## 🚀 How to Run
1. Install FFmpeg and add it to your System PATH.
2. Clone this repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
