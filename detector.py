import cv2
import imagehash
import whisper
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 初始化语音模型 (建议初次使用 tiny 以防电脑卡顿)
print("正在初始化 AI 模型...")
audio_model = whisper.load_model("tiny") 

def get_video_features(video_path):
    print(f"\n正在分析视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    hashes = []
    frame_count = 0
    
    # --- 1. 视觉抽帧 ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 30 == 0: # 每秒抽一帧
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hashes.append(imagehash.phash(img))
        frame_count += 1
    cap.release()
    
    # --- 2. 台词提取 ---
    result = audio_model.transcribe(video_path, language="zh")
    return hashes, result['text']

def run_analysis(path1, path2):
    # 获取两个视频的特征
    h1, text1 = get_video_features(path1)
    h2, text2 = get_video_features(path2)
    
    # 计算视觉相似度 (取平均汉明距离)
    visual_sim = 1 - (sum([abs(a - b) for a, b in zip(h1, h2)]) / (len(h1) * 64))
    
    # 计算台词相似度 (余弦相似度)
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    script_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    
    print("\n" + "="*30)
    print(f"检测报告")
    print(f"视觉相似度: {visual_sim:.2%}")
    print(f"台词相似度: {script_sim:.2%}")
    
    # 综合判定逻辑
    if visual_sim > 0.8:
        print("结论: 判定为【画面直接搬运】")
    elif script_sim > 0.8:
        print("结论: 判定为【剧本高度洗稿/模仿】")
    else:
        print("结论: 两个视频具有原创性差异")
    print("="*30)

if __name__ == "__main__":
    # 物理操作：请确保文件夹里有这两个 mp4 文件
    run_analysis("video1.mp4", "video2.mp4")