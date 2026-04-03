import streamlit as st
import cv2
import imagehash
from PIL import Image
import whisper
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import os
import pandas as pd
import shutil
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ==========================================
# --- 1. 页面高级配置 ---
# ==========================================
st.set_page_config(
    page_title="VideoDup AI - 视频原创度检测",
    page_icon="🎬",
    layout="wide"
)

# 自定义 CSS 提升 UI 质感
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 10px; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# --- 2. 核心算法函数 (带缓存优化) ---
# ==========================================
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

def get_features(video_path, _model):
    """多模态特征提取"""
    # 1. 视觉特征 (每2秒采样)
    hashes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % int(fps * 2 if fps > 0 else 2) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hashes.append(imagehash.phash(img))
        count += 1
    cap.release()

    # 2. 台词特征 (全量识别)
    try:
        result = _model.transcribe(video_path, language=None)
        text = result.get('text', "").strip()
        lang = result.get('language', "unknown")
    except:
        text, lang = "", "unknown"

    # 3. 音频指纹 (全量识别)
    try:
        y, sr = librosa.load(video_path, sr=22050, duration=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_fp = np.mean(mfcc.T, axis=0)
    except:
        audio_fp = None

    return hashes, text, audio_fp, lang

def calculate_similarity(h1, t1, a1, h2, t2, a2):
    """计算相似度"""
    v_sim, t_sim, a_sim = 0.0, 0.0, 0.0
    
    # 视觉对比
    if h1 and h2:
        sims = [1 - (h1[i] - h2[i]) / 64.0 for i in range(min(len(h1), len(h2)))]
        v_sim = sum(sims) / len(sims) if sims else 0
    
    # 文本对比
    if len(t1) > 2 and len(t2) > 2:
        try:
            tfidf = TfidfVectorizer().fit_transform([t1, t2])
            t_sim = (tfidf * tfidf.T).A[0, 1]
        except: t_sim = 0
        
    # 音频对比
    if a1 is not None and a2 is not None:
        a_sim = float(cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))[0][0])
        
    return v_sim, t_sim, a_sim

# ==========================================
# --- 3. UI 界面布局 ---
# ==========================================
st.title("🎬 VideoDup AI: 多模态视频重复度检测")
st.caption("本系统通过对视频的【像素哈希】、【语义文本】及【音频指纹】进行三维建模，精准识别搬运与洗稿。")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 权重配置")
    w_v = st.slider("画面权重 (Visual)", 0.0, 1.0, 0.4)
    w_t = st.slider("台词权重 (Text)", 0.0, 1.0, 0.4)
    w_a = st.slider("BGM权重 (Audio)", 0.0, 1.0, 0.2)
    st.divider()
    st.info("💡 提示：如果权重之和不等于 1.0，系统会自动进行归一化处理。")

# 文件上传区
col_a, col_b = st.columns(2)
with col_a:
    base_file = st.file_uploader("📤 上传【基准原创视频】", type=["mp4", "mov"])
with col_b:
    sample_files = st.file_uploader("📂 上传【待测样本库】(支持多选)", type=["mp4", "mov"], accept_multiple_files=True)

if base_file and sample_files:
    if st.button("🚀 开始深度比对分析", type="primary", use_container_width=True):
        # 创建临时目录
        tmp = "temp_run"
        if os.path.exists(tmp): shutil.rmtree(tmp)
        os.makedirs(tmp)
        
        # 预加载模型
        ai_model = load_model()
        
        # 保存并分析基准视频
        b_path = os.path.join(tmp, base_file.name)
        with open(b_path, "wb") as f: f.write(base_file.read())
        
        with st.status("正在进行特征建模...", expanded=True) as status:
            st.write("正在提取基准视频特征...")
            h_b, t_b, a_b, l_b = get_features(b_path, ai_model)
            st.write(f"✅ 基准视频分析完成 (语言检测: {l_b.upper()})")
            
            results = []
            for i, s_file in enumerate(sample_files):
                st.write(f"正在处理样本 ({i+1}/{len(sample_files)}): {s_file.name}")
                s_path = os.path.join(tmp, s_file.name)
                with open(s_path, "wb") as f: f.write(s_file.read())
                
                h_s, t_s, a_s, _ = get_features(s_path, ai_model)
                v_s, t_sim, a_sim = calculate_similarity(h_b, t_b, a_b, h_s, t_s, a_s)
                
                # 计算总分 (确保权重归一化)
                total_w = w_v + w_t + w_a
                final_score = (v_s * w_v + t_sim * w_t + a_sim * w_a) / total_w
                
                results.append({
                    "文件名": s_file.name,
                    "画面相似度": v_s,
                    "台词相似度": t_sim,
                    "BGM相似度": a_sim,
                    "综合风险分": final_score
                })
            status.update(label="🎉 检测报告生成完毕！", state="complete", expanded=False)

        # --- 结果展示区 ---
        st.divider()
        df = pd.DataFrame(results).sort_values(by="综合风险分", ascending=False)
        
        # 第一排：核心指标卡片
        m1, m2, m3 = st.columns(3)
        highest = df.iloc[0]
        m1.metric("总检测样本数", len(results))
        m2.metric("最高相似度", f"{highest['综合风险分']:.1%}", highest['文件名'])
        m3.metric("检测语种", l_b.upper())

        # 第二排：详细数据表与图表
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.subheader("📋 详细相似度清单")
            # 格式化显示
            df_styled = df.copy()
            for col in ["画面相似度", "台词相似度", "BGM相似度", "综合风险分"]:
                df_styled[col] = df_styled[col].map('{:.1%}'.format)
            st.table(df_styled)
        
        with col_res2:
            st.subheader("📊 风险分布图")
            st.bar_chart(df.set_index("文件名")["综合风险分"])

        # 第三排：风险判定
        st.subheader("🚩 重点关注对象")
        danger = df[df["综合风险分"] > 0.6]
        if not danger.empty:
            for _, row in danger.iterrows():
                st.warning(f"**{row['文件名']}** 存在高重复风险！综合得分: {row['综合风险分']:.1%}")
        else:
            st.success("所有样本原创度良好。")

        # 清理
        shutil.rmtree(tmp)
else:
    st.info("👋 欢迎使用 VideoDup AI！请在上方上传视频以启动多模态检测。")
    st.image("https://img.icons8.com/clouds/500/video-playlist.png", width=250)