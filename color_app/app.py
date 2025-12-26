import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="色彩匹配助手 v2.0",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS 样式优化 (核心：把按钮变成色块) ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
        div.stButton > button {
            display: flex; 
            align-items: center; 
            justify-content: center;
            width: 100%;
        }
        /* 针对单选按钮组的容器进行微调 */
        div.row-widget.stRadio > div {flex-direction: row;}
    </style>
""", unsafe_allow_html=True)

# --- 3. 核心算法函数 ---

def hex_to_rgb(hex_code):
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_image(image_file):
    img = Image.open(image_file)
    img = img.convert('RGB')
    return img

def is_not_black_or_white(rgb, l_threshold_low=15, l_threshold_high=90):
    """
    过滤掉过黑(L<15)或过白(L>90)的颜色
    """
    rgb_norm = np.array(rgb) / 255.0
    # 转换为 LAB 空间获取亮度 L
    lab = color.rgb2lab(rgb_norm.reshape(1, 1, 3))[0][0]
    L = lab[0]
    return l_threshold_low < L < l_threshold_high

def extract_palette_filtered(image, k_extract=10, k_final=5, image_resize=(150, 150)):
    """提取颜色并自动过滤背景杂色"""
    img_small = image.resize(image_resize)
    img_array = np.array(img_small)
    pixels = img_array.reshape(-1, 3)
    
    # 1. 提取较多颜色
    kmeans = KMeans(n_clusters=k_extract, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # 2. 过滤黑白
    filtered_colors = []
    filtered_counts = []
    
    for i in range(len(sorted_colors)):
        if is_not_black_or_white(sorted_colors[i]):
            filtered_colors.append(sorted_colors[i])
            filtered_counts.append(sorted_counts[i])
    
    # 如果过滤太狠导致没颜色了，就回退
    if len(filtered_colors) == 0:
        return sorted_colors[:k_final], sorted_counts[:k_final]
        
    return np.array(filtered_colors[:k_final]), np.array(filtered_counts[:k_final])

def calculate_similarity_ciede2000(rgb1, rgb2):
    color1_norm = np.array(rgb1) / 255.0
    color2_norm = np.array(rgb2) / 255.0
    color1_lab = color.rgb2lab(color1_norm.reshape(1, 1, 3))
    color2_lab = color.rgb2lab(color2_norm.reshape(1, 1, 3))
    delta_e = color.deltaE_ciede2000(color1_lab, color2_lab)[0][0]
    similarity = 100 / (1 + 0.1 * delta_e)**2
    return delta_e, similarity

def display_color_compact(rgb, label="", height=40, show_hex=True):
    hex_color = rgb_to_hex(rgb)
    text_color = '#000' if sum(rgb) > 382 else '#fff'
    hex_text = f" {hex_color.upper()}" if show_hex else ""
    st.markdown(
        f"""
        <div style="
            background-color: {hex_color};
            width: 100%;
            height: {height}px;
            border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            color: {text_color}; font-family: monospace;
            border: 1px solid rgba(0,0,0,0.1); margin-bottom: 5px;
        "><b>{label}{hex_text}</b></div>
        """, unsafe_allow_html=True
    )

# --- 4. 主界面逻辑 ---

# Session State 初始化
if 'selected_color_index' not in st.session_state:
    st.session_state.selected_color_index = 0

st.title("🎨 色彩匹配助手 (v2.0 新版)")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="medium")

# === 左侧：标准色 ===
with col_left:
    st.subheader("1. 设定标准色")
    tab1, tab2 = st.tabs(["🔢 输入色值", "🖼️ 从图片提取"])
    target_rgb = None
    
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1: hex_input = st.text_input("HEX", "#3366FF", label_visibility="collapsed")
        with c2: st.write("")
        rgb_res = hex_to_rgb(hex_input)
        if rgb_res:
            target_rgb = rgb_res
            display_color_compact(target_rgb, "标准色", 50)
            
    with tab2:
        up_t = st.file_uploader("上传标准图", type=["jpg","png","jpeg"], key="t")
        if up_t:
            img_t = load_image(up_t)
            # 标准色通常比较纯，稍微放宽过滤或者不过滤，这里简单取主色
            tc, _ = extract_palette_filtered(img_t, k_final=1) 
            target_rgb = tc[0]
            cc1, cc2 = st.columns([2,1])
            with cc1: display_color_compact(target_rgb, "提取结果", 50)
            with cc2: st.image(img_t, width=80)

# === 右侧：实物图 (核心交互升级) ===
with col_right:
    st.subheader("2. 上传实物图")
    st.caption("支持截图粘贴保存后上传")
    up_s = st.file_uploader("上传实物图", type=["jpg","png","jpeg"], key="s")
    
    selected_sample_rgb = None
    
    if up_s:
        img_s = load_image(up_s)
        
        # 1. 提取并过滤 (剔除黑白)
        with st.spinner("正在智能提取颜色..."):
            palette, counts = extract_palette_filtered(img_s, k_extract=10, k_final=5)
            total = sum(counts)
            
        if st.session_state.selected_color_index >= len(palette):
            st.session_state.selected_color_index = 0
            
        # 2. 布局
        ic1, ic2 = st.columns([1, 2])
        with ic1:
            st.image(img_s, caption="实物", use_container_width=True)
            
        with ic2:
            st.caption("🎨 点击色块选择主色 (已过滤黑白):")
            
            # --- 动态生成色块按钮 ---
            cols = st.columns(len(palette))
            for i, color_val in enumerate(palette):
                with cols[i]:
                    hex_c = rgb_to_hex(color_val)
                    is_sel = (i == st.session_state.selected_color_index)
                    percent = (counts[i]/total)*100
                    
                    # CSS 魔法：把按钮变成有颜色的块
                    # 注意：这里使用了 nth-of-type 来增强稳定性
                    btn_css = f"""
                    <style>
                        div[data-testid="stHorizontalBlock"] .stButton:nth-of-type({i+1}) button {{
                            background-color: {hex_c} !important;
                            color: {'#000' if sum(color_val)>382 else '#fff'} !important;
                            border: {'3px solid #FF4B4B' if is_sel else '1px solid #ddd'} !important;
                            height: 45px;
                        }}
                    </style>
                    """
                    st.markdown(btn_css, unsafe_allow_html=True)
                    
                    if st.button(f"{percent:.0f}%", key=f"c_{i}"):
                        st.session_state.selected_color_index = i
                        st.rerun()
            
            if len(palette) > 0:
                selected_sample_rgb = palette[st.session_state.selected_color_index]
            
            # 底部辅助提示
            st.caption(f"当前选中: {rgb_to_hex(selected_sample_rgb).upper()}")

# === 底部：对比结果 ===
st.markdown("---")
if target_rgb is not None and selected_sample_rgb is not None:
    de, sim = calculate_similarity_ciede2000(target_rgb, selected_sample_rgb)
    
    st.markdown(f"### 🎯 匹配度: :rainbow[{sim:.1f}%]")
    st.progress(sim/100)
    
    r1, r2, r3, r4 = st.columns([1.5, 1.5, 1, 2])
    with r1: display_color_compact(target_rgb, "标准", 60)
    with r2: display_color_compact(selected_sample_rgb, "实物", 60)
    with r3: st.metric("ΔE", f"{de:.2f}")
    with r4: 
        if de<2: st.success("✅ 完美匹配")
        elif de<5: st.warning("⚠️ 轻微色差")
        else: st.error("❌ 差异明显")
