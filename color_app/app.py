import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="色彩匹配助手 v2.1",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS 样式优化 ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
        div.stButton > button {
            display: flex; 
            align-items: center; 
            justify-content: center;
            width: 100%;
            font-size: 16px !important; 
        }
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
    """过滤掉过黑(L<15)或过白(L>90)的颜色"""
    rgb_norm = np.array(rgb) / 255.0
    lab = color.rgb2lab(rgb_norm.reshape(1, 1, 3))[0][0]
    L = lab[0]
    return l_threshold_low < L < l_threshold_high

def extract_palette_filtered(image, k_extract=10, k_final=5, image_resize=(150, 150)):
    """提取颜色并自动过滤背景杂色"""
    img_small = image.resize(image_resize)
    img_array = np.array(img_small)
    pixels = img_array.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k_extract, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    filtered_colors = []
    filtered_counts = []
    
    for i in range(len(sorted_colors)):
        if is_not_black_or_white(sorted_colors[i]):
            filtered_colors.append(sorted_colors[i])
            filtered_counts.append(sorted_counts[i])
    
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

if 'selected_color_index' not in st.session_state:
    st.session_state.selected_color_index = 0

st.title("🎨 色彩匹配助手 (v2.1)")
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
            tc, _ = extract_palette_filtered(img_t, k_final=1) 
            target_rgb = tc[0]
            cc1, cc2 = st.columns([2,1])
            with cc1: display_color_compact(target_rgb, "提取结果", 50)
            with cc2: st.image(img_t, width=80)

# === 右侧：实物图 ===
with col_right:
    st.subheader("2. 上传实物图")
    st.caption("支持截图粘贴保存后上传")
    up_s = st.file_uploader("上传实物图", type=["jpg","png","jpeg"], key="s")
    
    selected_sample_rgb = None
    
    if up_s:
        img_s = load_image(up_s)
        with st.spinner("正在智能提取颜色..."):
            palette, counts = extract_palette_filtered(img_s, k_extract=10, k_final=2)
            total = sum(counts)
            
        if st.session_state.selected_color_index >= len(palette):
            st.session_state.selected_color_index = 0
            
        ic1, ic2 = st.columns([1, 2])
        with ic1:
            st.image(img_s, caption="实物", use_container_width=True)
            
        with ic2:
            st.caption("🎨 点击色块选择主色 (Top 2):")
            cols = st.columns(len(palette), gap="medium")
            for i, color_val in enumerate(palette):
                with cols[i]:
                    hex_c = rgb_to_hex(color_val)
                    is_sel = (i == st.session_state.selected_color_index)
                    percent = (counts[i]/total)*100
                    
                    icon = "◉" if is_sel else "○"
                    label_text = f"{icon} {percent:.0f}%"

                    btn_css = f"""
                    <style>
                        div[data-testid="stHorizontalBlock"] .stButton:nth-of-type({i+1}) button {{
                            background-color: {hex_c} !important;
                            color: {'#000' if sum(color_val)>382 else '#fff'} !important;
                            border: {'4px solid #FF0000' if is_sel else '1px solid #ddd'} !important;
                            height: 50px;
                            box-shadow: {'0 6px 12px rgba(0,0,0,0.2)' if is_sel else 'none'} !important;
                            transform: {'scale(1.05)' if is_sel else 'scale(1)'} !important;
                            transition: all 0.2s ease-in-out !important;
                        }}
                    </style>
                    """
                    st.markdown(btn_css, unsafe_allow_html=True)
                    
                    if st.button(label_text, key=f"c_{i}"):
                        st.session_state.selected_color_index = i
                        st.rerun()
            
            if len(palette) > 0:
                selected_sample_rgb = palette[st.session_state.selected_color_index]
            
            st.caption(f"当前选中: {rgb_to_hex(selected_sample_rgb).upper()}")

# === 底部：对比结果 ===
st.markdown("---")
if target_rgb is not None and selected_sample_rgb is not None:
    de, sim = calculate_similarity_ciede2000(target_rgb, selected_sample_rgb)
    
    st.markdown(f"### 🎯 匹配度: :rainbow[{sim:.1f}%]")
    
    # 进度条颜色逻辑：根据新的标准调整
    bar_color = "green" if sim >= 60 else ("orange" if sim >= 40 else "red")
    st.progress(sim/100)
    
    r1, r2, r3, r4 = st.columns([1.5, 1.5, 1, 2])
    with r1: display_color_compact(target_rgb, "标准", 60)
    with r2: display_color_compact(selected_sample_rgb, "实物", 60)
    
    # 修改点：增加 Help 解释
    with r3: 
        st.metric(
            "色差 (ΔE)", 
            f"{de:.2f}", 
            help="ΔE (Delta E) 是国际标准色差值，数值越小越好。\n\n• 0-2: 人眼几乎无法分辨\n• 2-6: 属于相似色系\n• >6: 颜色差异明显"
        )
        
    # 修改点：调整评判文案
    with r4: 
        if sim >= 80: 
            st.success("🌟 完美匹配 (Perfect)")
        elif sim >= 60:
            st.info("✅ 高度相似 (Very Similar)")
        elif sim >= 40:
            st.warning("🆗 基本近似 (Similar)")
        else:
            st.error("❌ 差异明显 (Different)")
            
    # 底部补充说明
    st.caption("""
    📝 **关于 ΔE (色差值) 的说明：** ΔE 是衡量两个颜色在人眼感知上差距的单位。  
    - 当 **ΔE < 2.0** 时，普通人肉眼几乎看不出区别。  
    - 当 **ΔE 在 2.0 - 6.0** 之间时，颜色看起来很像，属于同一色系。  
    - 我们设定只要 **相似度 > 40%** (即 ΔE 约为 6 左右)，即可认为是“近似色”。
    """)
