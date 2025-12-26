import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="色彩匹配助手",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS 样式优化 ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
        /* 让按钮里的文字垂直居中 */
        div.stButton > button {
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. 核心算法函数 ---

def hex_to_rgb(hex_code):
    """HEX 转 RGB"""
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None

def rgb_to_hex(rgb):
    """RGB 转 HEX"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_image(image_file):
    """加载图片"""
    img = Image.open(image_file)
    img = img.convert('RGB')
    return img

def is_not_black_or_white(rgb, l_threshold_low=10, l_threshold_high=92):
    """
    判断颜色是否不是黑色或白色。
    原理：将 RGB 转换为 CIELAB 色彩空间，检查 L (亮度) 分量。
    L 的范围是 0 (纯黑) 到 100 (纯白)。
    默认剔除 L < 10 和 L > 92 的颜色。
    """
    # RGB 转 LAB 需要 [0, 1] 范围的 float 输入
    rgb_norm = np.array(rgb) / 255.0
    # skimage要求输入是 (M, N, 3) 的形状
    lab = color.rgb2lab(rgb_norm.reshape(1, 1, 3))[0][0]
    L = lab[0]
    # 如果亮度在阈值范围内，则认为不是黑白色，返回 True
    return l_threshold_low < L < l_threshold_high

def extract_palette_filtered(image, k_extract=8, k_final=5, image_resize=(150, 150)):
    """
    提取并过滤颜色。
    1. 先提取较多颜色 (k_extract)。
    2. 过滤掉黑白色。
    3. 返回占比最高的 k_final 个颜色。
    """
    img_small = image.resize(image_resize)
    img_array = np.array(img_small)
    pixels = img_array.reshape(-1, 3)
    
    # 1. 初始聚类提取
    kmeans = KMeans(n_clusters=k_extract, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # 按占比排序
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # 2. 过滤黑白色
    filtered_colors = []
    filtered_counts = []
    
    for i in range(len(sorted_colors)):
        if is_not_black_or_white(sorted_colors[i]):
            filtered_colors.append(sorted_colors[i])
            filtered_counts.append(sorted_counts[i])
            
    # 如果过滤后颜色不足，回退到使用原始结果的前几个，避免报错
    if len(filtered_colors) == 0:
        st.warning("⚠️ 未能提取到足够的彩色，已显示原始颜色。")
        return sorted_colors[:k_final], sorted_counts[:k_final]
        
    # 3. 截取最终需要的数量
    final_colors = np.array(filtered_colors[:k_final])
    final_counts = np.array(filtered_counts[:k_final])
    
    return final_colors, final_counts

def calculate_similarity_ciede2000(rgb1, rgb2):
    """计算 CIEDE2000 色差"""
    color1_norm = np.array(rgb1) / 255.0
    color2_norm = np.array(rgb2) / 255.0
    color1_lab = color.rgb2lab(color1_norm.reshape(1, 1, 3))
    color2_lab = color.rgb2lab(color2_norm.reshape(1, 1, 3))
    delta_e = color.deltaE_ciede2000(color1_lab, color2_lab)[0][0]
    similarity = 100 / (1 + 0.1 * delta_e)**2
    return delta_e, similarity

def display_color_compact(rgb, label="", height=40, show_hex=True):
    """显示紧凑的颜色条组件 (用于展示结果)"""
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
            display: flex;
            align-items: center;
            justify-content: center;
            color: {text_color};
            font-size: 0.9em;
            font-family: monospace;
            border: 1px solid rgba(0,0,0,0.1);
            margin-bottom: 5px;
        ">
            <b>{label}{hex_text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 4. 主界面逻辑 ---

# 初始化 session state 用于存储用户选择的实物颜色索引
if 'selected_color_index' not in st.session_state:
    st.session_state.selected_color_index = 0

st.title("🎨 色彩匹配助手")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="medium")

# ================= 左侧：标准色 (Target) =================
with col_left:
    st.subheader("1. 设定标准色")
    tab1, tab2 = st.tabs(["🔢 输入色值", "🖼️ 从图片提取"])
    target_rgb = None
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            hex_input = st.text_input("HEX 代码", "#3366FF", label_visibility="collapsed", placeholder="#3366FF")
        with c2: st.write("")
        rgb_result = hex_to_rgb(hex_input)
        if rgb_result:
            target_rgb = rgb_result
            display_color_compact(target_rgb, "当前标准色", height=50)
        else:
            st.error("代码无效")
    with tab2:
        uploaded_target = st.file_uploader("上传标准图片", type=["jpg", "png", "jpeg", "webp"], key="t_up", label_visibility="collapsed")
        if uploaded_target:
            img_target = load_image(uploaded_target)
            # 标准色提取不需要过滤黑白，取第一主色即可
            t_colors, _ = extract_palette_filtered(img_target, k_extract=5, k_final=1)
            target_rgb = t_colors[0]
            tc1, tc2 = st.columns([2, 1])
            with
