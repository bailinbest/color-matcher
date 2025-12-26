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

# --- 2. CSS 样式优化 (让界面更紧凑) ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
        div.stButton > button {width: 100%;}
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

def extract_palette(image, k=5, image_resize=(150, 150)):
    """提取颜色，返回按占比排序的颜色和像素数"""
    img_small = image.resize(image_resize)
    img_array = np.array(img_small)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    return colors[sorted_indices], counts[sorted_indices]

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
    """显示紧凑的颜色条组件"""
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

st.title("🎨 色彩匹配助手")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="medium")

# ================= 左侧：标准色 (Target) =================
with col_left:
    st.subheader("1. 设定标准色")
    
    # 使用 Tabs 节省垂直空间
    tab1, tab2 = st.tabs(["🔢 输入色值", "🖼️ 从图片提取"])
    
    target_rgb = None
    
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            hex_input = st.text_input("HEX 代码", "#3366FF", label_visibility="collapsed", placeholder="#3366FF")
        with c2:
            st.write("") # 占位
        
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
            t_colors, _ = extract_palette(img_target, k=3)
            target_rgb = t_colors[0] # 默认取主色
            
            # 布局：左边颜色，右边缩略图
            tc1, tc2 = st.columns([2, 1])
            with tc1:
                display_color_compact(target_rgb, "提取结果", height=50)
            with tc2:
                st.image(img_target, width=80, caption="原图预览")

# ================= 右侧：实物图 (Sample) =================
with col_right:
    st.subheader("2. 上传实物图")
    uploaded_sample = st.file_uploader("上传实物照片", type=["jpg", "png", "jpeg", "webp"], key="s_up", label_visibility="collapsed")
    
    selected_sample_rgb = None
    
    if uploaded_sample:
        img_sample = load_image(uploaded_sample)
        
        # 1. 提取颜色
        palette, counts = extract_palette(img_sample, k=5)
        total_pixels = sum(counts)
        
        # 2. 紧凑布局：横向排列
        ic1, ic2 = st.columns([1, 2])
        
        with ic1:
            # 限制图片宽度，避免占位太大
            st.image(img_sample, caption="实物", use_container_width=True) 
        
        with ic2:
            st.caption("🎨 请选择主色 (横向排列):")
            
            # 构造选项标签
            options = list(range(len(palette)))
            def format_func(i):
                return f"{int((counts[i]/total_pixels)*100)}%"

            # 横向单选按钮
            choice = st.radio(
                "选择颜色", 
                options, 
                format_func=format_func, 
                horizontal=True,
                label_visibility="collapsed"
            )
            
            selected_sample_rgb = palette[choice]
            
            # 显示当前选中的大色块
            display_color_compact(selected_sample_rgb, "已选实物色", height=40)
            
        # 视觉辅助条
        cols = st.columns(len(palette))
        for i, color_val in enumerate(palette):
            with cols[i]:
                h_code = rgb_to_hex(color_val)
                # 选中的加粗框
                border = "3px solid #FF4B4B" if i == choice else "1px solid #ddd"
                st.markdown(f"""
                <div style="background-color: {h_code}; height: 15px; border-radius: 2px; border: {border};" title="{h_code}"></div>
                """, unsafe_allow_html=True)


# ================= 底部：结果对比 =================
st.markdown("---")

if target_rgb is not None and selected_sample_rgb is not None:
    delta_e, similarity = calculate_similarity_ciede2000(target_rgb, selected_sample_rgb)
    
    # 结果区域
    with st.container():
        # 标题栏
        st.markdown(f"### 🎯 匹配度: :rainbow[{similarity:.1f}%]")
        
        # 进度条
        st.progress(similarity / 100)
        
        # 详细数据 (四列布局)
        rc1, rc2, rc3, rc4 = st.columns([1.5, 1.5, 1, 2])
        
        with rc1:
            display_color_compact(target_rgb, "标准", height=60, show_hex=True)
        with rc2:
            display_color_compact(selected_sample_rgb, "实物", height=60, show_hex=True)
        with rc3:
            st.metric("色差 (ΔE)", f"{delta_e:.2f}")
        with rc4:
            if delta_e < 2.0:
                st.success("✅ 完美匹配")
            elif delta_e < 5.0:
                st.warning("⚠️ 轻微色差")
            else:
                st.error("❌ 差异明显")
            st.caption("ΔE < 2.0 为极佳")

else:
    if not uploaded_sample:
        st.info("👈 等待上传实物图...")
