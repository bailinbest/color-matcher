import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

# --- 配置 ---
st.set_page_config(
    page_title="色彩相似度匹配器",
    page_icon="🎨",
    layout="wide"
)

# --- 工具函数 ---

def hex_to_rgb(hex_code):
    """将 HEX 颜色代码转换为 RGB 元组 (0-255)。"""
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None

def rgb_to_hex(rgb):
    """将 RGB 元组转换为 HEX 代码。"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_image(image_file):
    """加载并预处理图片。"""
    img = Image.open(image_file)
    img = img.convert('RGB')
    return img

def extract_palette(image, k=5, image_resize=(150, 150)):
    """
    使用 K-Means 聚类从图像中提取主要颜色。
    为了性能，默认将图像缩小处理。
    """
    # 调整大小以加快处理速度
    img_small = image.resize(image_resize)
    img_array = np.array(img_small)
    
    # 重塑数组 (Height * Width, 3)
    pixels = img_array.reshape(-1, 3)
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # 获取聚类中心（主色）
    colors = kmeans.cluster_centers_
    
    # 计算每个聚类的像素数量（占比）
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # 按占比从大到小排序
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    return sorted_colors, sorted_counts

def calculate_similarity_ciede2000(rgb1, rgb2):
    """
    计算两个 RGB 颜色在 CIELAB 空间中的 CIEDE2000 色差。
    返回：色差值 (Delta E) 和 相似度百分比。
    """
    # skimage 需要 float [0, 1] 范围的输入，且形状为 (1, 1, 3)
    color1_norm = np.array(rgb1) / 255.0
    color2_norm = np.array(rgb2) / 255.0
    
    color1_lab = color.rgb2lab(color1_norm.reshape(1, 1, 3))
    color2_lab = color.rgb2lab(color2_norm.reshape(1, 1, 3))
    
    # 计算 CIEDE2000 色差
    delta_e = color.deltaE_ciede2000(color1_lab, color2_lab)[0][0]
    
    # 转换 Delta E 为相似度百分比 (0-100%)
    # 平滑算法：更符合用户直觉
    similarity = 100 / (1 + 0.1 * delta_e)**2
    
    return delta_e, similarity

def display_color_block(rgb, label="Color", height=50):
    """在 Streamlit 中显示一个颜色块。"""
    hex_color = rgb_to_hex(rgb)
    # 计算亮度以决定文字颜色是黑还是白
    brightness = sum(rgb)
    text_color = '#000' if brightness > 382 else '#fff'
    
    st.markdown(
        f"""
        <div style="
            background-color: {hex_color};
            width: 100%;
            height: {height}px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {text_color};
            font-weight: bold;
            font-family: monospace;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 5px;
        ">
            {hex_color.upper()}
        </div>
        <div style="text-align: center; font-size: 0.85em; color: #666; margin-bottom: 10px;">{label}</div>
        """,
        unsafe_allow_html=True
    )

# --- 主界面逻辑 ---

st.title("🎨 AI 色彩相似度匹配器")
st.markdown("""
此工具使用 **CIELAB 色彩空间** 和 **CIEDE2000 色差公式** 来计算颜色的相似度。
它能比传统的 RGB 对比更准确地反映人眼的感知差异。
""")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

# --- 左栏：输入 A (标准色) ---
with col1:
    st.markdown("### 1. 设定标准色 (Target)")
    input_method = st.radio("选择输入方式:", ["输入 HEX 色值", "从图片提取"], horizontal=True)
    
    target_rgb = None
    
    if input_method == "输入 HEX 色值":
        hex_input = st.text_input("输入 HEX 代码 (例如 #FF5733):", "#3366FF")
        rgb_result = hex_to_rgb(hex_input)
        if rgb_result:
            target_rgb = rgb_result
            st.success("✅ 颜色有效")
            display_color_block(target_rgb, "标准色")
        else:
            st.error("❌ 无效的 HEX 代码")
            
    else: # 从图片提取
        uploaded_target = st.file_uploader("上传标准色卡/图片", type=["jpg", "png", "jpeg", "webp"], key="target_upload")
        if uploaded_target:
            img_target = load_image(uploaded_target)
            st.image(img_target, caption="标准图", use_container_width=True)
            
            with st.spinner("正在提取标准色..."):
                target_colors, _ = extract_palette(img_target, k=3)
                target_rgb = target_colors[0]
                st.write("提取到的主要标准色：")
                display_color_block(target_rgb, "提取的主色")

# --- 右栏：输入 B (实物图) ---
with col2:
    st.markdown("### 2. 上传实物图 (Sample)")
    uploaded_sample = st.file_uploader("上传需要对比的实物图片", type=["jpg", "png", "jpeg", "webp"], key="sample_upload")
    
    selected_sample_rgb = None
    
    if uploaded_sample:
        img_sample = load_image(uploaded_sample)
        st.image(img_sample, caption="实物图", use_container_width=True)
        
        with st.spinner("正在分析实物色彩构成..."):
            # 提取 Top 5 颜色
            palette, counts = extract_palette(img_sample, k=5)
            total_pixels = sum(counts)
            
            st.subheader("选择要对比的实物主色:")
            st.info("👇 点击下方单选按钮，选择最能代表实物的颜色（排除背景色）")
            
            # 创建一个用于选择颜色的列布局
            cols = st.columns(5)
            # 使用 session_state 来保存选择，或者简单的 radio
            choice = st.radio(
                "选择颜色索引:", 
                range(len(palette)), 
                label_visibility="collapsed", 
                horizontal=True, 
                format_func=lambda x: ""
            )
            
            for i, (color_val, count) in enumerate(zip(palette, counts)):
                percentage = (count / total_pixels) * 100
                with cols[i]:
                    display_color_block(color_val, f"{percentage:.0f}%", height=40)
                    if i == choice:
                        st.caption("⬆️ 已选")
            
            selected_sample_rgb = palette[choice]

# --- 核心对比逻辑 ---
st.divider()

if target_rgb is not None and selected_sample_rgb is not None:
    st.header("3. 分析结果 (Analysis Result)")
    
    container = st.container()
    with container:
        # 再次展示对比双方
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            display_color_block(target_rgb, "标准色 (A)", height=100)
        with c2:
            st.markdown("<h1 style='text-align: center; line-height: 100px; color: #888;'>VS</h1>", unsafe_allow_html=True)
        with c3:
            display_color_block(selected_sample_rgb, "实物取样色 (B)", height=100)
            
        # 计算差异
        delta_e, similarity = calculate_similarity_ciede2000(target_rgb, selected_sample_rgb)
        
        st.write("") # Spacer
        
        # 结果展示卡片
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
             st.markdown(f"### 相似度: :rainbow[{similarity:.1f}%]")
             st.progress(similarity / 100)
        
        with result_col2:
            st.metric(
                label="Delta E (色差值)", 
                value=f"{delta_e:.2f}", 
                delta="越小越好" if delta_e > 0 else None, 
                delta_color="inverse",
                help="CIEDE2000 色差标准：< 1.0 为人眼不可见，> 5.0 为明显色差"
            )
        
        # 专业解读
        st.subheader("📝 专业解读")
        if delta_e < 1.0:
            st.success("🌟 **完美匹配 (Perfect)**：人眼几乎无法察觉差异。")
        elif delta_e < 2.5:
            st.info("✅ **极佳匹配 (Excellent)**：差异极小，仅在近距离严苛对比时可见。")
        elif delta_e < 5.0:
            st.warning("⚠️ **良好匹配 (Good)**：存在可见色差，但在一般商业用途可接受范围内。")
        elif delta_e < 10.0:
            st.warning("🤔 **差异明显 (Fair)**：颜色明显不同，普通人一眼就能看出区别。")
        else:
            st.error("❌ **不匹配 (Poor)**：完全不同的颜色。")
        
else:
    st.info("👋 请在上方完成 Standard 和 Sample 的设置以开始分析。")