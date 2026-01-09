import streamlit as st
from PIL import Image
import io
from your_converter import EditableDocConverter, PPTXExporter
from pdf2image import convert_from_bytes

# 設定頁面
st.set_page_config(
    page_title="PDF/PNG 轉可編輯 PPTX",
    page_icon="📄",
    layout="wide"
)

# 標題
st.title("📄 NotebookLM 輸出轉可編輯工具")
st.markdown("上傳 PDF 或圖片，自動移除文字背景，生成可編輯 PPTX")

# 初始化（使用 @st.cache_resource 避免重複載入模型）
@st.cache_resource
def load_converter():
    return EditableDocConverter(lang='ch')

converter = load_converter()

# 側邊欄參數調整
with st.sidebar:
    st.header("⚙️ 進階設定")
    dilation_size = st.slider("Mask 擴大範圍", 3, 9, 5, 2)
    dilation_iter = st.slider("Mask 擴大次數", 1, 3, 2)
    
    st.markdown("---")
    st.markdown("### 📖 使用說明")
    st.markdown("""
    1. 上傳 NotebookLM 生成的 PDF/PNG
    2. 系統自動識別文字並移除
    3. 下載可編輯的 PPTX 文件
    
    **支援格式**：PDF, PNG, JPG
    """)

# 主要介面
uploaded_file = st.file_uploader(
    "選擇檔案",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    help="上傳 NotebookLM 生成的 infographic 或 slide"
)

if uploaded_file:
    # 顯示檔案資訊
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(f"📁 **檔案名稱**: {uploaded_file.name}")
        st.info(f"📦 **檔案大小**: {uploaded_file.size / 1024:.1f} KB")
    
    # 處理按鈕
    if st.button("🚀 開始處理", type="primary", use_container_width=True):
        
        # 進度條
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: 讀取檔案
            status_text.text("📖 讀取檔案中...")
            progress_bar.progress(10)
            
            if uploaded_file.name.endswith('.pdf'):
                # PDF 轉圖片
                images = convert_from_bytes(
                    uploaded_file.read(), 
                    dpi=300
                )
                total_pages = len(images)
            else:
                # 直接處理圖片
                images = [Image.open(uploaded_file)]
                total_pages = 1
            
            status_text.text(f"📄 偵測到 {total_pages} 頁")
            progress_bar.progress(20)
            
            # Step 2: 創建 PPTX exporter
            exporter = PPTXExporter()
            
            # Step 3: 逐頁處理
            for i, page_image in enumerate(images):
                status_text.text(f"🔍 處理第 {i+1}/{total_pages} 頁 - OCR 識別中...")
                progress_bar.progress(20 + int(70 * (i+0.3) / total_pages))
                
                # 暫存圖片
                temp_path = f'temp_page_{i}.png'
                page_image.save(temp_path)
                
                # OCR + Inpainting
                status_text.text(f"🎨 處理第 {i+1}/{total_pages} 頁 - 移除文字中...")
                progress_bar.progress(20 + int(70 * (i+0.6) / total_pages))
                
                clean_image, text_regions = converter.process_document(
                    temp_path,
                    f'temp_clean_{i}.png',
                    dilation_size=dilation_size,
                    dilation_iter=dilation_iter
                )
                
                # 添加到 PPTX
                status_text.text(f"📝 處理第 {i+1}/{total_pages} 頁 - 生成可編輯圖層...")
                exporter.add_slide_with_overlay(
                    f'temp_clean_{i}.png',
                    text_regions
                )
                
                progress_bar.progress(20 + int(70 * (i+1) / total_pages))
            
            # Step 4: 保存 PPTX
            status_text.text("💾 生成 PPTX 檔案...")
            progress_bar.progress(95)
            
            output_path = 'output_editable.pptx'
            exporter.save(output_path)
            
            # 完成
            progress_bar.progress(100)
            status_text.text("✅ 處理完成！")
            
            # 下載按鈕
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 下載可編輯 PPTX",
                    data=f.read(),
                    file_name=f"editable_{uploaded_file.name.split('.')[0]}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )
            
            # 顯示預覽
            st.success(f"🎉 成功處理 {total_pages} 頁！")
            
            with st.expander("📊 查看處理詳情"):
                st.write(f"- 識別文字數量：{sum(len(r) for r in text_regions)} 個")
                st.write(f"- Mask 擴大參數：{dilation_size}px × {dilation_iter} 次")
                st.write(f"- 輸出檔案：{output_path}")
        
        except Exception as e:
            st.error(f"❌ 處理失敗：{str(e)}")
            st.exception(e)

else:
    # 預設顯示示範
    st.info("👆 請上傳檔案開始處理")
    
    # 顯示示範圖片（可選）
    st.markdown("### 📸 效果示範")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**原始 PDF/PNG**")
        st.image("https://via.placeholder.com/400x300?text=Before", use_container_width=True)
    with col2:
        st.markdown("**可編輯 PPTX**")
        st.image("https://via.placeholder.com/400x300?text=After+(Editable)", use_container_width=True)
