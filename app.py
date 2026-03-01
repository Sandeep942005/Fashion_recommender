import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ═══════════════════════════════════════════════════════════════
#                    PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StyleMatch — Fashion Recommender",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#                       CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Hero Header ───────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #b0b0c8;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* ── Section Titles ────────────────────────── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0e0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #302b63;
}

/* ── Upload Card ───────────────────────────── */
.upload-zone {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 2px dashed #4a47a3;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s ease;
}
.upload-zone:hover {
    border-color: #ffd200;
}

/* ── Product Cards ─────────────────────────── */
.product-card {
    background: linear-gradient(145deg, #1e1e30, #252540);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}
.product-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.45);
}
.product-card img {
    width: 100%;
    height: 220px;
    object-fit: cover;
}
.product-info {
    padding: 0.8rem 1rem;
}
.product-category {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.product-score {
    color: #9e9eb8;
    font-size: 0.8rem;
    margin-top: 6px;
}

/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1a2e);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffd200;
}

/* ── Category Badge Colors ─────────────────── */
.cat-handbags { background: linear-gradient(135deg, #f093fb, #f5576c); }
.cat-jeans    { background: linear-gradient(135deg, #4facfe, #00f2fe); }
.cat-shirts   { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #111; }
.cat-tshirts  { background: linear-gradient(135deg, #fa709a, #fee140); color: #111; }
.cat-watches  { background: linear-gradient(135deg, #a18cd1, #fbc2eb); color: #111; }

/* ── Metric Cards ──────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1e1e30, #252540);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #9e9eb8;
    font-size: 0.85rem;
    margin-top: 4px;
}

/* ── Uploaded Image Styling ────────────────── */
.uploaded-img-container {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}

/* ── Footer ────────────────────────────────── */
.footer {
    text-align: center;
    color: #6b6b8a;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #2a2a40;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#                     LOAD MODELS & DATA
# ═══════════════════════════════════════════════════════════════
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CATEGORY_COLORS = {
    "Handbags": "cat-handbags",
    "Jeans": "cat-jeans",
    "Shirts": "cat-shirts",
    "Tshirts": "cat-tshirts",
    "Watches": "cat-watches",
}


@st.cache_resource
def load_models():
    """Load the KNN model and ResNet50 feature extractor."""
    knn = pickle.load(open(os.path.join(BASE_PATH, "knn_model.pkl"), "rb"))
    # Use ResNet50 directly for feature extraction (2048-dim with avg pooling)
    # instead of loading category_model.h5 to avoid Keras 3 compatibility issues
    feature_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    return knn, feature_model


@st.cache_data
def load_data():
    """Load filenames and categories."""
    filenames = pickle.load(open(os.path.join(BASE_PATH, "filenames.pkl"), "rb"))
    categories = pickle.load(open(os.path.join(BASE_PATH, "categories.pkl"), "rb"))
    features = np.load(os.path.join(BASE_PATH, "features.npy"))
    # Normalize Windows backslashes to forward slashes
    filenames = [f.replace("\\", "/") for f in filenames]
    return filenames, categories, features


knn_model, feature_model = load_models()
filenames, categories, features = load_data()


# ═══════════════════════════════════════════════════════════════
#                     HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and preprocess image for ResNet50."""
    image = image.resize((224, 224))
    img_array = np.array(image)
    # Remove alpha channel if present
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    # Handle grayscale
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float64))
    return img_array


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract feature vector using the ResNet50 model."""
    feat = feature_model.predict(image, verbose=0)
    return feat.flatten().reshape(1, -1)


def get_recommendations(query_features: np.ndarray, n: int = 5):
    """Return indices and distances for the top-n similar items."""
    distances, indices = knn_model.kneighbors(query_features, n_neighbors=n + 1)
    # Skip the first result if distance is 0 (exact match with itself)
    result_indices = []
    result_distances = []
    for i in range(len(indices[0])):
        if len(result_indices) >= n:
            break
        result_indices.append(indices[0][i])
        result_distances.append(distances[0][i])
    return result_indices[:n], result_distances[:n]


# ═══════════════════════════════════════════════════════════════
#                          SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✨ StyleMatch")
    st.markdown("---")
    st.markdown("### 📁 Dataset Overview")
    st.markdown(f"**Total Products:** {len(filenames):,}")

    from collections import Counter
    cat_counts = Counter(categories)
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        st.markdown(f"- **{cat}**: {count:,} items")

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    - **Feature Extractor**: ResNet50
    - **Similarity Metric**: Cosine
    - **Feature Dimensions**: 2,048
    """)
    st.markdown("---")
    st.markdown(
        "<div style='color:#6b6b8a; font-size:0.8rem;'>"
        "Built with TensorFlow + Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#                          HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>StyleMatch</h1>
    <p>Upload any fashion product image and discover 5 visually similar recommendations</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#                     METRICS ROW
# ═══════════════════════════════════════════════════════════════
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-value">15,190</div>'
        '<div class="metric-label">Products in Database</div>'
        '</div>', unsafe_allow_html=True
    )
with m2:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-value">5</div>'
        '<div class="metric-label">Categories</div>'
        '</div>', unsafe_allow_html=True
    )
with m3:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-value">2,048</div>'
        '<div class="metric-label">Feature Dimensions</div>'
        '</div>', unsafe_allow_html=True
    )
with m4:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-value">Cosine</div>'
        '<div class="metric-label">Similarity Metric</div>'
        '</div>', unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#                     IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📤 Upload a Fashion Product Image</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ═══════════════════════════════════════════════════════════════
#                  RESULTS: UPLOADED + RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Uploaded Image + Category ────────────
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(
            '<div class="section-title">🖼️ Your Product</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with right_col:
        with st.spinner("🔍 Analyzing image & finding similar products..."):
            processed = preprocess_image(image)
            query_features = extract_features(processed)
            rec_indices, rec_distances = get_recommendations(query_features, n=5)

        # Predict category of uploaded image
        # Use KNN to infer category from nearest neighbor
        predicted_category = categories[rec_indices[0]]
        cat_class = CATEGORY_COLORS.get(predicted_category, "")

        st.markdown(
            '<div class="section-title">📋 Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="product-category {cat_class}">'
            f"{predicted_category}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        similarity_score = 1 - rec_distances[0]  # cosine distance → similarity
        st.metric("Closest Match Similarity", f"{similarity_score:.1%}")
        st.markdown(
            f"Found **5 similar products** from a database of "
            f"**{len(filenames):,}** items."
        )

    # ── Recommendations Grid ─────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🎯 Top 5 Recommendations</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5, gap="medium")

    for i, (idx, dist) in enumerate(zip(rec_indices, rec_distances)):
        img_path = os.path.join(BASE_PATH, filenames[idx])
        cat = categories[idx]
        cat_class = CATEGORY_COLORS.get(cat, "")
        similarity = 1 - dist

        with cols[i]:
            if os.path.exists(img_path):
                rec_img = Image.open(img_path)
                st.image(rec_img, use_container_width=True)
            else:
                st.warning("Image not found")

            st.markdown(
                f'<div class="product-info">'
                f'<span class="product-category {cat_class}">{cat}</span>'
                f'<div class="product-score">Similarity: {similarity:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    # ── Placeholder when no image is uploaded ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👆 **Upload a fashion product image** (JPG, JPEG, or PNG) to get started. "
        "Try uploading an image of a handbag, pair of jeans, shirt, t-shirt, or watch!"
    )

    # ── Sample images preview ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">👗 Explore Categories</div>',
        unsafe_allow_html=True,
    )
    cat_cols = st.columns(5, gap="medium")
    for i, cat_name in enumerate(["Handbags", "Jeans", "Shirts", "Tshirts", "Watches"]):
        cat_dir = os.path.join(BASE_PATH, "dataset", cat_name)
        with cat_cols[i]:
            if os.path.isdir(cat_dir):
                sample_files = sorted(os.listdir(cat_dir))[:1]
                if sample_files:
                    sample_path = os.path.join(cat_dir, sample_files[0])
                    sample_img = Image.open(sample_path)
                    st.image(sample_img, use_container_width=True)
            cat_class = CATEGORY_COLORS.get(cat_name, "")
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="product-category {cat_class}">{cat_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════
#                          FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">'
    "✨ StyleMatch — Fashion Recommendation Engine &nbsp;|&nbsp; "
    "Powered by ResNet50 + KNN (Cosine Similarity) &nbsp;|&nbsp; "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)