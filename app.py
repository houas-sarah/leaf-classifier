import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import pandas as pd

# ---------------------------
# CONFIG DE LA PAGE
# ---------------------------
st.set_page_config(
    page_title="Leaf Classifier",
    page_icon="🌿",
    layout="centered"
)

# ---------------------------
# STYLES (sobre)
# ---------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .label-muted {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .pred-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# CHARGEMENT DU MODÈLE
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "resnet_leaf_tl.pth"

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt.get("classes", ["dentee", "lisse"])
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("std",  [0.229, 0.224, 0.225])

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, tfms, classes

model, tfms, CLASS_NAMES = load_model()

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<div class='main-title'>Leaf Classifier 🌿</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Application de classification de feuilles (par exemple lisse / dentée) avec un modèle ResNet18 fine-tuné.</div>",
    unsafe_allow_html=True
)

# ---------------------------
# CONTENU PRINCIPAL
# ---------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Image")
    st.markdown(
        "<p class='label-muted'>Ajoute une image de feuille (JPG, JPEG ou PNG).</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Aperçu de l'image", use_container_width=True)

with col_right:
    st.subheader("Prédiction")

    if uploaded_file is None:
        st.markdown(
            "<p class='label-muted'>En attente d'une image... téléverse une feuille à gauche pour lancer la prédiction.</p>",
            unsafe_allow_html=True,
        )
    else:
        if st.button("Lancer la prédiction"):
            with torch.no_grad():
                x = tfms(image).unsqueeze(0).to(DEVICE)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = probs.argmax()
            pred_class = CLASS_NAMES[pred_idx]
            conf = float(probs[pred_idx] * 100)

            st.markdown("<span class='pred-label'>Classe prédite</span>", unsafe_allow_html=True)
            st.markdown(f"### {pred_class}")

            st.markdown("<p class='label-muted'>Confiance estimée :</p>", unsafe_allow_html=True)
            st.progress(min(conf / 100.0, 1.0))
            st.write(f"**{conf:.1f} %**")

            st.markdown("---")
            st.markdown("**Probabilités par classe**")

            df_probs = pd.DataFrame(
                {"Classe": CLASS_NAMES, "Probabilité (%)": [float(p * 100) for p in probs]}
            ).set_index("Classe")

            st.bar_chart(df_probs)
