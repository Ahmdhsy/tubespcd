import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# ----------------------
# Fungsi ambil resep dari API
# ----------------------
def get_resep_makanan(nama_makanan):
    try:
        search_url = f"https://masak-apa-tomorisakura.vercel.app/api/search/?q={nama_makanan}"
        res = requests.get(search_url)
        if res.status_code == 200:
            data = res.json()
            if data['results']:
                key = data['results'][0]['key']
                detail_url = f"https://masak-apa-tomorisakura.vercel.app/api/recipe/{key}"
                detail_res = requests.get(detail_url)
                if detail_res.status_code == 200:
                    return detail_res.json().get('results', None)
    except Exception as e:
        print("❌ Error saat mengambil resep:", e)
    return None

# ----------------------
# Load model & class
# ----------------------
checkpoint = torch.load("model_makanan.pth", map_location='cpu')
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----------------------
# Transformasi Gambar
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Deteksi Makanan", layout="centered")
st.title("🍛 Deteksi Makanan Khas Indonesia")
st.markdown("Upload gambar makanan khas Indonesia. Sistem akan menebak nama makanannya lalu menampilkan resep secara otomatis dari API!")

uploaded_file = st.file_uploader("📸 Upload gambar makanan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_label = class_names[pred.item()]

        st.subheader(f"Hasil Deteksi: 🍽️ **{pred_label}**")

        # Ambil resep dari API
        st.markdown("### 📝 Resep dari API:")
        resep = get_resep_makanan(pred_label)

        if resep:
            st.image(resep['thumb'], caption=resep['title'])
            st.subheader("📋 Bahan-bahan:")
            for item in resep['ingredient']:
                st.markdown(f"- {item}")

            st.subheader("👨‍🍳 Langkah-langkah:")
            for idx, step in enumerate(resep['step'], start=1):
                st.markdown(f"{idx}. {step}")
        else:
            st.warning("Resep tidak ditemukan di API.")
    except Exception as e:
        st.error(f"Terjadi error saat memproses gambar: {e}")
