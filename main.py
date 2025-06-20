import streamlit as st
import tempfile
from app.yolo_detector import detect_food_objects
from app.themealdb_api import search_meal_by_name, get_ingredients
from app.label_mapping import label_to_search_query

st.title("🍽️ YOLO Food Detector + Resep Makanan")

uploaded_file = st.file_uploader("Unggah gambar makanan", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Gambar yang diunggah", use_container_width=True)

    # Simpan ke file temporer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name

    # Deteksi makanan
    labels = detect_food_objects(temp_image_path)

    st.markdown("### 🎯 Hasil Deteksi:")
    for label in labels:
        st.markdown(f"- **{label}**")

        # Ubah label jadi query pencarian
        query = label_to_search_query.get(label.lower(), label)
        meals = search_meal_by_name(query)
        st.write(f"Resep berkaitan dengan: '{label}'")

        if meals:
            for meal in meals:
                st.markdown("---")
                st.subheader(meal['strMeal'])
                st.image(meal['strMealThumb'], width=300)
                st.markdown(f"**Kategori:** {meal['strCategory']}  |  **Asal:** {meal['strArea']}")

                st.markdown("**📋 Bahan-bahan:**")
                for item in get_ingredients(meal):
                    st.write(item)

                st.markdown("**📖 Instruksi:**")
                st.write(meal['strInstructions'])

                if meal['strYoutube']:
                    st.markdown(f"[▶️ Tonton Video]({meal['strYoutube']})")
        else:
            st.warning(f"❗ Resep untuk '{label}' tidak ditemukan.")
