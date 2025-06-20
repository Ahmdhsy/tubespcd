# YOLO Meal App 🍕🤖

Aplikasi Streamlit berbasis YOLO untuk mendeteksi makanan dari gambar, lalu menampilkan resep otomatis via API TheMealDB.

## Fitur
- Upload gambar makanan
- Deteksi objek makanan pakai YOLOv5/v8
- Ambil resep dari [TheMealDB](https://www.themealdb.com/api.php)

## Cara Jalankan
1. Clone repo
2. Install dependency:
   ```bash
   pip install -r requirements.txt
3. Lakukan Training
   ```bash
   yolo detect train data=food-detection/data.yaml model=yolov8n.pt epochs=5 imgsz=640 device=0 project=runs name=myfood
4. Run Aplikasi
   ```bash
   streamlit run main.py