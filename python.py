import requests
from PIL import Image
from io import BytesIO


def search_meal(query):
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        meals = data.get('meals')
        if meals:
            for meal in meals:
                print(f"Nama Makanan : {meal['strMeal']}")
                print(f"Kategori     : {meal['strCategory']}")
                print(f"Area         : {meal['strArea']}")
                print(f"Instruksi    : \n{meal['strInstructions']}")
                print(f"Link Video   : {meal['strYoutube']}")

                print("Ingredients  :")
                for i in range(1, 21):
                    ingredient = meal.get(f"strIngredient{i}")
                    measure = meal.get(f"strMeasure{i}")
                    if ingredient and ingredient.strip() != "":
                        print(f"  - {ingredient} : {measure.strip() if measure else ''}")
                
                print("="*60)
                print(f"Gambar       : {meal['strMealThumb']}")

                # Tampilkan gambar secara langsung (jika pakai GUI atau desktop)
                try:
                    img_response = requests.get(meal['strMealThumb'])
                    img = Image.open(BytesIO(img_response.content))
                    img.show()  # Ini akan buka image viewer default di OS
                except:
                    print("Gagal menampilkan gambar.")

        else:
            print("Tidak ditemukan hasil untuk:", query)
    else:
        print("Terjadi kesalahan:", response.status_code)

# Contoh penggunaan
search_meal("nasi")
