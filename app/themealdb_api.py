import requests

def search_meal_by_name(query):
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('meals')
    return None

def get_ingredients(meal):
    ingredients = []
    for i in range(1, 21):
        ing = meal.get(f"strIngredient{i}")
        meas = meal.get(f"strMeasure{i}")
        if ing and ing.strip():
            ingredients.append(f"🔸 {ing.strip()} - {meas.strip() if meas else ''}")
    return ingredients
