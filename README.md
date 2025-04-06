
# 🧠 AI-Based Recipe Recommendation System 🍽️

This project is a beginner-friendly **AI-powered recipe recommender** that suggests recipes based on the **ingredients you have** and your **dietary preferences** (like vegetarian, vegan, gluten-free, etc.).

---

## 🔍 Features

- 🔍 Ingredient-based recipe matching
- 🥦 Filters recipes by diet (vegetarian, vegan, gluten-free, etc.)
- 🤖 Uses TF-IDF and Cosine Similarity (NLP-based)
- ✅ Beginner-friendly and easy to run

---

## 📁 Files

- `recipe_recommender.py` – The main Python program
- `recipes.csv` – Dataset of recipes (edit this name with your actual CSV)
- `README.md` – You're reading it!

---

 How to Run

### 1. Clone this repo:
```bash
git clone https://github.com/NishikacelineCS/ai-recipe-recommender.git
cd ai-recipe-recommender
```

### 2. Install dependencies:
```bash
pip install pandas scikit-learn
```

### 3. Run the program:
```bash
python recipe_recommender.py
```

---

## 🧪 Sample Output
```bash
📝 Enter ingredients you have (comma-separated): egg,milk 
🍽️ Enter your dietary preference (e.g., vegetarian, vegan, gluten-free, dairy-free, non-vegetarian): vegetarian

🍽️ Recommended Recipes:

➡️ Pancakes
   Ingredients: flour, egg, milk, sugar, butter
   Diet: vegetarian

➡️ Smoothie
   Ingredients: banana, milk, honey, yogurt
   Diet: vegetarian,gluten-free

➡️ Fish Curry
   Ingredients: rice, egg, soy sauce, vegetables
   Diet: non-vegetarian

➡️ Grilled Chicken
   Ingredients: chicken, olive oil, lemon, garlic, pepper
   Diet: non-vegetarian,dairy-free

➡️ Egg Fried Rice
   Ingredients: fish, tomato, onion, garlic, spices
   Diet: non-vegetarian,dairy-free
```

---

## 📚 Learnings

This was my first AI mini project. I learned:
- How to use pandas for data handling
- How to apply TF-IDF & cosine similarity
- How to filter data based on user input
- How to push a project to GitHub 💻

---

## Created by

**Nishika**  
Second-year Computer Science Engineering Student  
Passionate about AI, design, and helping people ❤️
