import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🥗 Load the dataset correctly with proper separators and quoting
df = pd.read_csv("recipes.csv", quotechar='"', skipinitialspace=True)

# 🧹 Clean column names and ingredient text
df.columns = df.columns.str.strip().str.lower()
df['ingredients'] = df['ingredients'].astype(str).apply(lambda x: x.lower().strip())
df['diet'] = df['diet'].astype(str).apply(lambda x: x.lower().strip())
df['recipe'] = df['recipe'].astype(str).apply(lambda x: x.strip())

# 🧠 Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
ingredient_vectors = vectorizer.fit_transform(df['ingredients'])

# 🔍 Compute cosine similarity between recipes
similarity_matrix = cosine_similarity(ingredient_vectors)

# 🔁 Recommendation function
def recommend_recipe(user_ingredients, user_diet, df, vectorizer, ingredient_vectors):
    user_input_vec = vectorizer.transform([user_ingredients])
    similarities = cosine_similarity(user_input_vec, ingredient_vectors).flatten()
    recommended_indices = similarities.argsort()[::-1]  # Most similar first

    count = 0
    print("\n🍽️ Recommended Recipes:")
    for idx in recommended_indices:
        recipe_diet = df.iloc[idx]['diet']
        
        # Check if diet match (partial match allowed)
        if user_diet in recipe_diet:
            print(f"\n➡️ {df.iloc[idx]['recipe'].title()}")
            print(f"   Ingredients: {df.iloc[idx]['ingredients']}")
            print(f"   Diet: {df.iloc[idx]['diet']}")
            count += 1
        if count == 5:
            break

    if count == 0:
        print("⚠️ Sorry, no recipes found matching your dietary preference and ingredients.")

# 🌟 Get user input
user_input = input("📝 Enter ingredients you have (comma-separated): ").lower().strip()
user_diet = input("🍽️ Enter your dietary preference (e.g., vegetarian, vegan, gluten-free, dairy-free, non-vegetarian): ").lower().strip()

# 🚀 Recommend!
recommend_recipe(user_input, user_diet, df, vectorizer, ingredient_vectors)

