# Projet : Classification de commentaires (positifs ou négatifs)
# Auteur : Yamina Tandjaoui

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Préparer les données (exemples simples)
data = {
    "texte": [
        "J'adore cette application, elle est incroyable !",
        "Très mauvaise expérience, je ne recommande pas.",
        "C'est super utile et rapide.",
        "Je déteste cette interface.",
        "Service client très réactif, bravo !",
        "Application inutile, perte de temps.",
        "Excellent outil, je l'utilise tous les jours.",
        "Beaucoup de bugs, trop lent."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positif, 0 = négatif
}

df = pd.DataFrame(data)

# 2. Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(df["texte"], df["label"], test_size=0.3, random_state=42)

# 3. Vectorisation (transformer texte -> vecteurs)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Entraînement du modèle
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Prédiction
y_pred = model.predict(X_test_vec)

# 6. Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nRapport de classification:\n", classification_report(y_test, y_pred))
