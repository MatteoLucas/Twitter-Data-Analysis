from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm  # Pour la barre de progression
import os
import sys

# Ajouter le chemin du dossier parent
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import TextProcessing as TP

# Récupération des données d'entraînement et de test
X_train, X_test, Y_train, Y_test = TP.get_X_Y()

# Si les données sont sparse, les convertir en dense
if hasattr(X_train, 'toarray'):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

# Définir les modèles de base (Gradient Boosting et SVM)
gb_model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, n_estimators=500)
svm_model = SVC(C=1.1, gamma='scale', kernel='rbf')

# Définir plusieurs méta-modèles à tester
meta_models = {
    'LogisticRegression': LogisticRegression()
}

# Tester les différents méta-modèles avec GridSearchCV pour ajuster les hyperparamètres
best_accuracy = 0
best_meta_model_name = None
best_stacking_model = None

# Ajouter une barre de progression avec tqdm pour suivre les tests des méta-modèles
for meta_model_name, meta_model in tqdm(meta_models.items(), desc="Test des méta-modèles"):
    print(f"\nTesting meta-model: {meta_model_name}")

    # Créer le modèle de stacking avec le méta-modèle actuel
    stacking_model = StackingClassifier(
        estimators=[('gb', gb_model), ('svm', svm_model)],
        final_estimator=meta_model,
        cv=5  # Utilisation de la validation croisée
    )

    # Paramètres à tuner pour chaque méta-modèle
    param_grid = {}
    if meta_model_name == 'LogisticRegression':
        param_grid = {'final_estimator__C': [0.1, 1, 10], 'final_estimator__solver': ['lbfgs', 'liblinear']}
    elif meta_model_name == 'RandomForestClassifier':
        param_grid = {'final_estimator__n_estimators': [100, 200, 500], 'final_estimator__max_depth': [3, 5, 10]}
    elif meta_model_name == 'DecisionTreeClassifier':
        param_grid = {'final_estimator__max_depth': [3, 5, 10], 'final_estimator__min_samples_split': [2, 5, 10]}

    # Effectuer une recherche GridSearchCV pour trouver les meilleurs hyperparamètres avec barre de progression
    with tqdm(total=len(param_grid), desc=f"Recherche d'hyperparamètres pour {meta_model_name}") as pbar:
        grid_search = GridSearchCV(stacking_model, param_grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        pbar.update(1)

    # Faire des prédictions avec le meilleur modèle trouvé
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)

    # Calculer l'accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy with {meta_model_name}: {accuracy * 100:.2f}%")

    # Garder le meilleur modèle
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_meta_model_name = meta_model_name
        best_stacking_model = best_model

# Afficher les résultats finaux
print(f"\nMeilleur modèle : {best_meta_model_name}")
print(f"Accuracy du meilleur modèle : {best_accuracy * 100:.2f}%")

# Calculer et afficher la matrice de confusion du meilleur modèle
cm = confusion_matrix(Y_test, best_stacking_model.predict(X_test))
print("Matrice de confusion du meilleur modèle :")
print(cm)
