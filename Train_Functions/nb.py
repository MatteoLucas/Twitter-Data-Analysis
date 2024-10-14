def nb_train():
    """Entraine le modèle Naive Bayes, effectue une recherche de paramètres, calcule la matrice de confusion et sauvegarde le modèle."""
    import sys
    import os
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV
    
    import numpy as np

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

    # Paramètres à tester dans le GridSearch pour Naive Bayes (Il n'y a pas beaucoup de paramètres à ajuster dans GaussianNB)
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Smoothing parameter to handle numerical stability
    }

    # Initialisation du modèle Naive Bayes
    nb = GaussianNB()

    # Mise en place du GridSearch pour rechercher les meilleurs hyperparamètres
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=10)

    # Entraînement du modèle avec le GridSearch
    grid_search.fit(X_train, Y_train)

    # Récupération des meilleurs paramètres trouvés par le GridSearch
    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres trouvés : {best_params}")

    # Utilisation du modèle optimisé avec les meilleurs paramètres
    best_nb_model = grid_search.best_estimator_

    # Prédictions sur les données de test
    Y_pred = best_nb_model.predict(X_test)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(Y_test, Y_pred)

    # Affichage de la matrice de confusion
    print("Matrice de confusion :")
    print(cm)

    # Sauvegarder le modèle entraîné
    TP.save_model([best_nb_model, X_train, X_test, Y_train, Y_test], 'nb')

    print("Modèle entraîné, matrice de confusion calculée et sauvegardé avec succès.")


if __name__ == "__main__":
    nb_train()
