def rf_train() :
    """Entraine le modèle Random Forest et le sauvegarde"""
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, Y_train, Y_test = TP.get_X_Y()

    # Définition du modèle Random Forest avec les meilleurs paramètres trouvés
    best_param = {'learning_rate': 0.1,'max_depth': 4,'n_estimators': 500}
    rf_model = RandomForestClassifier(n_estimators=best_param['n_estimators'], verbose=10, n_jobs=-1)
    rf_model.fit(X_train, Y_train)
    
    # Sauvegarder le modèle entraîné

    TP.save_model([rf_model, X_train, X_test, Y_train, Y_test], 'rf')

    print("Modèle entraîné et sauvegardé avec succès.")

if __name__=="__main__":
    rf_train()