def majority_vote(models):
    """
    Combine les prédictions de plusieurs modèles par vote majoritaire
    Args :
        model (list): Liste des modèles votant
    """
    from scipy.stats import mode
    import numpy as np
    from sklearn.metrics import accuracy_score
    import TextProcessing as TP
    from predict import predict

    # Empiler les prédictions en une seule matrice
    Y_pred, Y_test, X_train = predict(models[0])
    Y_preds = np.vstack([Y_pred]+[predict(model)[0] for model in models[1:]])
    
    # Calculer le vote majoritaire pour chaque échantillon
    Y_final_pred_full, counts = mode(Y_preds, axis=0)
    
    # En cas d'égalité, retenir la prédiction du premier modèle
    ties = (counts == 1)
    Y_final_pred_full[ties] = Y_preds[0, ties]

    Y_final_pred = Y_final_pred_full.ravel()

    model_name = ""
    for model in models:
        model_name+=model+"+"
    model_name = model[:-1]

    print("Accuracy : ", accuracy_score(Y_test, Y_pred))
    TP.save_predictions_to_csv(Y_final_pred, "Y_pred_"+model_name+".csv", X_train)

    return Y_final_pred

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("list", nargs='+', type=str, help="Une liste de chaînes de caractères à passer à la fonction")
    args = parser.parse_args()
    list_model= args.list[:-1]
    majority_vote(list_model)