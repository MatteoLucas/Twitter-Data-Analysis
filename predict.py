def predict(model_name):
    '''
    Fonction pour faire une prédiction à partir des modèles entrainés
    Args:
        model_name (str): Name of the model (rf, svc, dt, )
    '''
    from sklearn.metrics import accuracy_score
    import TextProcessing as TP
    import numpy as np
    


    model, X_train, X_test, Y_train, Y_test = TP.load_model(model_name) 


    X_train, X_test, Y_train, Y_test = TP.get_X_Y() #Pour etre sur que la prédiction sera sur le bon X_test
    
    print("Predicting ...")
    
    if model_name=="nb":
        X_test = X_test.toarray()

    # Prédiction et évaluation
    Y_pred = model.predict(X_test)

    # Convertir Y_pred en labels si nécessaire
    if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
        Y_pred = np.argmax(Y_pred, axis=1)
    # Convertir Y_test en labels si nécessaire
    if Y_test.ndim > 1 and Y_test.shape[1] > 1:
        Y_test = np.argmax(Y_test, axis=1)

    print("Accuracy : ", accuracy_score(Y_test, Y_pred))
    TP.save_predictions_to_csv(Y_pred, "Y_pred_"+model_name+".csv", X_train)
    return Y_pred, Y_test, X_train

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()
    predict(args.model)