# Twitter Data set for Arabic Sentiment Analysis

## Description
Ce projet est une proposition de solution au Twitter Data set for Arabic Sentiment Analysis. Le descriptif du data set est disponible ici : https://archive.ics.uci.edu/dataset/293/twitter+data+set+for+arabic+sentiment+analysis 
L'objectif est de mettre en œuvre différentes méthodes de Machine Learning et Deep Learning pour effectuer une prédiction.

## Table des Matières
- [Installation](#installation)
- [Usage](#usage)
- [Résultats](#résultats)
- [Auteurs](#auteurs)

## Installation
1. Clonez le dépôt : `git clone XXX`  
2. Allez dans le répertoire du projet : `cd XXX`  
3. Installez les dépendances : `npm install` ou `pip install -r requirements.txt`  

## Usage
Le code permet de faire différentes choses :
### Entrainement d'un modèle
Pour entrainer un modèle, il suffit de lancer dans le répertoire du projet :
```bash
python Train_Functions/model.py
```
En remplaçant : 
- `model` par le nom du modèle à entrainer : `svm`, `rf`, `knn`, `rn`, `gb`.
  

Par exemple :
```bash
python Train_Functions/knn.py
python Train_Functions/svm.py
```

### Prédiction
Pour effectuer une prédiction à partir d'un modèle entraîné, il suffit de lancer dans le répertoire du projet :
```bash
python ./predict.py model
```
En remplaçant : 
- `model` par le nom du modèle à partir duquel faire la prédiction : `svm`, `rf`, `knn`, `rn`, `gb`.

Par exemple :
```bash
python ./predict.py rf
```

### Vote majoritaire
Il est aussi possible d'effectuer un vote majoritaire entre plusieurs modèles entraînés. Pour ce faire, il suffit de lancer dans le répertoire du projet : 
```bash
python ./vote.py model1 model2 model3
```
En remplaçant : 
- `model1`, `model2`, `model3` par le nom des modèles à partir desquels faire le vote : `svm`, `rf`, `knn`, `rn`, `gb`. Il faut mettre au minimum 2 modèles. 

Par exemple :
```bash
python ./vote.py svm rn rf knn True
```

## Résultats
Le tableau ci-dessous regroupe les scores obtenus par nos différents modèles lors de la soumission des résultats sur le site du challenge.

| Modèle | Abréviation | Score lors de la soumission |
|-----------|-----------|-----------|
| Support Vector Machine  | svm  | xxx |
| Random Forest | rf | xxx |
| Gradient Boosting | gb | xxx |
| Réseau de neurones simple  | rn  | xxx  |
| K plus proches voisins | knn | xxx |  

Nous avons aussi commencé à implémenter une méthode basée sur le modèle BERT et une basée sur les CNN, mais nous avons été limités par nos capacités d'entraînement. Le travail en cours est disponible sur les branches éponymes.

## Auteurs
- Mattéo Lucas
- Kamil Hammani
- Nicolas Massabie
- Fabien Joao
