import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from joblib import load
import sys


def remove_diacritics(text):
    arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def remove_emoji(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def raw_to_tokens(raw_string):
  stop_words = set(stopwords.words('arabic'))
  raw_string = "".join([word for word in raw_string if word not in string.punctuation])
  raw_string = remove_emoji(raw_string)
  raw_string = remove_diacritics(raw_string)
  tokens = word_tokenize(raw_string)
  raw_string = ' '.join([word for word in tokens if word not in stop_words])
  return raw_string

def ouverture_fichiers():
  # Dossier où se trouvent les fichiers texte
  directory = "Data/"

  # Listes pour stocker les contenus et labels
  texts = []
  labels = []

  # Parcours des fichiers
  for filename in os.listdir(directory):
      if filename.endswith(".txt"):
          label = 1 if "positive" in filename else 0
          filepath = os.path.join(directory, filename)
          
          # Lire le contenu du fichier avec l'encodage cp1256
          with open(filepath, 'r', encoding='cp1256') as file:
              content = file.read()
              texts.append(raw_to_tokens(content))
              labels.append(label)

  # Création de la DataFrame pandas
  df = pd.DataFrame({
      'Texte': texts,
      'Label': labels
  })
  dump(df['Texte'], 'Matrix&Model/X.dataset')
  dump(df['Label'], 'Matrix&Model/Y.dataset')
  return df['Texte'], df['Label']

def get_X_Y() :
  """
  Cette fonction renvoie X_train, X_test, Y_train, Y_test
  """
  from joblib import load

  #On essaye d'ouvrir la matrice tfidf de X_train si elle existe sinon, on la crée
  try : 
    X_test  = load('Matrix&Model/X_test.dataset')
    X_train  = load('Matrix&Model/X_train.dataset')
    Y_test  = load('Matrix&Model/Y_test.dataset')
    Y_train  = load('Matrix&Model/Y_train.dataset')
  except FileNotFoundError :
    try :
      X = load('Matrix&Model/X.dataset')
      Y  = load('Matrix&Model/Y.dataset')
    except: 
      X, Y = ouverture_fichiers()
    #tfidf = TfidfVectorizer(lowercase=False)
    #X_tfidf = tfidf.fit_transform(X)
    cv = CountVectorizer(binary=True)
    cv.fit(X)
    X_tfidf = cv.transform(X)
    # Définir la proportion de l'ensemble de test
    test_portion = 1/10
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=test_portion, shuffle=True)
    dump(X_train, 'Matrix&Model/X_train.dataset')
    dump(X_test, 'Matrix&Model/X_test.dataset')
    dump(Y_train, 'Matrix&Model/Y_train.dataset')
    dump(Y_test, 'Matrix&Model/Y_test.dataset')

 
  return X_train, X_test, Y_train, Y_test


def save_predictions_to_csv(Y_pred, csv_name, X_train):
    """Sauvegarde les prédictions dans un fichier CSV."""
   
    # Créer un DataFrame avec les vraies valeurs et les prédictions
    results_df = pd.DataFrame({
        '': [i for i in range(X_train.shape[0],X_train.shape[0]+len(Y_pred))],
        'prdtypecode': Y_pred
    })
    
    # Sauvegarder le DataFrame dans un fichier CSV
    results_df.to_csv("Predictions/"+csv_name, index=False)
    
    print("Prédictions sauvegardées dans Predictions/"+csv_name)


def split_file(file_base, parts_directory = "Trained_Model/", file_extension = ".model", chunk_size=12 * 1024 * 1024):
  """Splits a binary file into multiple chunks of a given size.
  
  Args:
    parts_directory (str): Directory containing the chunk files.
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
    chunk_size (float): Maximum size of each chunk in bytes (default is 12.5 MB).
  """

  part_number = 0
  while True:
    chunk_file_name = os.path.join(parts_directory, f"{file_base}_part{part_number}{file_extension}")
    if not os.path.exists(chunk_file_name):
      break
    os.remove(parts_directory+file_base+"_part"+str(part_number)+file_extension)
    part_number += 1  

  file_path = parts_directory+file_base+file_extension
  file_base, file_extension = os.path.splitext(file_path)
  with open(file_path, 'rb') as file:
    chunk_count = 0
    while True:
      chunk = file.read(int(chunk_size))
      if not chunk:
        break
      chunk_file_name = f"{file_base}_part{chunk_count}{file_extension}"
      with open(chunk_file_name, 'wb') as chunk_file:
        chunk_file.write(chunk)
      chunk_count += 1


def merge_files(file_base, parts_directory = "Trained_Model/", file_extension = ".model"):
  """Merges multiple chunk files into a single binary file.
  
  Args:
    parts_directory (str): Directory containing the chunk files.
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
  """
  output_file = parts_directory+file_base+file_extension
  with open(output_file, 'wb') as merged_file:
    part_number = 0
    while True:
      chunk_file_name = os.path.join(parts_directory, f"{file_base}_part{part_number}{file_extension}")
      if not os.path.exists(chunk_file_name):
        break
      with open(chunk_file_name, 'rb') as chunk_file:
        merged_file.write(chunk_file.read())
      part_number += 1


def load_model(file_base,  parts_directory = "Trained_Model/", file_extension = ".model") :
  '''
  Args:
    file_base (str): Base name of the chunk files without the part number and extension.
    parts_directory (str): Directory containing the chunk files.
    file_extension (str): Extension of the chunk files.
  '''
  from joblib import load
  if not os.path.exists(parts_directory+file_base+"_part0"+file_extension) :
    raise FileNotFoundError("Entrainez d'abord le modèle avec la fonction "+file_base+"_train()")
  merge_files(file_base, parts_directory, file_extension)
  model, X_train, X_test, Y_train, Y_test = load(parts_directory+file_base+file_extension)
  os.remove(parts_directory+file_base+file_extension)
  return model, X_train, X_test, Y_train, Y_test


def save_model(model_list, file_base, parts_directory = "Trained_Model/", file_extension = ".model") :

  '''
  Args:
    model_list (list): [svc, X_train, X_test, Y_train, Y_test]
    parts_directory (str): Directory containing the chunk files.
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
  '''
  dump(model_list, parts_directory+file_base+file_extension)
  split_file(file_base, parts_directory, file_extension)
  os.remove(parts_directory+file_base+file_extension)