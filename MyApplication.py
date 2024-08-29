import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import io 

# Charger et afficher les données
data = pd.read_csv("Financial_inclusion_dataset.csv")
st.write(data)

# Gérer les valeurs manquantes et les doublons
data = data.dropna()  
data = data.drop_duplicates()

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data = remove_outliers(data, column)

st.write("Nombre de lignes après suppression des valeurs aberrantes : ", data.shape[0])

# Encoder les caractéristiques catégorielles
le = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = le.fit_transform(data[column])
buffer = io.StringIO()
data.info(buf=buffer)
info_string = buffer.getvalue()
st.text(info_string)

# Supprimer les colonnes non nécessaires
columns_to_drop = ['bank_account', 'country', 'year', 'uniqueid', 'cellphone_access', 'household_size', 'gender_of_respondent', 'relationship_with_head', 'marital_status']
X = data.drop(columns_to_drop + ['bank_account'], axis=1)
y = data['bank_account']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Former le classificateur
st.write("### Former le modèle de classification")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Afficher les résultats de l'entraînement
st.write("### Résultats de l'entraînement")
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
st.write(f"Précision sur les données d'entraînement: {train_accuracy}")
st.write(f"Précision sur les données de test: {test_accuracy}")

# Créer l'application Streamlit pour les prédictions
st.title('Prédiction de Possession de Compte Bancaire')
st.write('Veuillez entrer les valeurs des caractéristiques :')
input_features = []
for feature in X.columns:
    value = st.text_input(f'{feature}', '')
    if value.strip() == '':
        input_features.append(np.nan)  # Remplace None par np.nan
    else:
        try:
            input_features.append(float(value))  # Conversion en float
        except ValueError:
            st.error(f"La valeur pour {feature} n'est pas un nombre valide.")  # Affiche un message d'erreur
            input_features.append(np.nan)  # Ajoute np.nan en cas d'erreur

# Assurez-vous que la taille des caractéristiques est correcte
if len(input_features) != len(X.columns):
    st.error("La taille des caractéristiques d'entrée ne correspond pas à celle du modèle.")
else:
    # Remplacer np.nan par une valeur par défaut
    input_features = [x if not np.isnan(x) else X.mean().values[i] for i, x in enumerate(input_features)]
    
    # Prédire avec le modèle formé
    if st.button('Valider'):
        try:
            input_features_array = np.array(input_features).reshape(1, -1)
            prediction_proba = clf.predict_proba(input_features_array)
            prediction = clf.predict(input_features_array)
            
            st.write(f'Probabilités : {prediction_proba}')
            
            if prediction[0] == 1:
                st.write('La personne peut avoir un compte bancaire.')
            else:
                st.write('La personne ne peut pas avoir de compte bancaire.')
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")