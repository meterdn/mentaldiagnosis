import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, accuracy_score

data = pd.read_csv('Dataset-Mental-Disorders.csv')

data = data.drop(["Patient Number"], axis=1)


data["Sexual Activity"] = data["Sexual Activity"].apply(lambda x: x[0])
data["Concentration"] = data["Concentration"].apply(lambda x: x[0])
data["Optimisim"] = data["Optimisim"].apply(lambda x: x[0])

data["Sexual Activity"] = data["Sexual Activity"].astype(dtype=int)
data["Concentration"] = data["Concentration"].astype(dtype=int)
data["Optimisim"] = data["Optimisim"].astype(dtype=int)

data["Suicidal thoughts"] = data["Suicidal thoughts"].apply(lambda x: x.strip())


encodings_objs = {}
data_encoded = pd.DataFrame()
for column in data.columns:
    if data[column].dtype != int:
        encodings_objs[column] = LabelEncoder().fit(data[column])
        data_encoded[column] = encodings_objs[column].transform(data[column])
    else:
        data_encoded[column] = data[column]


columns = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 'Mood Swing',
       'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation',
       'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down',
       'Admit Mistakes', 'Overthinking', 'Sexual Activity', 'Concentration',
       'Optimisim']
X_train,X_test,Y_train,Y_test = train_test_split(data_encoded[columns], data["Expert Diagnose"], random_state=123, train_size=0.8, shuffle=True, stratify=data["Expert Diagnose"])

rf = RandomForestClassifier(n_estimators=9,random_state=123,criterion="entropy")
nb = CategoricalNB()

vc = VotingClassifier(estimators = [('rf',rf),('nb',nb)],voting="soft")
vc_c = vc.fit(X_train,Y_train)

vc_out = vc_c.predict(X_test)
print(f"Accuracy: {accuracy_score(vc_out,Y_test)*100}%")


patient = {
    'Sadness': 'Sometimes', 
    'Euphoric': 'Sometimes', 
    'Exhausted': 'Sometimes', 
    'Sleep dissorder': 'Sometimes', 
    'Mood Swing': 'YES',
    'Suicidal thoughts': 'NO', 
    'Anorxia': "NO", 
    'Authority Respect': 'YES', 
    'Try-Explanation': 'NO',
    'Aggressive Response': 'YES', 
    'Ignore & Move-On': 'YES', 
    'Nervous Break-down': 'YES',
    'Admit Mistakes': 'NO', 
    'Overthinking': 'YES', 
    'Sexual Activity': 8, 
    'Concentration':5,
    'Optimisim': 7, 
}

# Convert patient data into DataFrame
patient_df = pd.DataFrame(patient, index=[0])

# Encode categorical variables
for column in patient_df.columns:
    if column in encodings_objs:
        patient_df[column] = encodings_objs[column].transform(patient_df[column])

# Make predictions
patient_diagnosis = vc_c.predict(patient_df[columns])

# Print diagnosis
print("Patient Diagnosis:", patient_diagnosis[0])








