import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('heart.csv')
df['sexo'] = df['sexo'].replace([1,0], ['homem', 'mulher'])
df['tipo_dor_peitoral'] = df['tipo_dor_peitoral'].replace([0,1,2,3], ['angina tipica', 'angina atipica', 'dor nao anginosa', 'assintomatico'])
df['glicose'] = df['glicose'].replace([0,1], ['acima 120 mg/dl', 'abaixo 120 mg/dl'])
df['dor_exercitar'] = df['dor_exercitar'].replace([0,1], ['nao', 'sim'])
df['eletrocardiograma'] = df['eletrocardiograma'].replace([0,1,2], ['normal', 'anormal', 'hipertrofia ventricular'])
df['variacao_pico_segmento_exercicio'] = df['variacao_pico_segmento_exercicio'].replace([0,1,2], ['sem inclinacao', 'plano', 'descida'])
df['talassemia'] = df['talassemia'].replace([0,1,2,3], ['nulo', 'defeito corrigido', 'normal', 'defeito reversivel'])
df['output'] = df['output'].replace([0,1], ['menor chance de doenca cardiaca', 'maior chance de doenca cardiaca'])

scaler = StandardScaler()
df['idade'] = scaler.fit_transform(df[['idade']])
df['veias_principais'] = scaler.fit_transform(df[['veias_principais']])
df['pressao_sanguinea_repouso'] = scaler.fit_transform(df[['pressao_sanguinea_repouso']])   
df['colesterol'] = scaler.fit_transform(df[['colesterol']])   
df['maior_batimento_cardiaco'] = scaler.fit_transform(df[['maior_batimento_cardiaco']])
df['depressao_exercicio_repouso'] = scaler.fit_transform(df[['depressao_exercicio_repouso']]) 

# verify if there is any null value
print(df.isnull().sum())

# remove null values
df = df.dropna()

# Dividindo os dados em treino e teste
X = df.drop('output', axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print('Acuracia do modelo: ', gnb.score(X_test, y_test))

# Random Forest
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print('Acuracia do modelo: ', rfc.score(X_test, y_test))

# faça cross validation de 10 folds
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rfc, X, y, cv=10)

print('Acuracia do modelo: ', scores.mean())

# printe a acuracia, precisão, sensibilidade, f1-score e recall
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# faça a matriz de confusão
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

# faça a curva ROC

from sklearn.metrics import roc_curve, roc_auc_score

y_pred_proba = rfc.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

import matplotlib.pyplot as plt

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Random Forest ROC Curve')
plt.show()

# calcule a AUC
print('AUC: ', roc_auc_score(y_test, y_pred_proba))

# faça a curva Precision-Recall
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision-Recall Curve')
plt.show()

