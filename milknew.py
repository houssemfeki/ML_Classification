#Affichiage de jeu données
import pandas as pd
dm = pd.read_csv('C:/Users/MSI/Desktop/milk/milknew.csv')
print(dm.to_string())
#Interprétation
print(dm.describe())
# Visualisation de données
import seaborn as sns
import matplotlib.pyplot as plt
#Relation entre ph et Grade (Countplot)
sns.countplot(x='Grade',hue='pH',data=dm)
plt.title('pH vs Grade')
plt.show()
#Relation entre Grade et Odor (Countplot)
sns.countplot(x='Grade',hue='Odor',data=dm)
plt.title('Odor vs Grade')
plt.show()
#Relation entre Grade et Odor (boxplot)
sns.boxplot(x="Grade", y="Odor", data=dm)
plt.xlabel("Grade")
plt.ylabel("Odor")
plt.title("Relation entre Grade et Odor")
plt.show()
#Relation entre Taste et Turbidity (Countplot)
sns.countplot(x='Taste',hue='Turbidity',data=dm)
plt.title('Odor vs Grade')
plt.show()
#Relation entre Colour et Turbidity (Countplot)
sns.countplot(x='Colour',hue='Turbidity',data=dm)
plt.title('Colour vs Grade')
plt.show()
#Matrice de corrélation
corr = dm.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True,cmap="coolwarm", square=True)
plt.title("Matrice de corrélation")
plt.show()

from sklearn.model_selection import train_test_split
#PréTraitement
x = dm.drop('Grade' , axis=1)
y = dm['Grade']

# Division des données avec 75% pour l'apprentissage et 25% pour le test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
#SVM
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train ,y_train)
print("SVM Model")
print("Training Accuracy :",svc.score(X_train,y_train))
print("Testing Accuracy :",svc.score(X_test,y_test))
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train,y_train)
print("KNN Model")
print("Training Accuracy :",knn.score(X_train,y_train))
print("Testing Accuracy :",knn.score(X_test,y_test))

from sklearn.model_selection import cross_val_score
import numpy as np
# Effectuez la validation croisée
cv_scores = cross_val_score(knn, x, y, cv=4)
# Affichez les résultats
print("kNN Model")
print("Scores de validation croisée:", cv_scores)
print("Moyenne des scores:", np.mean(cv_scores))
# Effectuez la validation croisée
cv_scores = cross_val_score(svc, x, y, cv=4)
# Affichez les résultats
print("SVM Model")
print("Scores de validation croisée:", cv_scores)
print("Moyenne des scores:", np.mean(cv_scores))
#
from sklearn.metrics import confusion_matrix, precision_score, recall_score
y_pred = knn.predict(X_test)
# Calculez la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
# Calculez la précision et le rappel
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
# Affichez les résultats
print("kNN Model")
print("Matrice de Confusion:\n", conf_matrix)
print("Précision: {:.2f}".format(precision))
print("Rappel: {:.2f}".format(recall))

from sklearn.metrics import confusion_matrix, precision_score, recall_score
#Prediction
y_pred = (svc.predict(X_test))
# Calculez la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
# Calculez la précision et le rappel
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
# Affichez les résultats
print("SVM Model")
print("Matrice de Confusion:\n", conf_matrix)
print("Précision: {:.2f}".format(precision))
print("Rappel: {:.2f}".format(recall))


import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
# Définissez les valeurs de n_neighbors à évaluer
param_range = np.arange(1, 21)
# Calculez les performances pour l'ensemble d'entraînement et l'ensemble de test
train_scores, test_scores = validation_curve(KNeighborsClassifier(), x, y, param_name="n_neighbors", param_range=param_range, cv=4)
# Calculez la moyenne et l'écart-type des scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Tracage de la courbe de validation
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label="Entraînement", color="blue", marker="o")
plt.fill_between(
    param_range,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    color="blue"
)
plt.plot(param_range, test_mean, label="Test", color="green", marker="o")
plt.fill_between(
    param_range,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.15,
    color="green"
)

# Ajoutez des détails au graphique
plt.title("Courbe de Validation pour KNeighborsClassifier")
plt.xlabel("Nombre de voisins (n_neighbors)")
plt.ylabel("Score de classification")
plt.legend(loc="best")
plt.show()

# Importation de modèle SVM
from sklearn.svm import SVC

# Définition des valeurs de C à évaluer (paramètre de régularisation pour SVM)
param_range_svm = np.arange(1,21)

# Calculez les performances pour l'ensemble d'entraînement et l'ensemble de test
train_scores_svm, test_scores_svm = validation_curve(
    SVC(), x, y, param_name="C", param_range=param_range_svm, cv=4)

# la moyenne et l'écart-type des scores
train_mean_svm = np.mean(train_scores_svm, axis=1)
train_std_svm = np.std(train_scores_svm, axis=1)
test_mean_svm = np.mean(test_scores_svm, axis=1)
test_std_svm = np.std(test_scores_svm, axis=1)

#la courbe de validation pour SVM
plt.figure(figsize=(10, 6))
plt.plot(param_range_svm, train_mean_svm, label="Entraînement", color="blue", marker="o")
plt.fill_between(
    param_range_svm,
    train_mean_svm - train_std_svm,
    train_mean_svm + train_std_svm,
    alpha=0.15,
    color="blue"
)
plt.plot(param_range_svm, test_mean_svm, label="Test", color="green", marker="o")
plt.fill_between(
    param_range_svm,
    test_mean_svm - test_std_svm,
    test_mean_svm + test_std_svm,
    alpha=0.15,
    color="green"
)

#Détails du graphique
plt.title("Courbe de Validation pour SVM")
plt.xlabel("Paramètre de Régularisation (C)")
plt.ylabel("Score de classification")
plt.xscale('log')  # Échelle logarithmique pour C
plt.legend(loc="best")
plt.show()


