import pandas
from pylab import *
import matplotlib.pyplot as plt
from math import pi
from sklearn.neighbors import KNeighborsClassifier
import random
from scikit_t import scikit_test

k = 3  #Nombre de plus proches voisins

#Echelle du graphe
ratio = 1

#Coorcdonnées d'un point limite de frontière

x_test2 = 2.5
y_test2 = 0.75
r_test2 = 0.85  #rayon autour du pt 2.5 0.75

#Chargement du jeu de données
iris = pandas.read_csv("iris.csv")
x = iris["petal_length"]
y = iris["petal_width"]
lab = iris["species"]

####Choix aléatoire d'un jeu de test########################################
#Mélange aléatoire (seed = None) des données, aléatoire controlé avec seed
seed = 98497848
iris = iris.sample(frac=1,random_state=seed)

#Réindexation du jeu de données mélangé
iris.reset_index(drop = True, inplace = True)
iris_test = iris.loc[:29][:]    #jeu de 30 fleurs à tester
iris_train = iris.loc[30:][:]   #jeu de 120 fleurs pour s'entrainer
###############################################################################

#Dessin du nuages de points de chaque famille de fleurs####################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(ratio)  # je choisi le ratio DX/DY pour les echelles des axes
ax.scatter(x[lab == 0], y[lab == 0], color='g', label='setosa', alpha=0.3)
ax.scatter(x[lab == 1], y[lab == 1], color='r', label='virginica', alpha=0.3)
ax.scatter(x[lab == 2], y[lab == 2], color='b', label='versicolor', alpha=0.3)

#Point test##############################################################
ax.scatter([x_test2], [y_test2], marker='X', color='k', label='inconnu', s=25)

ax.legend()

#Tracer de cerclces autour des points pour lesquels on veut faire une prédiction
t = np.linspace(0, 2 * pi, 100)
ax.plot(x_test2 + r_test2 * np.cos(t), y_test2 + r_test2 * np.sin(t), color='k')
##############################################################

####################Préparation du jeu de données et de test##################
iris_train['species'].replace([0, 1, 2], ['setosa', 'virginica', 'versicolor'],inplace=True)
iris_test['species'].replace([0, 1, 2], ['setosa', 'virginica', 'versicolor'],inplace=True)

#On enregistre la colonne species pour pouvoir comparer après les prédiction
iris_test_species_record = iris_test['species']
iris_test.drop('species', axis=1, inplace=True)  #On enlève la colonne species

###############################################################################


def distance(pt1, pt2):
    """
    Description :Calcule la distance entre deux points 

    paramètres :
        - pt1 : tuple contenant x1 et y1
        - pt2 : tuple contenant x2 et y2

    retour :
        - Distance entre deux points en tenant compte du ratio
    """
    return (sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))  
    #Distance entre les deux points 


def plus_proche_voisin(pt, train, k):
    """
    Explications : Cherche les k voisins les plus proche du point (pt) dans le jeu
    de données train et retourne l'espèce (species) la plus fréquente parmi les
    k voisins trouvés

    Paramètres :
        - pt : point sous forme de tuple
        - k : nombre de voisins les plus proches
        - train : jeu de données

    retour : l'espèce (species) la plus fréquente parmi les
    k voisins trouvés (str)

    """
    distances = []

    for ligne_train in train.itertuples():

        dist = distance((ligne_train.petal_length, ligne_train.petal_width),pt)
        distances.append(dist)

    train['dist'] = distances
    distances = iris_train[['dist', 'species']]
    distances = distances.sort_values(by='dist', ascending=True)
    head = distances.head(k)
    result = head['species'].value_counts()
    
    return result.index[0]  
    #renvoie l'index de la valeur la plus élevée (fréquence la plus élevée)


def eprouver_test(test, k):
    """
    Explications : prend un jeu de test et donne les prévisions sur les species
    de ce test (k plus proches voisins)

    Paramètres :
        - k : nombre de voisins les plus proches
        - test : jeu de données à tester dont on veut prédire les espèces

    retour : previsions (liste des prévisions)

    """
    previsions = []
    for ligne_test in test.itertuples():
        L_test = ligne_test.petal_length
        l_test = ligne_test.petal_width
        pt = (L_test, l_test)

        result = plus_proche_voisin(pt, iris_train, k)
        previsions.append(result)  #

    return (previsions)


def score_knn(k):
    """
    Explications : Renvoie le pourcentage de succès des prédictions "maison"

    Paramètres :
        - k : nombre de voisins les plus proches

    retour : tuple (pourcentage de prévisons,liste des prévisions)
    """
    score = 0
    previsions = eprouver_test(iris_test, k)

    taille_jeu_test = shape(iris_test)[0]
    for i in range(taille_jeu_test):
        if iris_test_species_record[i] == previsions[i]:
            score += 1
        else:
            print('erreur de prévision')
    return score / taille_jeu_test * 100, previsions


print("\n\nPrévisions maison avec k = ", k)
score, previsions = score_knn(k)
print(previsions)
print('score : ', score, ' %')

##############Prévisions####################################################
pt = (x_test2, y_test2)
print('Pt1 : ', plus_proche_voisin(pt, iris_train, k))

print("\nVraies valeurs : ")
print(list(iris_test_species_record))
#############################################################################

#############SCIKIT-LEARN#####################################################
scikit_test(iris_test,iris_train,iris_test_species_record,previsions,k)
#############################################################################

plt.show()
