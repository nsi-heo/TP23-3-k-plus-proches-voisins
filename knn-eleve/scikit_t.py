from sklearn.neighbors import KNeighborsClassifier
from pylab import *

x_test2 = 2.5
y_test2 = 0.75



def scikit_test(iris_test,iris_train,iris_test_species_record,previsions,k):
  print("\n\nScikit-learn")

  y= iris_train[['species']]
  x = iris_train[['petal_length','petal_width']]
  y=y.values.ravel()  #pour formater en un tableau à 1d

  model=KNeighborsClassifier(n_neighbors=k)

  model.fit(x,y)
  score = model.score(x,y) #pourcentage de pertinence du modèle

  print('pertinence modèle', score*100,' %')


  #Prédiction avec scikit-learn pour 1 point limite
  test= np.array([x_test2,y_test2]).reshape(1,2)
  print('Pt1 : ', model.predict(test)[0])


  ########################################################


  #Scikit-learn sur jeu de test iris_test
  result_scikit = model.predict(iris_test)
  print('\nresult Sci-kit : \n', result_scikit)


  ##########Comparaison  des méthodes############################
  erreur_scik = 0
  erreur_maison = 0
  difference_scik_maison = 0
  for i in range (len(result_scikit)):
      if result_scikit[i]!=iris_test_species_record[i]:
          erreur_scik +=1
      if previsions[i]!=iris_test_species_record[i]:
          erreur_maison +=1
      if previsions[i]!=result_scikit[i]:
          difference_scik_maison +=1

  print ("erreur scikit : ",erreur_scik , " erreur maison : ",erreur_maison, "  Différences scikit et maison : ",difference_scik_maison)
  ##############################################################################