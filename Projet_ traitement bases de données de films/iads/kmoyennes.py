# -*- coding: utf-8 -*-

"""
BOURAI Assia_3804206
BENMESSAOUD Hamza_3802273
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(data):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    return (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.linalg.norm(v1-v2)
# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(mat):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return pd.DataFrame(mat.mean()).T
# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    s=0
    for i in range(len(df)):
        s+=dist_vect(df.iloc[i],centroide(df)) **2 
    return s

# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,df):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    longueur=len(df)
    valeur_rendre=[]
    for i in range (K):
        valeur_rendre.append(np.random.choice(longueur))
    l=list()
    for i in valeur_rendre:
        l.append(df.iloc[i])
    return pd.DataFrame(l)
    

# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(ex,cent):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    l=list()
    for i in range(len(cent)):
        l.append(dist_vect(ex,cent.iloc[i]))
    return np.argmin(l) 
# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(df,cent):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    mat=dict()
    l=list()
    for i in range(len(df)):
        l.append(plus_proche(df.iloc[i],cent))
    for j in range(len(cent)):
        mat[j]=[i for i in range(len(l)) if l[i]==j]
    return  mat  
# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(df,matrice):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    centroides = []
    for l in matrice.values():
        mat_df = df.iloc[l]
        centroides.append(centroide(mat_df))
    dico = dict()
    for c in df.columns.values:
        dico[c]=[ct[c][0] for ct in centroides]
    return pd.DataFrame(dico)
# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(df,matrice):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    inertie_g=0
    for val in matrice.values():
        mat_df = df.iloc[val]
        inertie_g+=inertie_cluster(mat_df)
    return inertie_g
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, df, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    centroides =initialisation(K,df) # chaque exemple est le centroide d'un cluster 
    i = 0 # itération
    inert = inertie_cluster(df)
    
    while(i < iter_max ):
        
        mat_affect = affecte_cluster(df, centroides) # affecter tous les exmples de df au cluster dont ils sont le plus proche du centroide 
        
        centroides = nouveaux_centroides(df, mat_affect)# calculer les nouveaux centroides
        
        inertie = inertie_globale(df, mat_affect) # calculer la nouvelle inertie 
        
        diff_inert = abs(inertie - inert)
        print("interation ",i," inertie ",inert," différence ",diff_inert)
        if (abs(diff_inert) < epsilon):
            break
        
        #incrementer les variables
        inert = inertie
        i += 1
    
    return centroides, mat_affect
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(df,centroides,matrice):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    labels = df.columns.values 
    x = labels[0]
    y = labels[1]
    
    for l in matrice.values():
        tmp_df = df.iloc[l]
        c = np.random.rand(3)#choisir une couleur au hasard k=3 
        plt.scatter(tmp_df[x], tmp_df[y], color=c)
    plt.scatter(centroides[x],centroides[y],color='r',marker='x')
# -------
def dist_intracluster(df):
    dist_max=0
    for i in range(len(df)):
        for j in range (i+1,len(df)):
            dist=km.dist_vect(df.iloc[i],df.iloc[j])
            if(dist_max<dist):
                dist_max=dist
    return dist_max      
#----------------------
def global_intraclusters(df,af):
    dist_max = 0
    for i in af.keys(): # af.keys= récupérer les clusters
        dist =  dist_intracluster( df.iloc[af.get(i)]) # af.get(i) = récupérer les exemples de chaque cluster ensuite récupérer le x et y de l'exemple 
        
        if (dist > dist_max):
            dist_max = dist
    return dist_max
#----------------------------
def sep_clusters(centres):
    dist_min=float('inf')
    for i in range(len(centres)):
        for j in range(i+1,len(centres)):
            dist = km.dist_vect(centres.iloc[i],centres.iloc[j])
            if (dist < dist_min):
                dist_min = dist
    return dist_min
#--------------------------------
def evaluation(index,df,cent,af):
    if(index=="Dunn"):
        return global_intraclusters(df,af)/sep_clusters(cent)
    elif (index=="XB"):
        return km.inertie_globale(df,af)/sep_clusters(cent)
    else :
        return None
#------------------------------------
def optimisation_K(dataframe, index, epsilon, iter_max):
    index_k=[2,3,4,5,6,7,8,9,10]
    resultat=[]
    for k in index_k:
        centers, affectations = km.kmoyennes(k, dataframe, epsilon, iter_max)
        #print(centers)
        #print(affectations)
        res = evaluation(index, dataframe, centers, affectations)
        resultat.append(res)
    print(resultat)
    plt.plot(index_k,resultat)
    plt.title("Evaluation du clustering avec l'index "+index+" en fonction de K\n")
    plt.xlabel("K")
    plt.ylabel("Valeur index")
    return index_k[np.argmin(resultat)]