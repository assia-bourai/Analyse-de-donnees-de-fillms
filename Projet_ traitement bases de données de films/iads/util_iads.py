# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importation de LabeledSet
from . import LabeledSet as ls

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
        - nb_points (int):  nombre de points de chaque classe à générer
    """
    #TODO: A Compléter
    base=ls.LabeledSet(2)
    pos=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    neg=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    for i in pos:
        base.addExample(i,1)

    for i in neg:
        base.addExample(i,-1)
    return base
    raise NotImplementedError("Please Implement this method")
    
def split(labeledSet,p_train):
    set_train=ls.LabeledSet(labeledSet.input_dimension)
    set_test =ls.LabeledSet(labeledSet.input_dimension)
   
    taille= labeledSet.size() # récupérer la taille du labledSet
    tab=[i for i in range(taille)]
    np.random.shuffle(tab)
    for i in range(len(tab)):
        if(i<p_train*taille):
            set_train.addExample(labeledSet.getX(tab[i]),labeledSet.getY(tab[i]))
        else:
            set_test.addExample(labeledSet.getX(tab[i]),labeledSet.getY(tab[i]))
            
    return set_train,set_test

def loss(self,dataset):
    "le but de nos algorithmes est de minimiser la fonction de coût"
    d=0
    for i in range(dataset.size()):
        d+=m.pow(dataset.getY(i)-np.dot(self.w,dataset.getX(i)),2)
    return d
        

   
    
