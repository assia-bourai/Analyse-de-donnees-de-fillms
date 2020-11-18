# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
# Importation de la librairie iads
import iads as iads

# importation de LabeledSet
from iads import LabeledSet as ls

# importation de Classifiers
from iads import Classifiers as cl

# importation de utils
from iads import util_iads as ut
import random
import math

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
 
     #TODO: A Compléter

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        #TODO !!!!
        d=0
        j=dataset.size()
        for i in range(0,j):
            
            x=dataset.getX(i)
            if((dataset.getY(i)*self.predict(x))>0):
                d+=1
            
        return d/dataset.size()*100
           

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    #TODO: A Compléter
    
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.w=np.random.randn(input_dimension)  #génére le vecteur normal aléatoirement 
        self.input_dimension= input_dimension
      
        #raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        pred=0
        for i in range(0,self.input_dimension):
            pred+=x[i]*self.w[i]
        return pred
    
        #raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        raise NotImplementedError("Please Implement this method")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension= input_dimension
        self.k=k
        
        #raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        taille=self.labeledSet.size()
        tableau_dist=[]
        
        for i in range(0,taille):
            tableau_dist.append(np.linalg.norm(self.labeledSet.getX(i)-x))
            #print(dist)
               
            
        tableau_dist= np.argsort(tableau_dist)
        #print(tableau_dist)
        v=0.0
        for i in range(0,self.k):
            
            v+=self.labeledSet.getY(tableau_dist[i])
            
        return (v/self.k)
       
        

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.labeledSet=labeledSet  
      
        #raise NotImplementedError("Please Implement this method")

# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v
        
      
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        if (z<0):
            return -1
        return +1

        
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet=labeledSet
        tab= []
        # points d'apprentissage
        for i in range(labeledSet.size()):
            tab.append(i)
        np.random.shuffle(tab)  # mélange aléatoirement les points d'apprentissage 
        for i in tab:
            z=self.predict(self.labeledSet.getX(i))

            #if (z*self.labeledSet.getY(i)<=0):
            self.w = self.w + self.learning_rate * self.labeledSet.getX(i) * (self.labeledSet.getY(i)- np.sign(z))
    def loss(self,dataset):
        "le but de nos algorithmes est de minimiser la fonction de coût"
        d=0
        for i in range(dataset.size()):
            d+=m.pow(dataset.getY(i)-np.dot(self.w,dataset.getX(i)),2)
        return d
        
# ---------------------------
class ClassifierPerceptronRandom(Classifier):

    def __init__(self, input_dimension):

        """ Argument:

                - input_dimension (int) : dimension d'entrée des exemples

            Hypothèse : input_dimension > 0

        """  

        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions

        self.w = 2*v - 1

        self.w = self.w / np.linalg.norm(self.w)     # on normalise par la norme de w   
    
        #v = np.random.rand(input_dimension)     # en effet, cette façon de caulculer n'est pas correcte

        #self.w = (2* v - 1) / np.linalg.norm(v)  

    def predict(self, x):

        """ rend la prediction sur x (-1 ou +1)

        """
        z = np.dot(x, self.w)
        return z     

    def train(self,labeledSet):

        """ Permet d'entrainer le modele sur l'ensemble donné

        """        
        print("No training needed")


#-----------------------------
class ClassifierGradientStochastique(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        ##TODO
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v
        
      
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        
        return z

        
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet=labeledSet
        tab= []
        # points d'apprentissage
        for i in range(labeledSet.size()):
            tab.append(i)
        np.random.shuffle(tab)  # mélange aléatoirement les points d'apprentissage 
        for i in tab:
            z=self.predict(self.labeledSet.getX(i))

            
            self.w = self.w + self.learning_rate * self.labeledSet.getX(i) * (self.labeledSet.getY(i)- z)
                
    def loss(self,dataset):
        "le but de nos algorithmes est de minimiser la fonction de coût"
        d=0
        for i in range(dataset.size()):
            d+=m.pow(dataset.getY(i)-np.dot(self.w,dataset.getX(i)),2)
        return d

    
# ---------------------------
    
    
class ClassifierGradientBatch(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v
        
      
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        
        return z

        
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet=labeledSet
        tab= []
        # points d'apprentissage
        for i in range(labeledSet.size()):
            tab.append(i)
        np.random.shuffle(tab)  # mélange aléatoirement les points d'apprentissage 
        grad=np.zeros(self.input_dimension)
        for i in tab:
            z=self.predict(self.labeledSet.getX(i))
            grad+=self.labeledSet.getX(i) * (self.labeledSet.getY(i)- z)
            
        self.w = self.w + self.learning_rate *grad 
                
    def loss(self,dataset):
        "le but de nos algorithmes est de minimiser la fonction de coût"
        d=0
        for i in range(dataset.size()):
            d+=m.pow(dataset.getY(i)-np.dot(self.w,dataset.getX(i)),2)
        return d
#------------------------------------------
class ClassifierPerceptronKernel(Classifier):

    def __init__(self,dimension_kernel,learning_rate,kernel):

        """ Argument:

                - intput_dimension (int) : dimension d'entrée des exemples

                - learning_rate :

            Hypothèse : input_dimension > 0

        """

        self.w = np.zeros(dimension_kernel)

        self.e = learning_rate

        self.k = kernel

    def predict(self,x):

        """ rend la prediction sur x (-1 ou +1)

        """

        z = self.k.transform(x)

        res = np.dot(z, self.w)

        return res   

    def train(self,labeledSet):

        """ Permet d'entrainer le modele sur l'ensemble donné

        """

        ordre = np.arange(labeledSet.size())

        np.random.shuffle(ordre)

        for i in ordre:

            elem = labeledSet.getX(i)

            elem = self.k.transform(elem)

            z = np.dot(elem, self.w)

            if z * labeledSet.getY(i) <= 0:

                self.w += self.e * elem * labeledSet.getY(i)

                # La normalisation de w a été choisie pour garantir

                #que chaque modification de w est petite (de l'ordre de self.e) 

                #par rapport à la valeur précédente de w.

                self.w /= np.linalg.norm(self.w)   

    def train_bad(self,labeledSet):

        """ Version sans normalisation de w 

        """  

        i = np.random.randint(labeledSet.size())

        elem = labeledSet.getX(i)

        elem = self.k.transform(elem)

        z = np.dot(elem, self.w)

        if z * labeledSet.getY(i) <= 0:

            self.w = self.w + self.e *(elem * labeledSet.getY(i))

            

# --------------------------- 
class ClassifierGradientStoKernel(Classifier):

    """ Descent du gradient stochastique kernelisé

    """

    def __init__(self,dimension_kernel,learning_rate,kernel):

        """ Argument:

                - dimension_kernel (int) : dimension du kernel

                - learning_rate : e

            Hypothèse : dimension_kernel > 0

        """

        self.e = learning_rate

        #w initialisé de façon aléatoire

        self.w = (np.random.rand(dimension_kernel) - 0.5) * self.e

        self.k = kernel

       



    def predict(self,x):

        """ rend la prediction sur x 

        """

        z = self.k.transform(x)

        res = np.dot(z, self.w)

        return res



    

    def train(self,labeledSet):

        """ Permet d'entrainer le modele sur l'ensemble donné

        """

        # parcours des données du labeledSet en ordre aléatoire

        ordre = np.arange(labeledSet.size())

        np.random.shuffle(ordre)

        for i in ordre:

            elem = labeledSet.getX(i)

            z = self.predict(elem)

            elem = self.k.transform(elem)

            #pas necessaire de tester, on change w toujours

            self.w += self.e * (labeledSet.getY(i) - z) * elem 

            

            

    def loss(self, labeledSet):

        """Calcul de la fonction de loss sur le dataset labeledSet.

        """

        val_loss = 0

        for i in range(labeledSet.size()):

            elem = labeledSet.getX(i)

            z = self.predict(elem)

            val_loss += (labeledSet.getY(i) - z)**2

        return val_loss/labeledSet.size()  

    

# ---------------------------       

class ClassifierGradientBatchKernel(Classifier):

    """ Descent du gradient en batch kernelisé

    """
    def __init__(self,dimension_kernel,learning_rate,kernel):

        """ Argument:

                - dimension_kernel (int) : dimension du kernel

                - learning_rate : e

            Hypothèse : dimension_kernel > 0

        """
        self.e = learning_rate

        #w initialisé de façon aléatoire

        self.w = (np.random.rand(dimension_kernel) - 0.5) * self.e

        self.k = kernel

    def predict(self,x):

        """ rend la prediction sur x 

        """

        z = self.k.transform(x)

        res = np.dot(z, self.w)

        return res    

    def train(self,labeledSet):

        """ Permet d'entrainer le modele sur l'ensemble donné

        """

        # parcours des données du labeledSet

        gradient = np.zeros(self.w.size)

        for i in range(labeledSet.size()):

            elem = labeledSet.getX(i)

            z = self.predict(elem)

            elem = self.k.transform(elem)

            gradient += (labeledSet.getY(i) - z) * elem

        self.w += self.e * gradient / labeledSet.size()

    def loss(self, labeledSet):

        """Calcul de la fonction de loss sur le dataset labeledSet.

        """
        val_loss = 0

        for i in range(labeledSet.size()):

            elem = labeledSet.getX(i)

            z = self.predict(elem)

            val_loss += (labeledSet.getY(i) - z)**2

        return val_loss/labeledSet.size()  

# ---------------------------
def shannon(dist_proba):
    p=0
    k=len(dist_proba)
    if((p==None)or(k==1)):
        return 0.0
    for i in dist_proba:
        if(i!=0):
            p+=-i* math.log(i,k)
            #y.append(p)
    return p    
#------------------------------
def classe_majoritaire(self):
    nombre_pos=0
    nombre_neg=0
    for i in range(self.size()):
        if(self.getY(i)==+1):
            nombre_pos+=1
        else : 
            nombre_neg+=1
    if(nombre_pos>=nombre_neg):
        return 1
    return -1
#---------------------------------------
def entropie(the_set):
    nb=0;
    for i in range(the_set.size()):
        if(the_set.getY(i)==1):
            nb+=1
    # probabilité qu'il appartienne à la classe P
    prob_classP=0
    if(the_set.size()!=0):
        prob_classP=float(nb/the_set.size())
    # probabilité qu'il appartienne à la classe N    
    prob_classN=1-prob_classP      
    return shannon([prob_classP,prob_classN])
#------------------------------------------------------
def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)
#--------------------------------------------------------------------
def divise(Lset,att,seuil):
    E1=ls.LabeledSet(2)
    E2=ls.LabeledSet(2)
    
    for i in range(Lset.size()-1):
        if(Lset.getX(i)[att]<seuil):
            E1.addExample(Lset.getX(i),Lset.getY(i))
        else :
            E2.addExample(Lset.getX(i),Lset.getY(i))
    return E1,E2
#------------------------------------------------------------------
class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

#-------------------------------------------------------------------------       
class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
#-------------------------------------------------------------------
def construit_AD(LSet,epsilon):
    e = entropie(LSet)    #'entropie de Shannon de l'ensemble courant 
    un_arbre= ArbreBinaire()
    if(e<=epsilon):        
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
     
    else:
        tab_seuil=[]
        tab_entropie=[]
        for i in range(0,LSet.getInputDimension()):
            seuil_entropie = discretise(LSet,i) 
            tab_seuil.append(seuil_entropie[0]) #  le seuil pour chaque attribut 
            tab_entropie.append(seuil_entropie[1])#  l'entropie pour chaque attribut 
            
        tab = np.array(tab_entropie)
        att_min = tab.argmin() # attribut qui donne une valeur d'entropie minimale 
        gain=e-tab_entropie[att_min]
        if(gain < epsilon):
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))
        else:
            l1,l2 = divise(LSet,att_min,tab_seuil[att_min])
            un_arbre= ArbreBinaire()
            un_arbre1 = construit_AD(l1,epsilon)
            un_arbre2 = construit_AD(l2,epsilon)
            un_arbre.ajoute_fils(un_arbre1, un_arbre2, att_min, tab_seuil[att_min])  
    return un_arbre
#-----------------------------------------------------------------
def tirage(Lset_train,nb,b):
    l=list()
    if(b==False):
        l=random.sample(Lset_train,nb) # sans remise
    else:
        for i in range (nb):
            l.append(random.choice(Lset_train))
    
    return l
#---------------------------------------------------------
def echantillonLS(X,m,b):
    index=[i for i in range (X.size())] 
    echantillon=ls.LabeledSet(2)
    j=tirage(index,m,b)
    for i in j:
        echantillon.addExample(X.getX(i),X.getY(i))
    return echantillon
#--------------------------------------------------------------------

class ClassifierBaggingTree(Classifier):
    

    def __init__(self,B,pourcentage,entropie,boolean):
        self.B=B
        self.pourcentage=pourcentage
        self.boolean=boolean
        self.entropie=entropie       
        
        
    def train(self,labeledSet):
        self.arbres=set()
        for i in range (self.B):
            arbre=ArbreDecision(self.entropie)
            ech=echantillonLS(labeledSet,int(labeledSet.size()*self.pourcentage),self.boolean)
            arbre.train(ech)
            self.arbres.add(arbre)
            
        
           
    def predict(self, x):
        vote=0
        for arbre in self.arbres :
            vote+=arbre.predict(x)
        if(vote<0):
            return -1
        return 1
#---------------------------------------------------------------
def echantillonLSOOB(X,m,b):
    index=[i for i in range (X.size())]
    echantillon_train=ls.LabeledSet(2)
    echantillon_test=ls.LabeledSet(2)
    j=tirage(index,m,b)
    for i in range(0,X.size()):
        if( i in j ):
            echantillon_train.addExample(X.getX(i),X.getY(i))
        else:
            echantillon_test.addExample(X.getX(i),X.getY(i))
    return echantillon_train,echantillon_test
#-------------------------------------------------------------------
class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    def __init__(self,B,pourcentage,entropie,boolean):
        self.B=B
        self.pourcentage=pourcentage
        self.boolean=boolean
        self.entropie=entropie       
        
        
    def train(self,labeledSet):
        self.arbres=set()
        self.X=list() # la base de train qui servira à la construction des arbres de décision  
        self.T=list() # la base de test qui servira à tester les arbres de décision 
        for i in range (self.B):
            arbre=ArbreDecision(self.entropie)
            x,t=echantillonLSOOB(labeledSet,int(labeledSet.size()*self.pourcentage),self.boolean)
            arbre.train(x)
            self.arbres.add(arbre)
            self.X.append(x)
            self.T.append(t)
        
        
        
        
    def predict(self,exemple):
        pred = 0    
        for arbre in self.arbres :
            pred += arbre.predict(exemple)
        if(pred < 0):
            return -1
        return 1
    
    def accuracy(self):
        i=0
        accuracy_Test = 0
        accuracy_Train = 0
        for arbre in self.arbres:
            accuracy_Test += (1.0/self.B)*arbre.accuracy(self.T[i])
            accuracy_Train += (1.0/self.B)*arbre.accuracy(self.X[i])
            i+=1
        return accuracy_Test,accuracy_Train
#-------------------------------------------------------------------------------
def construit_AD_aleatoire(LSet,epsilon,nbatt):
    e = entropie(LSet)    #'entropie de Shannon de l'ensemble courant 
    un_arbre= ArbreBinaire()
    if(e<=epsilon):        
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
     
    else:
        tab_seuil=[]
        tab_entropie=[]
        for i in  tirage([i for i in range(0,LSet.getInputDimension())],nbatt,False):
            seuil_entropie = discretise(LSet,i) 
            tab_seuil.append(seuil_entropie[0]) #  le seuil pour chaque attribut 
            tab_entropie.append(seuil_entropie[1])#  l'entropie pour chaque attribut 
            
        tab = np.array(tab_entropie)
        att_min = tab.argmin() # attribut qui donne une valeur d'entropie minimale 
        gain=e-tab_entropie[att_min]
        if(gain < epsilon):
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))
        else:
            l1,l2 = divise(LSet,att_min,tab_seuil[att_min])
            un_arbre= ArbreBinaire()
            un_arbre1 = construit_AD(l1,epsilon)
            un_arbre2 = construit_AD(l2,epsilon)
            un_arbre.ajoute_fils(un_arbre1, un_arbre2, att_min, tab_seuil[att_min])  
    return un_arbre
#--------------------------------------------------------------------------
class ArbreDecisionAleatoire(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon,nbatt):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
        self.nbatt = nbatt
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD_aleatoire(set,self.epsilon,self.nbatt)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
#-------------------------------------------------------------------
