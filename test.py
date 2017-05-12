# -*- coding: utf-8 -*-
"""
This script test the feature selection methods implemented by FCBF_module.
"""

from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.grid_search import GridSearchCV

classifiers = [('DecisionTree', DecisionTreeClassifier(), {'max_depth' : [5, 10, 15]}), 
              ('LogisticRegression', LogisticRegression(), {'C' : [0.1, 1, 10]})]
              

dataset = load_digits()
n_features = dataset.data.shape[1]
npieces = get_i(n_features)



for tag, clf, param_grid in classifiers:
    """
    No Feature Selection
    """
    grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
    grid.fit(dataset.data, dataset.target)    
    
    print "No Feature Selection"
    print "Classifer: {}".format(tag)
    print "Best score: {}\n".format(grid.best_score_)
    
    """
    FCBF
    """
    fcbf = FCBF()
    t0 = time.time()
    fcbf.fit(dataset.data, dataset.target)
    elapsed_t = time.time()-t0
    
    """
    Validation 
    """        
    grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
    grid.fit(dataset.data[:,fcbf.idx_sel], dataset.target)
    
    print "FCBF"
    print "Classifer: {}".format(tag)
    print "Best score: {}".format(grid.best_score_)
    print "Elapsed Time: {}\n".format(elapsed_t)    
    
    k = len(fcbf.idx_sel) #Number of selected features for FCBFK and FCBFiP
    
    """
    FCBF#
    """
    fcbfk = FCBFK(k = k)
    t0 = time.time()
    fcbfk.fit(dataset.data, dataset.target)
    elapsed_t = time.time()-t0
    
    """
    Validation 
    """        
    grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
    grid.fit(dataset.data[:,fcbfk.idx_sel], dataset.target)
    
    print "FCBF#"
    print "Classifer: {}".format(tag)
    print "Best score: {}".format(grid.best_score_)    
    print "Elapsed Time: {}\n".format(elapsed_t)    
    
    """
    FCBiP
    """
    for i in npieces:
        
        fcbfip = FCBFiP(npieces= i, k = k)
        t0 = time.time()
        fcbfip.fit(dataset.data, dataset.target)
        elapsed_t = time.time()-t0
        
        """
        Validation 
        """        
        grid = GridSearchCV(clf, param_grid, cv = 10, scoring = 'accuracy')
        grid.fit(dataset.data[:,fcbfip.idx_sel], dataset.target)
        
        print "FCBFiP with {} pieces".format(i)
        print "Classifer: {}".format(tag)
        print "Best score: {}".format(grid.best_score_)    
        print "Elapsed Time: {}\n".format(elapsed_t)   
"""
OUTPUT

No Feature Selection
Classifer: DecisionTree
Best score: 0.836393989983

FCBF
Classifer: DecisionTree
Best score: 0.82081246522
Elapsed Time: 1.53129601479

FCBF#
Classifer: DecisionTree
Best score: 0.823594880356
Elapsed Time: 1.55748701096

FCBFiP with 2 pieces
Classifer: DecisionTree
Best score: 0.827490261547
Elapsed Time: 2.3456659317

FCBFiP with 4 pieces
Classifer: DecisionTree
Best score: 0.797996661102
Elapsed Time: 1.23591303825

FCBFiP with 8 pieces
Classifer: DecisionTree
Best score: 0.820255982193
Elapsed Time: 0.638503074646

FCBFiP with 16 pieces
Classifer: DecisionTree
Best score: 0.816917084029
Elapsed Time: 0.343441963196

FCBFiP with 32 pieces
Classifer: DecisionTree
Best score: 0.821925431274
Elapsed Time: 0.196324110031

No Feature Selection
Classifer: LogisticRegression
Best score: 0.936004451864

FCBF
Classifer: LogisticRegression
Best score: 0.903171953255
Elapsed Time: 1.5462949276

FCBF#
Classifer: LogisticRegression
Best score: 0.903171953255
Elapsed Time: 1.56521987915

FCBFiP with 2 pieces
Classifer: LogisticRegression
Best score: 0.875904284919
Elapsed Time: 2.38653302193

FCBFiP with 4 pieces
Classifer: LogisticRegression
Best score: 0.894268224819
Elapsed Time: 1.23389911652

FCBFiP with 8 pieces
Classifer: LogisticRegression
Best score: 0.884251530328
Elapsed Time: 0.643393039703

FCBFiP with 16 pieces
Classifer: LogisticRegression
Best score: 0.90428491931
Elapsed Time: 0.346354961395

FCBFiP with 32 pieces
Classifer: LogisticRegression
Best score: 0.903728436283
Elapsed Time: 0.195168018341

"""
