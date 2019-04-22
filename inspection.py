'''
$ python --version
Python 2.7.15
$ pip list
Package                       Version   
----------------------------- ----------
absl-py                       0.6.1     
astor                         0.7.1     
backports.functools-lru-cache 1.5       
backports.weakref             1.0.post1 
certifi                       2018.11.29
cycler                        0.10.0    
enum34                        1.1.6     
funcsigs                      1.0.2     
futures                       3.2.0     
gast                          0.2.0     
grpcio                        1.16.1    
h5py                          2.8.0     
joblib                        0.13.0    
Keras                         2.2.4     
Keras-Applications            1.0.6     
Keras-Preprocessing           1.0.5     
kiwisolver                    1.0.1     
Markdown                      3.0.1     
matplotlib                    2.2.3     
mock                          2.0.0     
mpmath                        1.0.0     
numpy                         1.15.4    
pandas                        0.23.4    
patsy                         0.5.1     
pbr                           5.1.1     
Pillow                        5.3.0     
pip                           18.1      
protobuf                      3.6.1     
pymc3                         3.5       
pyparsing                     2.3.0     
python-dateutil               2.7.5     
pytz                          2018.7    
PyYAML                        3.13      
scipy                         1.1.0     
setuptools                    39.0.1    
six                           1.11.0    
subprocess32                  3.5.3     
sympy                         1.3       
tensorboard                   1.12.0    
tensorflow                    1.12.0    
termcolor                     1.1.0     
Theano                        1.0.3     
tqdm                          4.28.1    
Werkzeug                      0.14.1    
wheel                         0.32.3
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cmap

import numpy as np
np.random.seed(206)

import theano
import theano.tensor as tt
import pymc3 as pm

from sympy import diff

lengthscale = 0.1
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=0.1");

lengthscale = 1
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=1");

lengthscale = 10
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=10");

plt.ylabel("Y");
plt.xlabel("X");
plt.legend(bbox_to_anchor=(0.6, 1), loc=2, borderaxespad=0.)

plt.savefig('testAccuracyVersusNumOfEpochs.png')

#********************************************************************************************

plt.clean()

lengthscale = 0.1
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)
print(type(cov))

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=0.1");

lengthscale = 1
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=1");

lengthscale = 10
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)

X = np.linspace(-1, 1, 1500)[:,None]
K = cov(X).eval()

plt.plot(X, pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K).random(size=1).T, label="ro=10");

plt.ylabel("Y");
plt.xlabel("X");
plt.legend(bbox_to_anchor=(0.6, 1), loc=2, borderaxespad=0.)

plt.savefig('trainAccuracyVersusNumOfEpochs.png')