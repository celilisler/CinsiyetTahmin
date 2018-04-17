"""
Created on Mon Apr 16 01:08:33 2018

@author: yusuf
"""

import numpy as np


#sigmoid fonksiyonu
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))


#eğitim verilerimiz
"""
kişilerin boy,kilo ve ayak numarası verilerek onların erkek-kadın olma durumları öğretilir.
"""
X =  np.array([[1.81, .80, .44], [1.77, .70, .43], [1.60, .60, .38], [1.54, .54, .37], [1.66, .65, .40],
     [1.90, .90, .47], [1.75, .64, .39], [1.77, .70, .40], [1.59, .55, .37], [1.71, .75, .42],
     [1.81, .85, .43]])
y = np.array([[1],[1], [0], [0], [0], [1], [0], [1], [0], [1], [1]])   

np.random.seed(10)

# mean 0 olacak şekilde rastgele W ağırlık değerleri atayalım
syn0 = 2*np.random.random((3,10)) - 1
syn1 = 2*np.random.random((10,1)) - 1

for j in range(60000):

	# ileri besleme ile 0, 1, and 2 katmanların hesaplanması
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # output değere ne kadar uzak kaldığımız
    l2_error = y - l2
    
    if (j% 5000) == 0:
        print("Doğruluk Oranı % :", str(100-100*np.mean(np.abs(l2_error))))
        
    # ne kadar düzeltilecek
    l2_delta = l2_error*nonlin(l2,deriv=True)

    #ağırlıklara göre hata oranının dağıtılması
    l1_error = l2_delta.dot(syn1.T)
    
    # l1 ne kadar düzeltilecek
    l1_delta = l1_error * nonlin(l1,deriv=True)

    #yeni ağırlıklarımız
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


def  test(_X):    
    l0 = _X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    sonuc = l2
    if sonuc[0][0] > 0.5:
        print("erkek: ",sonuc[0][0])
    else:
        print("kadın: ",sonuc[0][0])