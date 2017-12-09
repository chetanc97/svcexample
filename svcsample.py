import numpy  as np
import matplotlib.pyplot as plt
from sklearn   import  svm
from sklearn import datasets
from matplotlib import style
style.use("ggplot")

digits =  datasets.load_digits()
x=[1,1.5,1.8 ,4 ]
y=[2 , 2.1 , 1.2 ,5]

#plt.scatter(x,y)
# plt.show()

XY = np.array([[1,2] ,
[1.5, 1.7],
[5,7],
[4,6],
])

label = [1, 1, 0, 0]

clf = svm.SVC(kernel = 'linear' , C = 1.0 )

clf.fit(XY , label )


print(clf.predict([5 , 8]))

w = clf.coef_[0]
print(w)


print(digits.data)

clf2 =  svm.SVC(gamma =  0.001 , C=100)

x,y =  digits.data[:-1] , digits.target[:-1]
clf.fit(x , y)

print('Predict  :  ' ,  clf.predict(digits.data[-1]))
plt.imshow(digits.images[-1] , cmap =  plt.cm.gray_r , interpolation =  "nearest"  )
plt.show()
