
# coding: utf-8

# In[4]:


import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris.feature_names)
print(iris.target_names)
print(np.unique(iris_y))


# In[5]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5



# In[6]:


plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# In[7]:


# Plot the distribution in boxplot
plt.boxplot(iris.data)
plt.xlabel('Features')
plt.ylabel('Cm')
plt.show()


# In[8]:


np.random.seed(0)
indices = np.random.permutation(len(iris_X))
# Split iris data in train and test data
# A random permutation, to split the data randomly
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 


# In[10]:


knn.predict(iris_X_test)
print iris_y_test


# In[11]:


n_neighbors = 1


# In[12]:




# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
h = .02  # step size in the mesh


# In[13]:


for n_neighbors in range(1,30,5):
    print "Plot with k = ",n_neighbors
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        knn = KNeighborsClassifier(n_neighbors, weights=weights)
        knn.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()


# In[14]:


from sklearn import svm


# In[15]:


svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)  
print svc.predict(iris_X_test)
print iris_y_test


# In[16]:


# SVM with linear kernel
svc = svm.SVC(kernel='poly')
svc.fit(iris_X_train, iris_y_train)  
print svc.predict(iris_X_test)
print iris_y_test


# In[17]:


# SVM with rbf kernel
svc = svm.SVC(kernel='rbf')
svc.fit(iris_X_train, iris_y_train)  
print svc.predict(iris_X_test)
print iris_y_test


# In[18]:


# SVM with rbf kernel
svc = svm.SVC(kernel='sigmoid')
svc.fit(iris_X_train, iris_y_train)  
print svc.predict(iris_X_test)
print iris_y_test


# In[21]:


for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        print "Plot with kernel= ",k
    
        # we create an instance of Neighbours Classifier and fit the data.
        svc = svm.SVC(kernel=k)
        svc.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z =  svc.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.show()


# In[ ]:





# In[ ]:




