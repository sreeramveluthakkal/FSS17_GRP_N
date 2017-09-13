### UNITY ID : smirhos

### 2. Explore the effects for different K in K Nearest Neighbor classification.  
I used a loop to run the code with different k:

``` python
k = [1, 5, 10, 50, 100]

for n in k:
    n_neighbors = n
    h = .02  # step size in the mesh
    
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

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
```
I have added the plots to this directory. The file name is `smirhos_<k>_uniform.png` and `smirhos_<k>_distance.png`.

Chooseing higher k values resulted in having tighter bounderies (noise has small effect on classification) but with small k value noise has a bigger effect on classification.

### 2. Explain how we should choose K. 
The optimum k is dependent on the data itself. Some resources online suggest square root of n but maybe another approuch is to try different k and do validation for best classification. K should be big enough to not be effected by small noise but should also be small enough to make best classification result.


### 3. Explore different kernels of Support Vector Machine.
I wrote a loop to run and create plot for different kernels. 
``` python
from sklearn import svm
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in kernels:
    
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    svc = svm.SVC(kernel=k)
    svc.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

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
```
The plots are in this directory and called `smirhos_<kernel>.png`.

Based on the plots, you can see `linear` classified following straight lines, and `poly` created curves.

### 4. Explain how we should choose the kernel.  
