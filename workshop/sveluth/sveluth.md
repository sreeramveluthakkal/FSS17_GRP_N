###UNITY ID - sveluth

###Explore the effects for different K in K Nearest Neighbor classification.  
The code was modified by looping through various values of k  
`for n_neighbors in [5,10,15,20,25]:`  
The images shown are named using the following convention  
unityid_k-#_weight.png  
for egs sveluth_k-20_uni.png means k is set as 20 and weight is uniform.

###Explain how we should choose K.  
K depends on the data. If the data is largely clustered together, a smaller K will be more accurate. If it is highly distributed without proper boundaries for clustering, then smaller Ks will cause pockets of clusters within other clusters while larger Ks will cause lesser accuracy because of overfitting.


###Explore different kernels of Support Vector Machine.
Ans - Iterated over the three kernels available for SVMs
It is observed that for the given data set, all the kernels' classifications are same. ie. [1 1 1 0 0 0 2 1 2 0]
Code given below: (complete code in sveluth.py)
REFERENCE - http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html  
`for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):  
    svc = svm.SVC(kernel=kernel)  
    svc.fit(iris_X_train, iris_y_train)  
    svc.predict(iris_X_test)  
    print iris_y_test  `

###Explain how we should choose the kernel.  
The polynomial and RBF should be used when the data is not linearly separable.  
Reference - http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html  