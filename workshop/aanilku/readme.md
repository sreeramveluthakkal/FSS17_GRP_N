Explored the effect of k on running k-nearest neighbour:
As the value of k increases, the boundries are getting more defined, outliers are getting ignored.
This is because at higher values of k, the effect of noise reduces and the algorithm will go more with 
the general trend and ignore once off values which are mostly because of noise in the data.

A large value of k will however make the computation more expensive.
So, an optimal value of k would be sqrt(N) where N is the number of samples.

Ref: https://stackoverflow.com/questions/11568897/value-of-k-in-k-nearest-neighbour-algorithm


Explored SVM with different kernal functions:
With the linear model, the boundries are more rigid.
With the polinomial model, the boundries are more flexible.
With the rbf model also, we get flexible boundries but not as much as with the linear model.

The optimal kernal to use depends on the data. From what I read online, its always better to try out
different models starting with the simplest one. Select the one that works best for your data.



