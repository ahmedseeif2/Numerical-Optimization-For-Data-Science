#!/usr/bin/env python
# coding: utf-8

# # Ahmed Mohamed Ceif
# 
# ## Practical Work 1
# ### Group 1 Alex

# For this practical work, the student will have to develop a Python program that is able to implement the gradient descent in order to achieve the linear regression of a set of datapoints.

# #### Import numpy, matplotlib.pyplot and make it inline

# In[59]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import genfromtxt
from sklearn.metrics import r2_score


# #### Read RegData csv file into numpy array  (check your data)
# ##### Data source
# https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# 

# In[9]:


Reg_Data_Array=genfromtxt("RegData.csv", delimiter=',')


# #### Explore your data

# In[10]:


print(len(Reg_Data_Array))
Reg_Data_Array


# #### Define variables X and y. Assign first column data to X and second column to y
# <b>Note:</b> X is the independent variable (input to LR model) and y is the dependent variable (output)

# In[116]:


X=Reg_Data_Array[:,0]
y=Reg_Data_Array[:,1]


# #### Explore your data

# In[12]:


print("X= " ,X)


# In[117]:


print("y= " ,y)
# print(y.shape)


# #### Plot the original data (scatter plot of X,y)

# In[17]:


plt.scatter(X,y)
plt.xlabel("X Data")
plt.ylabel("y Data")
plt.title("Plot X y DAta")
plt.show()


# ## LR Full Implementation

# ### Step1: Initialize parameters (theta_0 & theta_1) with random value or simply zero. Also choose the Learning rate. 

# ![image.png](attachment:image.png)

# In[20]:


theta_0=theta_1=0.1
learning_rate=0.01


# ### Step2: Use (theta_0 & theta_1) to predict the output h(x)= theta_0 + theta_1 * x.![image.png](attachment:image.png)
# #### Note: you will need to iterate through all data points

# In[21]:


def predicting(theta_0,theta_1,X):
    h_x=theta_0 +theta_1*X
    return h_x
predicting(theta_0,theta_1,X)


# ### Step3: Calculate Cost function 𝑱(theta_0,theta_1 ).![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[34]:


def loss_fun (y_predicted,y_actual):
    return np.sum((y_predicted -y_actual)**2)


# ### Step4: Calculate the gradient.![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[27]:


def grad_0(h_x,y,m):
    return np.sum(h_x-y)/m
def grad_1(h_x,y,m,x):
    return np.sum(h_x-y)*(x.T)/m


# ### Step5: Update the parameters (simultaneously).![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[79]:


def update_theta_0(theta_0,alpha,h_x,y,m):
    return theta_0 - alpha* grad_theta0(h_x,y,m)
def update_theta_1(theta_1,alpha,h_x,y,m,x):
    return theta_1 - alpha* grad_theta1(h_x,y,m,x)


# ### Step6: Repeat from 2 to 5 until converge to the minimum or achieve maximum iterations.![image.png](attachment:image.png)

# In[80]:


def GD(x_points,y_points,eps,alpha,theta_0,theta_1):
    m=len(x_points)
    grad_norm=1
    grad=0
    loss=0
    while(grad_norm>=eps):
        y=theta_0+theta_1*x_points
        loss=loss_fun(y,y_points)/(2*m) 
        theta_0=theta_0-alpha*np.sum((y-y_points))/m
        theta_1=theta_1-alpha*np.sum((y-y_points)*(x_points.T))/m
        grad=[np.sum((y-y_points))/m,np.sum((y-y_points)*(x_points.T))/m]
        grad_norm=np.linalg.norm(grad)
    return theta_0,theta_1,y


# #### Predict y values using the LR equation
# ##### h(x)= theta_0 + theta_1 * x

# In[81]:


theta_0,theta_1,y_predicted=GD(X,y,1e-3,0.01,2,1)
theta_0,theta_1,y_predicted


# #### Plot  LR equation output (fitted line) with the original data (scatter plot of X,y)

# In[82]:


plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.plot(X,y_predicted,color='r')
plt.title("Plot the original data ")
plt.show()


# #### Use R2 score to evaluate LR equation output
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# https://en.wikipedia.org/wiki/Coefficient_of_determination

# In[118]:


r2_score(y,y_predicted)
# print(y_predicted.shape)
# print(y.shape)


# ## GD vectorize Implementation
# ### Implement GD without iterate through data points i.e. use vector operations

# In[84]:


def GD_vectorized(x_points,y_points,eps,alpha,theta,epochs):
    m=len(x_points)
    grad_norm=1
    grad=0
    loss=0
    for i in range(epochs):
        # compute predicted y
        # compute the grad
        # update theta0,theta1
        # theta 1x2 X 13x2 >>1x2 * 2x13 >> 1x13
        #print(x_points.shape)
        #print(theta.shape)
        y=theta @ x_points.T# 1x13
        #print(y.shape)
        #print(y_points.shape) #13x1
        loss=np.sum(((y.T-y_points)**2))
        # 1/2m * (xtheta-y).T(xtheta-y)
        #print(grad.shape)
        grad= np.array([(np.mean((y - y_points)@ x_points,axis=0))])
        theta=(theta - alpha*grad).flatten()
    return theta,y


X=Reg_Data_Array[:,0]
y=Reg_Data_Array[:,1]
#print(new_x)
#np.reshape(X,(13,2))
new_x=X.reshape(len(X),-1)
#print(new_x.shape)
new_x1=np.append(np.ones(new_x.shape),new_x,axis=1) # shape 13x2
theta=np.array(([2,1]))
theta_vect=theta.reshape(1,len(theta)) # shape 2x1
#print(theta_vect.shape)
y=y.reshape(1,len(y))
theta,y_predicted_vec=GD_vectorized(new_x1,y,1e-3,0.0001,theta_vect,100)
theta


# In[85]:


Reg_Data_Array


# #### Plot the output and calculate R2 score
# ##### Make sure that you obtained the same results

# In[86]:


plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.plot(X,y_predicted_vec.T,color='r')
plt.title("Plot the original data with a fitted line")
plt.show()


# In[119]:


r2_score(y,y_predicted_vec)


# In[ ]:





# ## Plot loss function
# ### Repeat your last vectorized implementaion version and save loss for each iteration (epoch)

# In[88]:


def GD_vectorized(x_points,y_points,eps,alpha,theta,epochs=100):
    m=len(x_points)
    grad_norm=1
    grad=0
    loss=np.array([])
    for i in range(epochs):
        # compute predicted y
        # compute the grad
        # update theta0,theta1
        # theta 1x2 X 13x2 >>1x2 * 2x13 >> 1x13
        #print(x_points.shape)
        #print(theta.shape)
        y=theta @ x_points.T# 1x13
        #print(y.shape)
        #print(y_points.shape) #13x1
        loss=np.append(loss,np.sum(((y.T-y_points)**2)))
        # 1/2m * (xtheta-y).T(xtheta-y)
        #print(grad.shape)
        grad= np.array([(np.mean((y - y_points)@ x_points,axis=0))])
        theta=(theta - alpha*grad).flatten()
    return theta,y,loss


X=Reg_Data_Array[:,0]
y=Reg_Data_Array[:,1]
#print(new_x)
#np.reshape(X,(13,2))
new_x=X.reshape(len(X),-1)
#print(new_x.shape)
new_x1=np.append(np.ones(new_x.shape),new_x,axis=1) # shape 13x2
theta=np.array(([2,1]))
theta_vect=theta.reshape(1,len(theta)) # shape 2x1
#print(theta_vect.shape)
y=y.reshape(1,len(y))
theta,y_predicted_vec,loss=GD_vectorized(new_x1,y,1e-3,0.0001,theta_vect,100)


# ### Plot loss vs. iterations

# In[89]:


epochs=100
iterations = list(range(0,epochs))
plt.plot(iterations,loss,color='b')
plt.title("loss vs. iterations")
plt.show()


# ## Multivariate LR

# #### Read MultipleLR csv file into numpy array  (check your data)
# ##### Data source
# https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# 

# In[90]:


data=genfromtxt("MultipleLR.csv", delimiter=',')


# In[122]:


X=data[:,:-1]
y_mlr=data[:,-1:]


# In[127]:


X
print(X.shape)


# In[120]:


y_mlr
print(y_mlr.shape)


# In[ ]:





# ### Repeat your implementation but for more than one variable

# In[124]:


def GD_vectorized_mlR(x_points,y_points,alpha,theta,epochs):
    m=len(x_points)
    grad=0
    loss=np.array([])
    #print(y_points)
    for i in range(epochs):
        # compute predicted y
        # compute the grad
        # update theta
        y=(theta @  np.transpose(x_points))
        loss=np.append(loss,.5*len(y)*(np.linalg.norm(y-y_points))**2)
        grad= np.array([(np.mean((y - y_points)@ x_points,axis=0))]).flatten()
        #print(grad)
        theta=(theta - alpha*grad)
    return theta,y,loss


# #### Predict y values using the LR equation
# ##### h(x)= theta_0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3

# In[129]:



# add 1 to x
#print(theta.shape)
new_x=X.reshape(len(X),-1)
new_x1=np.append(new_x,np.ones((X.shape[0],1)),axis=1)
#print(new_x1)
y_mlr=y_mlr.reshape(len(y_mlr),-1)
#print(y)
theta=np.array([[0] * (new_x1.shape[1])])
theta,y_new,loss=GD_vectorized_mlR(new_x1,y_mlr,0.000001,theta,1000)
y_new


# ### Plot loss vs. iterations

# In[132]:


epochs=1000
iterations = list(range(0,epochs))
plt.plot(loss)
plt.title("loss vs. iterations")
plt.show()


# #### Use R2 score to evaluate LR equation output

# In[131]:


r2_score(y_mlr,y_new.T)


# # Bonus
# ## LR Using sklearn

# ### Single Variable

# #### Build a LR model usin linearmodel.LinearRegression() from sklearn library

# In[100]:


from sklearn.linear_model import LinearRegression
data =genfromtxt("RegData.csv",delimiter=',')
x = data[:,:-1]
y = data[:,-1:]


# In[ ]:





# #### Train the model (fit the model to the training data)

# In[101]:


model_reg= LinearRegression().fit(x, y)


# #### Predict y values using the trained model

# In[102]:


pred = model_reg.predict(x)


# #### Plot model output (fitted line) with the original data (scatter plot of X,y)

# In[103]:


plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("y")
plt.plot(x,pred,color='r')
plt.title("Plot the data (scatter plot of X,y)with a fitted line")
plt.show()


# #### Use R2 score to evaluate model output

# In[104]:


r2_score(y,pred)


# In[ ]:





# ### Repeat for Mulivariate

# In[105]:


data =genfromtxt("MultipleLR.csv",delimiter=',')
x = data[:,:-1]
y = data[:,-1:]


# In[106]:


model_reg= LinearRegression().fit(x, y)


# In[107]:


pred = model_reg.predict(x)


# In[108]:


r2_score(y,pred)


# In[ ]:





# In[ ]:




