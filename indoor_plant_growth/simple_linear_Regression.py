import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split  
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv("indoor_plant_growth.csv");

st.title(" 🌱 Indoor Plant growth using Linear Regression ");
st.info("""
### 📊 About Data

Synthetic data is used for this model.

This model has **1 feature**:

- Light Exposure (hours per day)

It predicts the **label**:

- Plant Growth (cm)
### 🤖 Model Working

This model uses **Linear Regression** to show the relationship
between light exposure and plant growth.

Linear Regression is a **supervised machine learning algorithm**
used to predict **continuous numerical values**.

This model is implemented using the **Ordinary Least Squares (OLS)**
technique.

OLS finds the **best-fit line** by minimizing the
**Sum of Squared Errors (SSE)**.

The slope is calculated using:

**b₁ = Cov(X,Y) / Var(X)**

The intercept is calculated using:

**b₀ = mean(Y) - b₁ × mean(X)**

The regression equation is:

**ŷ = b₀ + b₁X**

Where:
- **b₁** = slope — tells us how much the predicted value of Y changes when X increases by 1.
- **b₀** = intercept — tells us the predicted value of Y when X = 0.
- **ŷ** = predicted value of Y.

The best-fit line is the line that minimizes the
**Sum of Squared Errors (SSE)**.
""");
st.header("ML Model");
st.write(" Linear Regression Model to predict the weekly growth of the indoor plant based on the daily light exposure in hours");
st.header("Data Set");
st.write(df);

Y=df["Weekly_Growth_cm"];
X=df["Light_Hours_Per_Day"];
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=52);
#  70 percent for training and remaining 30 percent data for testing
fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(15,8));
axis1.scatter(x=x_train,y=y_train);
axis1.set_title("Relationship between Variable")
axis1.set_xlabel("Light Exposure");
axis1.set_ylabel("Weekly growth");
numer1= np.sum((x_train-np.mean(x_train))*(y_train-np.mean(y_train)));
denom1=np.sum( np.square (x_train-np.mean(x_train)));

b1=numer1/denom1;# slope it will tell how much change is in Y when we increase one unit of X
b0=np.mean(y_train)-b1*np.mean(x_train); #intercept ,base value when input is 0 what it will  predicts

# now our model learned b0 and b1 so
predicted_Y=b0+b1*x_test;
# this is predicted Y values growth per week of plants
# now we calculate the errors that is actual_y- predicted_y

e=y_test-predicted_Y; #residual error
SSE=np.sum(np.square(e));# Sum of squared error it will tell how far the actual data is  from predicted 
MSE=SSE/len(y_test);
SST=np.sum(np.square(y_test-np.mean(y_test))); # Total sum of squares, it will tell how far the actual data from mean of the data
RMSE=np.sqrt(MSE);
R2= 1- SSE/SST;
MAE=np.mean(np.abs(e));
st.subheader("Equation of Linear regression ")
st.write("Y=b0+b1*X");
col1,col2=st.columns(2);
with col1:
    st.write("b1 slope",f"{b1:.3f}");
    st.caption("when we increse the 1 unit of X how much increase in Y")
with col2:
    st.write("b0 intercept",f"{b0:.3f}");
    st.caption("value of Y when X=0")


axis2.scatter(x=predicted_Y,y=e);
axis2.axhline(0)
axis2.set_title("Residual error")
axis2.set_xlabel("Prediction");
axis2.set_ylabel("residual");

# for visualization 
x_line=np.linspace(np.min(X),np.max(X),100)
y_line=b0+b1*x_line;
sns.scatterplot(x=X, y=Y,ax=axis3)
sns.lineplot(x=x_line,y=y_line,ax=axis3);
axis3.set_title("best fit line")

st.pyplot(fig);

st.header("Model performance")
st.metric("R2 :",f"{R2*100:.1f}%");
st.caption("Explains how much variation in growth predicted by our model");
plt.tight_layout()
