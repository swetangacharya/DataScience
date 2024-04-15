import pandas as pd
import numpy as np
def predict(x,w,b): return w*x+b

def update_w_and_b(spendings,sales,w,b,alpha):
    dl_dw,dl_db,N=0.0,0.0,len(spendings)
    for i in range(N):
        dl_dw+=-2*spendings[i]*(sales[i]-(w*spendings[i]+b))
        dl_db+=-2*(sales[i]-(w*spendings[i]+b))

    #update w and b
        w=w-(1/float(N))*dl_dw*alpha
        b=b-(1/float(N))*dl_db*alpha

        return w,b

def train(spendings,sales,w,b,alpha,epochs):
    for e in range(epochs):
        w,b=update_w_and_b(spendings,sales,w,b,alpha)
        #logging
        if e%400==0:
            print("epoch:",e," loss: ", avg_loss(spendings,sales,w,b))

    return w,b

def read_parse_csv(file1):
    df=pd.read_csv(file1)
    
    Radio1=df['Radio']
    Sales1=df['Sales']
    return Radio1,Sales1
        

def avg_loss(spendings,sales,w,b):
    N,total_error=len(spendings),0.0
    for i in range(N):
        total_error+= (sales[i]-(w*spendings[i]+b))**2
    return total_error/float(N)





file1='advertising.csv'
x,y=read_parse_csv(file1)

w,b=train(x,y,0.0,0.0,0.001,15000)
x_new=23.0
y_new=predict(x_new,w,b)
print(f'pridected value is {y_new} when x value provided is {x_new}')
