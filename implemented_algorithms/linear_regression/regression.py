import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import mean_squared_error

def transform_data(x, d):
    phi = np.vander(x, d, increasing=True)
    return phi

def train_model(phi, y):
    phiT = phi.transpose()
    phiTphi = phiT @ phi
    pseudoInverse = np.linalg.inv(phiTphi)
    pseudoInversephiT = pseudoInverse @ phiT
    weights = pseudoInversephiT @ y
    return weights

def test_model(w, phi, y_true):
    y_pred = phi @ w
    #print("y_pred")
    #print(y_pred)
    return np.mean((y_true - y_pred)**2)

n = 200
x = np.random.uniform(0.0 , 6.0 , n)
y = np.sin(x) + np.sin (2*x) + np.random.normal(0.0 , 0.25 , n)
x_true = np.arange(0.0 , 6.0 , 6.0/n) # Uniform input data mesh
x_true_1 = np.arange(0.0, 6.0, 0.05) # Uniform input data mesh
y_true = np.sin(x_true) + np.sin (2 * x_true)
plt.plot( x_true , y_true , linestyle="-", linewidth =3.0 , color ="k")
plt.plot(x, y, marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='none')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('1. b. Dataset vs True function')
plt.show()
#exit()

dataset = x
indices = np.random.permutation(dataset.shape[0])
count = int(n*0.75)
training_idx = indices[:count]
test_idx = indices[count:]
x_trn= dataset[training_idx[0:]]
y_trn = y[training_idx[0:]]
x_tst = dataset[test_idx[0:]]
y_tst = y[test_idx[0:]]
plt.plot( x_trn , y_trn ,marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='none')
plt.xlabel('x-axis');
plt.ylabel('y-axis');
plt.title('1. c. Training Data')
plt.show()
#plt.gcf().clear()
plt.plot(x_tst, y_tst, marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='none')
plt.xlabel('x-axis');
plt.ylabel('y-axis');
plt.title('1. c. Test Data')
plt.show()
#exit()


plt.gcf().clear()
plt.xlabel('x_tst');
plt.ylabel('y_pred / y_tst');

colors = "bcmybcmybcmy"
color_index = 0
#plt.gcf().clear()

e_mse_tst = []
e_mse_trn = []
deg = []
for d in range (5, 16):                  #Iterate over poly. order 
    deg.append(d)
    phi_trn = transform_data(x_trn, d)   #Transform training data
    wtrn = train_model(phi_trn, y_trn)      #Learn model on training data
    phi_tst = transform_data(x_tst, d)   #Transform test data
    phi_tst_1 = transform_data(x_true_1, d)   #Transform test data
    phi_trn = transform_data(x_trn, d)   #Transform test data
    e_mse = test_model(wtrn, phi_tst, y_tst)#Evaluate model on test data
    e_mse_1 = test_model(wtrn, phi_trn, y_trn)#Evaluate model on test data
    e_mse_tst.append(e_mse)
    e_mse_trn.append(e_mse_1)
    y_pred = phi_tst @ wtrn
    y_pred_1 = phi_tst_1 @ wtrn
    lbl = 'y_tst true data(Training Set)'
    plt.plot( x_tst, y_tst, label = lbl, marker ='o', markerfacecolor ='r', markeredgecolor ='r', linestyle='none', color ="r", markersize=4)

    plt.title('Plot for d='+str(d)+' with e_mse='+str(round(e_mse, 6)))
    lbl = 'Order d='+str(d)+' e_mse='+str(round(e_mse, 6))
    #plt.plot(x_tst, y_pred, label=lbl, color = colors[color_index], marker ='o', markerfacecolor = colors[color_index], markeredgecolor =colors[color_index], linestyle ='none')
    plt.plot(x_true_1, y_pred_1, label=lbl, color = colors[color_index], linestyle ='-',  linewidth =3.0)

    plt.legend(loc='upper right')
    print("mean sqaured error for "+str(d)+" = "+str(e_mse))
    color_index += 1
    plt.show()

#plt.plot(deg, e_mse_tst, marker ='*', markerfacecolor ='g', markeredgecolor ='g', linestyle='none', color ="g", markersize=6)
#plt.plot(deg, e_mse_trn, marker ='o', markerfacecolor ='r', markeredgecolor ='r', linestyle='none', color ="r", markersize=6)
#
