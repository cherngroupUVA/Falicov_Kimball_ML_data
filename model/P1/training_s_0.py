#%%
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import os
import numpy as np
from matplotlib import pyplot as plt

path = "/scratch/sz8ea/FK/model/new_R10/model/P1/"

#%%
df_train = pd.read_csv("/scratch/sz8ea/FK/model/new_R10/relax/T0.05/job2/P1/input1.txt",sep='\t', header=None)
df_train_y = pd.read_csv("/scratch/sz8ea/FK/model/new_R10/relax/T0.05/job2/P1/output1.txt",sep='\t', header=None)

for i in range(1,11):
    for j in range(1,5):
        data_name_in = "/scratch/sz8ea/FK/model/new_R10/job"+str(i)+"/P1/"+"input"+str(j)+".txt" 
        data_name_out = "/scratch/sz8ea/FK/model/new_R10/job"+str(i)+"/P1/"+"output"+str(j)+".txt"
        #print(data_name_in)
        #print(data_name_out)       
        in_data = pd.read_csv(data_name_in,sep='\t', header=None)
        out_data = pd.read_csv(data_name_out,sep='\t', header=None)
        df_train = pd.concat([df_train,in_data])
        df_train_y = pd.concat([df_train_y,out_data])


for i in range(2,3):
    for j in range(2,101):
        data_name_in = "/scratch/sz8ea/FK/model/new_R10/relax/T0.05/job"+str(i)+"/P1/"+"input"+str(j)+".txt" 
        data_name_out = "/scratch/sz8ea/FK/model/new_R10/relax/T0.05/job"+str(i)+"/P1/"+"output"+str(j)+".txt"
        #print(data_name_in)
        #print(data_name_out)       
        in_data = pd.read_csv(data_name_in,sep='\t', header=None)
        out_data = pd.read_csv(data_name_out,sep='\t', header=None)
        df_train = pd.concat([df_train,in_data])
        df_train_y = pd.concat([df_train_y,out_data])



#df_test = pd.read_csv(data_path+"test_matrix_input.csv", header=None)
#df_test_y = pd.read_csv(data_path+"test_matrix_output.csv", header=None)
#%%




df_train_matrix = df_train.values
df_train_matrix_y = df_train_y.values
df_train_matrix_y = df_train_matrix_y[:,0]
df_train_matrix_y = df_train_matrix_y.reshape(-1,1)
#%%
scaler_x = StandardScaler()
scaler_y = StandardScaler()
#%%
df_train_scaled = scaler_x.fit_transform(df_train_matrix)
#%%
df_train_scaled_y = scaler_y.fit_transform(df_train_matrix_y)
#%%
import joblib
joblib.dump(scaler_x, path+'X_scaler_0.pkl')
joblib.dump(scaler_y, path+'Y_scaler_0.pkl')
#%%
df_train_scaled = df_train_scaled.reshape(-1,1,316) 
x_train = torch.DoubleTensor(df_train_scaled)
y_train = torch.DoubleTensor(df_train_scaled_y)
#%%



'''

df_test_matrix = df_test.values
df_test_matrix_y = df_test_y.values
#%%
df_test_scaled = scaler_x.transform(df_test_matrix)
df_test_scaled_y = scaler_y.transform(df_test_matrix_y)
#%%
df_test_matrix = df_test.values
df_test_matrix_y = df_test_y.values
#%%
df_test_scaled = scaler_x.transform(df_test_matrix)
df_test_scaled_y = scaler_y.transform(df_test_matrix_y)
#%%
x_test = torch.DoubleTensor(df_test_scaled)
y_test = torch.DoubleTensor(df_test_scaled_y)
'''

input_s = x_train.shape[1]
output_s = y_train.shape[1]
#%%
torch_data_set = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=32,
    shuffle=True
)
#%%

#%%
class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential( #(1,316)
            torch.nn.Conv1d(1,16,5,1,0), #(16,312)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,3)  #(16,104)
)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(16,32,5,1,1),  #(32,102)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,3)  #(32,34)
)
        #self.bn_1 = torch.nn.BatchNorm1d(512)
        self.hidden_1 = torch.nn.Linear(32*34, 256)
        #self.bn_2 = torch.nn.BatchNorm1d(256)
        self.hidden_2 = torch.nn.Linear(256, 128)
        #self.bn_3 = torch.nn.BatchNorm1d(128)
        self.hidden_3 = torch.nn.Linear(128,64)
        self.hidden_4 = torch.nn.Linear(64,32)
        #self.bn_4 = torch.nn.BatchNorm1d(64)
        self.output = torch.nn.Linear(32, output_size)

    def forward(self, x):
        #x = F.relu(self.bn_1(self.input(x)))
        #x = F.relu(self.bn_2(self.hidden_1(x)))
        #x = F.relu(self.bn_3(self.hidden_2(x)))
        #x = F.relu(self.bn_4(self.hidden_3(x)))     
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = self.output(x)
        return x
#%%
net = Net(input_s, output_s)
net = net.double()
if os.access(path+"model_0.pt",os.R_OK):
    net.load_state_dict(torch.load(path + "model_0.pt"))
lrspeed=1000
lr=1/lrspeed
optimizer = torch.optim.Adam(net.parameters(), lr)
loss_func = torch.nn.MSELoss()
#%%
image_data = []
step_count = 0
for epoch in range(40):
    for step, (feature, out) in enumerate(loader):
        optimizer.zero_grad()
        prediction = net(feature)
        print(prediction.shape)
        # print(energy_prediction.unsqueeze(0))
        #
        # # loss_1 = loss_func(energy_prediction, torch.sum(energy).float())
        # # loss_1.backward(retain_graph=True)
        #
#         force_prediction = -torch.autograd.grad(energy_prediction, temp_coordinate, create_graph=True)[0]
        # print(force_prediction, force)
        # energy_prediction_fit = energy_net(energy_prediction.unsqueeze(0))
        #
        # print(energy_prediction_fit)
        #
        # # loss_2 = loss_func(force_prediction, force[0])
        # # print(force_prediction.float())
        #
        # print(energy_prediction_fit)
        loss = loss_func(prediction, out)
        #net = net.eval()
        #loss_test = loss_func(net(x_test), y_test)
        # # loss = loss_func(energy_prediction.float(), torch.sum(energy).float())
        # # print(energy_prediction_temp)
        # # print(torch.sum((force_prediction - force[0])**2.0))
        #
        # # print('Epoch: ', epoch, '| Step: ', step, '| loss_1: ', loss_1)
        print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.data.numpy())
        image_data.append([step_count, loss.data.numpy()])
        step_count+=1
        print("******************************************")
        #
        loss.backward()
        optimizer.step()

#%%

#%%

#%%

#%%
image_data = np.asarray(image_data)
np.savetxt("error.csv", image_data, delimiter=",")
#%%
torch.save(net.state_dict(),path + "model_0.pt")
#%%

