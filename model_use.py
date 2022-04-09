import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import os
import joblib
import numpy as np
from hamil import mod

class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        '''
        self.input = torch.nn.Linear(input_size, 512)
        #self.bn_1 = torch.nn.BatchNorm1d(512)
        self.hidden_1 = torch.nn.Linear(512, 256)
        #self.bn_2 = torch.nn.BatchNorm1d(256)
        self.hidden_2 = torch.nn.Linear(256, 256)
        #self.bn_3 = torch.nn.BatchNorm1d(128)
        self.hidden_3 = torch.nn.Linear(256, 256)
  
        self.hidden_4 = torch.nn.Linear(256,256)
        self.hidden_5 = torch.nn.Linear(256,128)
        self.hidden_6 = torch.nn.Linear(128,64)
        #self.bn_4 = torch.nn.BatchNorm1d(64)
        #self.hidden_4 = torch.nn.Linear(512, 512)
        #self.hidden_5 = torch.nn.Linear(512, 512)

        #self.bn_5 = torch.nn.BatchNorm1d(32)
        self.output = torch.nn.Linear(64, output_size)
        '''
        self.conv1 = torch.nn.Sequential( #(1,316)
            torch.nn.Conv1d(1,16,5,1,0), #(16,312)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,3)  #(16,104)
)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(16,32,5,1,1),  #(16,66)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,3)  #(32,22)
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
        '''
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = F.relu(self.hidden_6(x))
        #x = F.relu(self.hidden_4(x))
        #x = F.relu(self.hidden_5(x))
        #x = F.relu(self.bn_1(self.input(x)))
        #x = F.relu(self.bn_2(self.hidden_1(x)))
        #x = F.relu(self.bn_3(self.hidden_2(x)))
        #x = F.relu(self.bn_4(self.hidden_3(x)))  
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        
        
        x = self.output(x)
        return x



class All_Model:
    net_P1 = 0
    net_P2_A_0 = 0
    net_P2_A_1 = 0
    net_P2_B_0 = 0
    net_P2_B_1 = 0
    net_P3_0 = 0
    net_P3_1 = 0
    net_P3_2 = 0
    net_P4_0 = 0
    net_P4_1 = 0
    net_P4_2 = 0
    net_P4_3 = 0
    X_scaler_P1 = 0
    Y_scaler_P1 = 0
    X_scaler_P2_A_0 = 0
    Y_scaler_P2_A_0 = 0
    X_scaler_P2_A_1 = 0
    Y_scaler_P2_A_1 = 0
    X_scaler_P2_B_0 = 0
    Y_scaler_P2_B_0 = 0
    X_scaler_P2_B_1 = 0
    Y_scaler_P2_B_1 = 0
    X_scaler_P3_0 = 0
    Y_scaler_P3_0 = 0
    X_scaler_P3_1 = 0
    Y_scaler_P3_1 = 0
    X_scaler_P3_2 = 0
    Y_scaler_P3_2 = 0
    X_scaler_P4_0 = 0
    Y_scaler_P4_0 = 0
    X_scaler_P4_1 = 0
    Y_scaler_P4_1 = 0
    X_scaler_P4_2 = 0
    Y_scaler_P4_2 = 0
    X_scaler_P4_3 = 0
    Y_scaler_P4_3 = 0

    def __init__(self): 
        self.net_P1 = Net(316, 1)
        self.net_P1 = self.net_P1.double()
        self.net_P2_A_0 = Net(316, 1)
        self.net_P2_A_0 = self.net_P2_A_0.double()
        self.net_P2_A_1 = Net(316, 1)
        self.net_P2_A_1 = self.net_P2_A_1.double()
        self.net_P2_B_0 = Net(316, 1)
        self.net_P2_B_0 = self.net_P2_B_0.double()
        self.net_P2_B_1 = Net(316, 1)
        self.net_P2_B_1 = self.net_P2_B_1.double()
        self.net_P3_0 = Net(316, 1)
        self.net_P3_0 = self.net_P3_0.double()
        self.net_P3_1 = Net(316, 1)
        self.net_P3_1 = self.net_P3_1.double()
        self.net_P3_2 = Net(316, 1)
        self.net_P3_2 = self.net_P3_2.double()
        self.net_P4_0 = Net(316, 1)
        self.net_P4_0 = self.net_P4_0.double()
        self.net_P4_1 = Net(316, 1)
        self.net_P4_1 = self.net_P4_1.double()
        self.net_P4_2 = Net(316, 1)
        self.net_P4_2 = self.net_P4_2.double()
        self.net_P4_3 = Net(316, 1)
        self.net_P4_3 = self.net_P4_3.double()
        path = "./model/"
        model_path = path+ "P1/"
        self.net_P1.load_state_dict(torch.load(model_path + "model_0.pt"))  
        self.net_P1 = self.net_P1.eval()
        self.X_scaler_P1 = joblib.load(model_path + "X_scaler_0.pkl")
        self.Y_scaler_P1 = joblib.load(model_path + "Y_scaler_0.pkl")  
        model_path = path+"P2_A/"
        self.net_P2_A_0.load_state_dict(torch.load(model_path + "model_0.pt"))  
        self.net_P2_A_1.load_state_dict(torch.load(model_path + "model_1.pt"))  
        self.net_P2_A_0 = self.net_P2_A_0.eval()
        self.net_P2_A_1 = self.net_P2_A_1.eval()
        self.X_scaler_P2_A_0 = joblib.load(model_path + "X_scaler_0.pkl")
        self.Y_scaler_P2_A_0 = joblib.load(model_path + "Y_scaler_0.pkl")         
        self.X_scaler_P2_A_1 = joblib.load(model_path + "X_scaler_1.pkl")
        self.Y_scaler_P2_A_1 = joblib.load(model_path + "Y_scaler_1.pkl")    
        model_path = path + "P2_B/"
        self.net_P2_B_0.load_state_dict(torch.load(model_path + "model_0.pt"))  
        self.net_P2_B_1.load_state_dict(torch.load(model_path + "model_1.pt"))
        self.net_P2_B_0 = self.net_P2_B_0.eval()
        self.net_P2_B_1 = self.net_P2_B_1.eval()   
        self.X_scaler_P2_B_0 = joblib.load(model_path + "X_scaler_0.pkl")
        self.Y_scaler_P2_B_0 = joblib.load(model_path + "Y_scaler_0.pkl")         
        self.X_scaler_P2_B_1 = joblib.load(model_path + "X_scaler_1.pkl")
        self.Y_scaler_P2_B_1 = joblib.load(model_path + "Y_scaler_1.pkl")   
        model_path = path + "P3/"
        self.net_P3_0.load_state_dict(torch.load(model_path + "model_0.pt"))  
        self.net_P3_1.load_state_dict(torch.load(model_path + "model_1.pt"))  
        self.net_P3_2.load_state_dict(torch.load(model_path + "model_2.pt"))  
        self.net_P3_0 = self.net_P3_0.eval()
        self.net_P3_1 = self.net_P3_1.eval()
        self.net_P3_2 = self.net_P3_2.eval()
        self.X_scaler_P3_0 = joblib.load(model_path + "X_scaler_0.pkl")
        self.Y_scaler_P3_0 = joblib.load(model_path + "Y_scaler_0.pkl")   
        self.X_scaler_P3_1 = joblib.load(model_path + "X_scaler_1.pkl")
        self.Y_scaler_P3_1 = joblib.load(model_path + "Y_scaler_1.pkl")   
        self.X_scaler_P3_2 = joblib.load(model_path + "X_scaler_2.pkl")
        self.Y_scaler_P3_2 = joblib.load(model_path + "Y_scaler_2.pkl")
        model_path = path+"P4/"
        self.net_P4_0.load_state_dict(torch.load(model_path + "model_0.pt"))  
        self.net_P4_1.load_state_dict(torch.load(model_path + "model_1.pt"))  
        self.net_P4_2.load_state_dict(torch.load(model_path + "model_2.pt"))  
        self.net_P4_3.load_state_dict(torch.load(model_path + "model_3.pt"))  
        self.net_P4_0 = self.net_P4_0.eval()
        self.net_P4_1 = self.net_P4_1.eval()
        self.net_P4_2 = self.net_P4_2.eval()
        self.net_P4_3 = self.net_P4_3.eval()
        self.X_scaler_P4_0 = joblib.load(model_path + "X_scaler_0.pkl")
        self.Y_scaler_P4_0 = joblib.load(model_path + "Y_scaler_0.pkl")   
        self.X_scaler_P4_1 = joblib.load(model_path + "X_scaler_1.pkl")
        self.Y_scaler_P4_1 = joblib.load(model_path + "Y_scaler_1.pkl")   
        self.X_scaler_P4_2 = joblib.load(model_path + "X_scaler_2.pkl")
        self.Y_scaler_P4_2 = joblib.load(model_path + "Y_scaler_2.pkl")
        self.X_scaler_P4_3 = joblib.load(model_path + "X_scaler_3.pkl")
        self.Y_scaler_P4_3 = joblib.load(model_path + "Y_scaler_3.pkl")         

    def model_use_func(self, model_name, df_train_x, ref, direct, kbT):
        out_put = np.zeros(4)
        df_train_x = np.asarray(df_train_x)
        df_train_x = df_train_x.reshape(-1,316)
        if model_name == "P1":
            df_train_scaled = self.X_scaler_P1.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P1(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P1.inverse_transform(out_pred_nt)
            out_put[direct] = np.exp(-out_pred_t[0,0]/kbT)
        if model_name == "P2_A":
            df_train_scaled = self.X_scaler_P2_A_0.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P2_A_0(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P2_A_0.inverse_transform(out_pred_nt)
            a = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P2_A_1.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P2_A_1(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P2_A_1.inverse_transform(out_pred_nt)
            c = out_pred_t[0,0]
            if direct == 0:
                out_put[0] = np.exp(-(a+c*ref[6])/2.0/kbT)
                out_put[2] = np.exp(-(a-c*ref[6])/2.0/kbT)
            else:
                out_put[1] = np.exp(-(a+c*ref[6])/2.0/kbT)
                out_put[3] = np.exp(-(a-c*ref[6])/2.0/kbT)       
        if model_name == "P2_B":
            df_train_scaled = self.X_scaler_P2_B_0.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P2_B_0(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P2_B_0.inverse_transform(out_pred_nt)
            a = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P2_B_1.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P2_B_1(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P2_B_1.inverse_transform(out_pred_nt)
            c = out_pred_t[0,0]
            out_put[mod(direct+1,4)] = np.exp(-(a+c*ref[4])/2.0/kbT)
            out_put[mod(direct+2,4)] = np.exp(-(a-c*ref[4])/2.0/kbT)
        if model_name == "P3":
            df_train_scaled = self.X_scaler_P3_0.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P3_0(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P3_0.inverse_transform(out_pred_nt)
            b = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P3_1.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P3_1(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P3_1.inverse_transform(out_pred_nt)
            c = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P3_2.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P3_2(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P3_2.inverse_transform(out_pred_nt)
            d = out_pred_t[0,0]
            out_put[mod(direct+1,4)] = np.exp(-(b+d*ref[4])/2.0/kbT)
            out_put[mod(direct+2,4)] = np.exp(-c/kbT)
            out_put[mod(direct+3,4)] = np.exp(-(b-d*ref[4])/2.0/kbT)
        if model_name == "P4":
            df_train_scaled = self.X_scaler_P4_0.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P4_0(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P4_0.inverse_transform(out_pred_nt)
            a = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P4_1.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P4_1(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P4_1.inverse_transform(out_pred_nt)
            b = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P4_2.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P4_2(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P4_2.inverse_transform(out_pred_nt)
            c = out_pred_t[0,0]
            df_train_scaled = self.X_scaler_P4_3.transform(df_train_x)
            df_train_scaled = df_train_scaled.reshape(-1,1,316)
            x_train = torch.DoubleTensor(df_train_scaled)
            out_pred_nt = self.net_P4_3(x_train).detach().numpy()
            out_pred_t = self.Y_scaler_P4_3.inverse_transform(out_pred_nt)
            d = out_pred_t[0,0]
            out_put[0] = np.exp(-(a+b*ref[2]+(c+d*ref[2])*ref[6])/4.0/kbT)
            out_put[1] = np.exp(-(a-b*ref[2]+(c-d*ref[2])*ref[7])/4.0/kbT)
            out_put[2] = np.exp(-(a+b*ref[2]-(c+d*ref[2])*ref[6])/4.0/kbT)
            out_put[3] = np.exp(-(a-b*ref[2]-(c-d*ref[2])*ref[7])/4.0/kbT)
        return out_put



