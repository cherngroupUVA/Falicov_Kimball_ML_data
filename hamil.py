import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.special import expit
import scipy

thop = 1.0
Uhub = 2.0
L = 150
filling_fraction = 0.55
radius = 10.01

def mod(x, m):
    if 0 <= x < m:
        return x
    elif x < 0:
        return m - 1 - mod(-1 - x, m)
    else:
        return x % m

def sgn(x):
    if x>=0:
        return 1
    else:
        return -1

def fermi_density(x, kT, mu):
    alpha = (x-mu)/kT
    if kT<1e-15 or np.abs(alpha)>20:
        if x<mu:
            return 1.0
        else:
            return 0.0
    else:
        return 1.0/(np.exp(alpha)+1.0)



class Site:
    neighbor = []

    def __init__(self, i, Len):
        self.neighbor = []
        self.neighbor.append(mod(i % Len + 1, Len) + math.floor(i / Len) * Len)
        self.neighbor.append(i % Len + mod(math.floor(i / Len) - 1, Len) * Len)
        self.neighbor.append(mod(i % Len - 1, Len) + math.floor(i / Len) * Len)
        self.neighbor.append(i % Len + mod(math.floor(i / Len) + 1, Len) * Len)


    def __del__(self):
        class_name = self.__class__.__name__


class lattice_class:
    Len = 0
    Nl = 0
    pt = []

    def __init__(self, Len):
        self.Len = Len
        self.Nl = Len * Len
        for i in range(0, self.Nl):
            self.pt.append(Site(i, Len))

    def __del__(self):
        class_name = self.__class__.__name__


class SYSTEM:
    W1 = 0
    occup = 0
    empty = 0
    Hamil = 0
    eigE = 0
    #eigvec = 0
    kbT = 0 
    mu = 0
    total_energy = 0
    ion_num = 0

    def __init__(self):
        #self.omega = -6.0 + n * 12.0/(Nomega-1.0)
        self.total_energy = 0
        self.kbT = 0.05
        self.mu = 0
        self.W1 = np.zeros(L*L, dtype=np.int8)
        self.ion_num = 4225
        self.occup = np.zeros(self.ion_num,dtype=np.int32)
        self.empty = np.arange(L*L-self.ion_num,dtype=np.int32)
        self.Hamil = np.zeros((L*L, L*L), dtype=np.float64)
        #self.eigvec = np.zeros((L*L, L*L), dtype=np.float64)
        self.eigE = np.zeros(L*L, dtype=np.float64)


    def setinit(self):
        self.total_energy = 0
        self.mu = 0
        arr = np.arange(L*L)
        np.random.shuffle(arr)
        #self.ion_num = np.random.randint(1, L*L)
        occup_num=0
        empty_num=0
        for i in range(0,L*L):
            if i<self.ion_num:
                self.occup[occup_num]=arr[i]
                occup_num+=1
            else:
                self.empty[empty_num]=arr[i]
                empty_num+=1
            #self.occup = arr[0:self.ion_num]
            #self.empty = arr[self.ion_num:L*L]
        for i in range(0, self.ion_num):
            self.W1[self.occup[i]]=1
        for i in range(0, L*L-self.ion_num):
            self.W1[self.empty[i]]=0

    def calHamil(self, LAT):
        self.Hamil = np.zeros((L*L, L*L), dtype=np.float64)
        #self.eigvec = np.zeros((L*L, L*L), dtype=np.float64)
        self.eigE = np.zeros(L*L, dtype=np.float64)
        for i in range(0, L*L):
            if self.W1[i]==1:
                self.Hamil[i,i] = Uhub
            for j in range(0, 4):
                self.Hamil[i, LAT.pt[i].neighbor[j]] = -thop
        #self.eigE, self.eigvec = np.linalg.eig(self.Hamil)
        self.eigE = scipy.linalg.eigh(self.Hamil, eigvals_only=True)

    def compute_avg_density(self):
        a = (-(self.eigE-self.mu)/self.kbT).astype(np.float64)
        a = expit(a)
        return np.sum(a)/(L*L)

    def compute_chemical_potential(self):
        self.mu = 0
        x1 = np.min(self.eigE)
        x2 = np.max(self.eigE)
        max_count = 100
        cvt_value = 1e-12
        iter = 0
        while iter<max_count or np.abs(x2-x1)>cvt_value:
            self.mu = 0.5*(x1+x2)
            density = self.compute_avg_density()
            if density<=filling_fraction:
                x1 = self.mu
            else:
                x2 = self.mu
            iter+=1

    def compute_total_energy(self):
        self.total_energy = 0
        a = (-(self.eigE-self.mu)/self.kbT).astype(np.float64)
        a = expit(a)
        E = a*self.eigE
        self.total_energy = np.sum(E)


    def setW1(self):
        for i in range(0, self.occup.size):
            self.W1[self.occup[i]]=1
        for i in range(0, self.empty.size):
            self.W1[self.empty[i]]=0
    
    def descriptor_D4(self, pos, refsgn):
        arr = []
        #arr.append(refsgn[1]*refsgn[2]*refsgn[3])
        #arr.append(refsgn[2]*refsgn[4]*refsgn[5])
        #arr.append(refsgn[3]*refsgn[6]*refsgn[7])
        limit = (np.ceil(radius)+1).astype(np.int32)  
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    if j==0:       
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        arr.append(a+b+c+d)        
                        arr.append((a-b+c-d)*refsgn[2])   
                        arr.append((a-c)*refsgn[6]+(b-d)*refsgn[7])   
                        arr.append(((a-c)*refsgn[6]-(b-d)*refsgn[7])*refsgn[2]) 
                    elif i==j:
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        arr.append(a+b+c+d)        
                        arr.append((a-b+c-d)*refsgn[3])   
                        arr.append((a-c)*refsgn[4]+(b-d)*refsgn[5])   
                        arr.append(((a-c)*refsgn[4]-(b-d)*refsgn[5])*refsgn[3]) 
                    else:  
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        d = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        e = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        f = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        g = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        h = self.W1[label]   
                        arr.append(a+b+c+d+e+f+g+h)                                             #G1
                        arr.append((a-b+c-d+e-f+g-h)*refsgn[1])                                 #G2
                        arr.append((a+b-c-d+e+f-g-h)*refsgn[2])                                 #G3
                        arr.append((a-b-c+d+e-f-g+h)*refsgn[3])                                 #G4
                        arr.append((b-c-f+g)*refsgn[4]+(a+d-e-h)*refsgn[5])                     #G5
                        arr.append(((b-c-f+g)*refsgn[4]-(a+d-e-h)*refsgn[5])*refsgn[3])         #G6
                        arr.append((a-d-e+h)*refsgn[4]+(b+c-f-g)*refsgn[5])                     #G7
                        arr.append(((a-d-e+h)*refsgn[4]-(b+c-f-g)*refsgn[5])*refsgn[3])         #G8            
        return arr        
 
    def calref_D4(self, pos):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        a=b=c=d=e=f=g=h=0
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    factor = 0.5*(np.cos(np.sqrt(i*i+j*j)*np.pi/radius)+1.0)
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    a += self.W1[label] * factor
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    b += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    c += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    d += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    e += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    f += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    g += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    h += self.W1[label] * factor
        arr.append(a+b+c+d+e+f+g+h)   #A1
        arr.append(a-b+c-d+e-f+g-h)   #A2
        arr.append(a+b-c-d+e+f-g-h)   #B1
        arr.append(a-b-c+d+e-f-g+h)   #B2
        arr.append(b-c-f+g)           #E1x
        arr.append(a+d-e-h)           #E1y
        arr.append(a+b-e-f)           #E2x
        arr.append(c+d-g-h)           #E2y
        return arr                      

    def calref_D2(self, pos, direct):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        a=b=c=d=e=f=g=h=0
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    factor = 0.5*(np.cos(np.sqrt(i*i+j*j)*np.pi/radius)+1.0)
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    a += self.W1[label] * factor
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    b += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    c += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    d += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    e += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    f += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    g += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    h += self.W1[label] * factor
        if direct==1:                
            t1=g
            t2=h
            h=f
            g=e
            f=d
            e=c
            d=b
            c=a
            b=t2
            a=t1
        arr.append(a+b+e+f)   #A
        arr.append(c+d+g+h)   #A'
        arr.append(a-b+e-f)   #B1
        arr.append(c-d+g-h)   #B1'
        arr.append(a-b-e+f)   #B2
        arr.append(c+d-g-h)   #B2'
        arr.append(a+b-e-f)   #B3
        arr.append(c-d-g+h)   #B3'
        return arr 

    def descriptor_D2(self, pos, refsgn, direct):
        arr = []
        #arr.append(refsgn[2]*refsgn[4]*refsgn[6])
        limit = (np.ceil(radius)+1).astype(np.int32)  
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    if j==0:       
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(a+c)        
                        arr.append(b+d)   
                        arr.append((b-d)*refsgn[4])   
                        arr.append((a-c)*refsgn[6]) 
                    elif i==j:
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(a+b+c+d)        
                        arr.append((a-b+c-d)*refsgn[2])   
                        arr.append((a-b-c+d)*refsgn[4])   
                        arr.append((a+b-c-d)*refsgn[6]) 
                    else:  
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        d = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        e = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        f = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        g = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        h = self.W1[label]   
                        if direct==1:                
                            t1=g
                            t2=h
                            h=f
                            g=e
                            f=d
                            e=c
                            d=b
                            c=a
                            b=t2
                            a=t1
                        arr.append(a+b+e+f)        
                        arr.append(c+d+g+h)   
                        arr.append((a-b+e-f)*refsgn[2])   
                        arr.append((c-d+g-h)*refsgn[2])   
                        arr.append((a-b-e+f)*refsgn[4])   
                        arr.append((c+d-g-h)*refsgn[4]) 
                        arr.append((a+b-e-f)*refsgn[6])   
                        arr.append((c-d-g+h)*refsgn[6])                      
        return arr 

    def calref_C2_A(self, pos, direct):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        a=b=c=d=e=f=g=h=0
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    factor = 0.5*(np.cos(np.sqrt(i*i+j*j)*np.pi/radius)+1.0)
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    a += self.W1[label] * factor
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    b += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    c += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    d += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    e += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    f += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    g += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    h += self.W1[label] * factor
        if direct==1:              #empty b  
            t1=a
            t2=b
            a=c
            b=d
            c=e
            d=f
            e=g
            f=h            
            g=t1
            h=t2
        if direct==2:              #empty c  
            t1=a
            t2=b
            t3=c
            t4=d
            a=e
            b=f
            c=g
            d=h
            e=t1
            f=t2            
            g=t3
            h=t4
        if direct==3:                #empty d
            t1=g
            t2=h
            h=f
            g=e
            f=d
            e=c
            d=b
            c=a
            b=t2
            a=t1
        arr.append(a+b)   #A
        arr.append(c+h)   #A'
        arr.append(d+g)   #A''
        arr.append(e+f)   #A'''
        arr.append(a-b)   #B
        arr.append(c-h)   #B'
        arr.append(d-g)   #B''
        arr.append(e-f)   #B'''
        return arr 

    def descriptor_C2_A(self, pos, refsgn, direct):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    if j==0:       
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=b
                            b=c
                            c=d
                            d=t
                        if direct==2:
                            t1=a
                            t2=b
                            a=c
                            b=d
                            c=t1
                            d=t2
                        if direct==3:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(b+d)        
                        arr.append(a)   
                        arr.append(c)   
                        arr.append((b-d)*refsgn[4]) 
                    elif i==j:
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=b
                            b=c
                            c=d
                            d=t
                        if direct==2:
                            t1=a
                            t2=b
                            a=c
                            b=d
                            c=t1
                            d=t2
                        if direct==3:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(a+b)        
                        arr.append(c+d)   
                        arr.append((a-b)*refsgn[4])   
                        arr.append((c-d)*refsgn[4]) 
                    else:  
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        d = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        e = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        f = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        g = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        h = self.W1[label]   
                        if direct==1:              #empty b  
                            t1=a
                            t2=b
                            a=c
                            b=d                
                            c=e
                            d=f
                            e=g
                            f=h            
                            g=t1
                            h=t2
                        if direct==2:              #empty c  
                            t1=a
                            t2=b
                            t3=c
                            t4=d
                            a=e
                            b=f
                            c=g
                            d=h
                            e=t1
                            f=t2            
                            g=t3
                            h=t4
                        if direct==3:                #empty d
                            t1=g
                            t2=h
                            h=f
                            g=e
                            f=d
                            e=c
                            d=b
                            c=a
                            b=t2
                            a=t1
                        arr.append(a+b)        
                        arr.append(c+h)   
                        arr.append(d+g)   
                        arr.append(e+f)   
                        arr.append((a-b)*refsgn[4])   
                        arr.append((c-h)*refsgn[4]) 
                        arr.append((d-g)*refsgn[4])   
                        arr.append((e-f)*refsgn[4])                      
        return arr 

    def calref_C2_B(self, pos, direct):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        a=b=c=d=e=f=g=h=0
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    factor = 0.5*(np.cos(np.sqrt(i*i+j*j)*np.pi/radius)+1.0)
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    a += self.W1[label] * factor
                    label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    b += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    c += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                    d += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                    e += self.W1[label] * factor
                    label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                    f += self.W1[label] * factor
                    label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    g += self.W1[label] * factor
                    label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                    h += self.W1[label] * factor
        if direct==1:              #empty b  
            t1=a
            t2=b
            a=c
            b=d
            c=e
            d=f
            e=g
            f=h            
            g=t1
            h=t2
        if direct==2:              #empty c  
            t1=a
            t2=b
            t3=c
            t4=d
            a=e
            b=f
            c=g
            d=h
            e=t1
            f=t2            
            g=t3
            h=t4
        if direct==3:                #empty d
            t1=g
            t2=h
            h=f
            g=e
            f=d
            e=c
            d=b
            c=a
            b=t2
            a=t1
        arr.append(a+h)   #A
        arr.append(b+g)   #A'
        arr.append(c+f)   #A''
        arr.append(d+e)   #A'''
        arr.append(a-h)   #B
        arr.append(b-g)   #B'
        arr.append(c-f)   #B''
        arr.append(d-e)   #B'''
        return arr 

    def descriptor_C2_B(self, pos, refsgn, direct):
        arr = []
        limit = (np.ceil(radius)+1).astype(np.int32)  
        for i in range(1, limit):
            for j in range(0,i+1):
                if i*i+j*j <= radius*radius:
                    if j==0:       
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=b
                            b=c
                            c=d
                            d=t
                        if direct==2:
                            t1=a
                            t2=b
                            a=c
                            b=d
                            c=t1
                            d=t2
                        if direct==3:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(a+d)        
                        arr.append(b+c)   
                        arr.append((a-d)*refsgn[4])   
                        arr.append((b-c)*refsgn[4]) 
                    elif i==j:
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        d = self.W1[label]  
                        if direct==1:
                            t=a
                            a=b
                            b=c
                            c=d
                            d=t
                        if direct==2:
                            t1=a
                            t2=b
                            a=c
                            b=d
                            c=t1
                            d=t2
                        if direct==3:
                            t=a
                            a=d
                            d=c
                            c=b
                            b=t
                        arr.append(a)        
                        arr.append(c)   
                        arr.append(b+d)   
                        arr.append((b-d)*refsgn[4]) 
                    else:  
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        a = self.W1[label]
                        label = (mod(pos%L + i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        b = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        c = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)-i, L)*L).astype(np.int32)
                        d = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)-j, L)*L).astype(np.int32)
                        e = self.W1[label]
                        label = (mod(pos%L - i, L) + mod(np.floor(pos/L)+j, L)*L).astype(np.int32)
                        f = self.W1[label]
                        label = (mod(pos%L - j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        g = self.W1[label]
                        label = (mod(pos%L + j, L) + mod(np.floor(pos/L)+i, L)*L).astype(np.int32)
                        h = self.W1[label]   
                        if direct==1:              #empty b  
                            t1=a
                            t2=b
                            a=c
                            b=d                
                            c=e
                            d=f
                            e=g
                            f=h            
                            g=t1
                            h=t2
                        if direct==2:              #empty c  
                            t1=a
                            t2=b
                            t3=c
                            t4=d
                            a=e
                            b=f
                            c=g
                            d=h
                            e=t1
                            f=t2            
                            g=t3
                            h=t4
                        if direct==3:                #empty d
                            t1=g
                            t2=h
                            h=f
                            g=e
                            f=d
                            e=c
                            d=b
                            c=a
                            b=t2
                            a=t1
                        arr.append(a+h)        
                        arr.append(b+g)   
                        arr.append(c+f)   
                        arr.append(d+e)   
                        arr.append((a-h)*refsgn[4])   
                        arr.append((b-g)*refsgn[4]) 
                        arr.append((c-f)*refsgn[4])   
                        arr.append((d-e)*refsgn[4])                      
        return arr 

