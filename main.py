import math
import numpy as np
import hamil
from hamil import lattice_class
from hamil import SYSTEM
from hamil import mod
from matplotlib import pyplot as plt
from model_use import All_Model as AM
import sys

thop = 1.0
Uhub = 2.0
L = 150
filling_fraction = 0.55
radius = 10.01


LAT = lattice_class(L)
SYS = SYSTEM()
SYS.setinit()
AllMod = AM()
path = "/scratch/sz8ea/FK/ML_RW/largesize/150/T0.05/more_sim/"
fout = open(path+"config"+str(sys.argv[1])+".csv",'w')
np.savetxt(fout,SYS.W1,fmt="%d",delimiter=',')

old_energy = 0
SYS.setinit()
#SYS.calHamil(LAT)
#SYS.compute_chemical_potential()
#SYS.compute_total_energy()
#old_energy = SYS.total_energy
#print(0,old_energy)
stuck=0
'''
time_step = 1
while time_step < 100001:     
'''
for time_step in range(1,1000*SYS.ion_num+1):
#for time_step in range(1,11):
    pos_num = np.random.randint(0,SYS.ion_num)
    #pos_num=1
    pos = SYS.occup[pos_num]
    neigh_occup = 0
    out = np.zeros(4)
    for neigh_num in range(0,4):
        neigh_pos = LAT.pt[pos].neighbor[neigh_num]
        if SYS.W1[neigh_pos]==1:
            neigh_occup += 1
    if neigh_occup==0:       # 4 empty neigh
        refsgn = SYS.calref_D4(pos)
        refsgn = np.sign(refsgn)
        cond = 1
        for refsgn_num in range(1,refsgn.size):
            if refsgn[refsgn_num]==0:
                cond = 0
        if cond==1:
            arr = SYS.descriptor_D4(pos, refsgn)
            out = AllMod.model_use_func("P4", arr, refsgn, 0, SYS.kbT)
    if neigh_occup==1:          #3 empty neigh
        direct = 0
        for neigh_num in range(0,4):
            neigh_pos = LAT.pt[pos].neighbor[neigh_num]
            if SYS.W1[neigh_pos]==1:
                direct = neigh_num
                break
        refsgn = SYS.calref_C2_A(pos, direct)
        refsgn = np.sign(refsgn)
        cond = 1
        if refsgn[4]==0:
            cond = 0
        if cond==1:
            arr = SYS.descriptor_C2_A(pos, refsgn, direct)
            out = AllMod.model_use_func("P3", arr, refsgn, direct, SYS.kbT)
    if neigh_occup==2:          #2 empty neigh 
        a_pos = LAT.pt[pos].neighbor[0]
        b_pos = LAT.pt[pos].neighbor[1]
        c_pos = LAT.pt[pos].neighbor[2]
        d_pos = LAT.pt[pos].neighbor[3]
        if (SYS.W1[a_pos]+SYS.W1[c_pos]==2 or SYS.W1[b_pos]+SYS.W1[d_pos]==2):                                  #D2 sym
            if SYS.W1[a_pos]+SYS.W1[c_pos]==2:
                direct = 1
            else:
                direct = 0
            refsgn = SYS.calref_D2(pos, direct)
            refsgn = np.sign(refsgn)
            cond = 1
            if refsgn[2]*refsgn[4]*refsgn[6]==0:
                cond = 0
            if cond==1:
                arr = SYS.descriptor_D2(pos, refsgn, direct)
                out = AllMod.model_use_func("P2_A", arr, refsgn, direct, SYS.kbT)
        else:         #C''2 sym
            direct = 0
            for neigh_num in range(0,3):
                neigh_pos = LAT.pt[pos].neighbor[neigh_num]
                neigh_pos_next = LAT.pt[pos].neighbor[neigh_num+1]
                if SYS.W1[neigh_pos]+SYS.W1[neigh_pos_next]==2:
                    direct = neigh_num+1
                    break
            refsgn = SYS.calref_C2_B(pos, direct)
            refsgn = np.sign(refsgn)
            cond = 1
            if refsgn[4]==0:    
                cond = 0
            if cond==1:
                arr = SYS.descriptor_C2_B(pos, refsgn, direct)
                out = AllMod.model_use_func("P2_B", arr, refsgn, direct, SYS.kbT)
    if neigh_occup==3:        #1 empty neigh
        direct = 0
        for neigh_num in range(0,4):
            neigh_pos = LAT.pt[pos].neighbor[neigh_num]
            if SYS.W1[neigh_pos]==0:
                direct = neigh_num
                break
        refsgn = SYS.calref_C2_A(pos, direct)
        refsgn = np.sign(refsgn)
        cond = 1
        if refsgn[4]==0:
            cond = 0
        if cond==1:
            arr = SYS.descriptor_C2_A(pos, refsgn, direct)
            out = AllMod.model_use_func("P1", arr, refsgn, direct, SYS.kbT)



    rate = np.minimum(out, np.ones(4))/4.0
    update_rand = np.random.rand()
    if update_rand<rate[0]:
        SYS.W1[pos]=0
        SYS.W1[LAT.pt[pos].neighbor[0]]=1
        SYS.occup[pos_num] = LAT.pt[pos].neighbor[0]
    elif update_rand-rate[0]<rate[1]:
        SYS.W1[pos]=0
        SYS.W1[LAT.pt[pos].neighbor[1]]=1
        SYS.occup[pos_num] = LAT.pt[pos].neighbor[1]
    elif update_rand-rate[0]-rate[1]<rate[2]:
        SYS.W1[pos]=0
        SYS.W1[LAT.pt[pos].neighbor[2]]=1
        SYS.occup[pos_num] = LAT.pt[pos].neighbor[2]
    elif update_rand-rate[0]-rate[1]-rate[2]<rate[3]:
        SYS.W1[pos]=0
        SYS.W1[LAT.pt[pos].neighbor[3]]=1
        SYS.occup[pos_num] = LAT.pt[pos].neighbor[3]
    else:
        stuck+=1

    if time_step%SYS.ion_num == 0:
        np.savetxt(fout,SYS.W1,fmt="%d",delimiter=',')
        #SYS.calHamil(LAT)
        #SYS.compute_chemical_potential()
        #SYS.compute_total_energy()
        print(time_step)
    #time_step+=1

#SYS.calHamil(LAT)
#SYS.compute_chemical_potential()
#SYS.compute_total_energy()
print(SYS.total_energy)


print(stuck)
