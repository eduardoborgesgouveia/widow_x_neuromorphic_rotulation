import numpy as np
import copy
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from typing import Deque, Any
from collections import deque
from queue import LifoQueue
import math


def fila(t, x, y, p):  
    
    tore=0
    T= 5000000
    T_linha=150
   

    #declaração das matrizes P+ e P-
    TORE_positivo = np.full((128,128,4),[0,0,0,0],list)
    TORE_negativo = np.full((128,128,4),[0,0,0,0],list)
    TORE_fp = np.full((128,128,4),[0,0,0,0],list)
    TORE_fn = np.full((128,128,4),[0,0,0,0],list)
    
    #print('X: '+str(x))
    #print('Y: '+str(y))

    

    for i in range(len(t)):
        tore=0
        if p[i]==1:
            #k=len(TORE_positivo[x[i],y[i]])
            #k=k-1
            aux = np.delete(TORE_positivo[x[i],y[i]],0)
            aux = np.append(aux,t[i])
            TORE_positivo[x[i],y[i]] = aux
            for num in range(4):
                tore += max(min(math.log(t[i] - TORE_positivo[x[i],y[i]][num] +1),T),math.log(T_linha))
            aux_t = np.delete(TORE_fp[x[i],y[i]],0)
            aux_t = np.append(aux_t,tore)
            TORE_fp[x[i],y[i]] = aux_t
         
                
        elif  p[i]==0:
                #l=len(TORE_negativo[x[i],y[i]])
                aux = np.delete(TORE_negativo[x[i],y[i]],0)
                aux = np.append(aux,t[i])
                TORE_negativo[x[i],y[i]] = aux
                for num in range(4):
                    tore += max(min(math.log(t[i] - TORE_negativo[x[i],y[i]][num] +1),T),math.log(T_linha))
                aux_t = np.delete(TORE_fn[x[i],y[i]],0)
                aux_t = np.append(aux_t,tore)
                TORE_fn[x[i],y[i]] = aux_t
               
        
    
    #print(k)
    print('P=1 -> '+ str(TORE_positivo[9,0]))
    print('P=0 -> '+ str(TORE_negativo[9,0]))
    print('Tore p ->' +str(TORE_fp[9,0]))
    print('Tore n ->' +str(TORE_fn[9,0]))

    return TORE_fp,TORE_fn


def main():

    #Path to .aedat file
    path = 'Cup.aedat'
    
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)
    #print(type(x))

    tI = 200000
    images = aedatUtils.getFramesTimeBased(t,p,x,y,tI)
    
    fig,axarr = plt.subplots(1)
    handle = None
    
    
    matrix = np.zeros([128, 128])
    count = 0
    for tore in tores:
        matrix = np.zeros([128, 128])
        for x in range(128):            
            for y in range(128):                 
                if(tore[0][x,y,-1] > 0):
                    matrix[x,y] = tore[0][x,y,-1]
                elif(tore[1][x,y,-1] > 0):
                    matrix[x,y] = tore[1][x,y,-1]
        maxValue = matrix.max()
        matrix = matrix/maxValue
        # matrix[matrix != 0] = 1
        
        matrix = (matrix * 255) # Normaliza a matriz para 8bits -> 0 - 255
        matrix = aedatUtils.rotateMatrix(matrix)
        f = matrix.astype(np.uint8)
        if handle is None:
            handle = plt.imshow(np.dstack([f,f,f]))
        else:
            handle.set_data(np.dstack([f,f,f]))

        plt.pause(tI/1000000)
        plt.draw()

    print("saiu")

       



    



if __name__ == "__main__":
	main()
