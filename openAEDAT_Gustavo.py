import os
#import cv2
import sys
import struct
import numpy as np
import math

class aedatUtils:


    def loadaerdat(datafile='path.aedat', length=0, version="aedat", debug=1, camera='DVS128'):
        # constants
        aeLen = 8  # 1 AE event takes 8 bytes
        readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
        td = 0.000001  # timestep is 1us
        if(camera == 'DVS128'):
            xmask = 0x00fe  # Bin -> 0000 0000 1111 1110 || Dec -> 254
            xshift = 1
            ymask = 0x7f00  # Bin -> 0111 1111 0000 0000 || Dec -> 32512
            yshift = 8
            pmask = 0x1     # Bin -> 0000 0000 0000 0001 || Dec -> 1
            pshift = 0
        else:
            raise ValueError("Unsupported camera: %s" % (camera))

        aerdatafh = open(datafile, 'rb')
        k = 0  # line number
        p = 0  # pointer, position on bytes
        statinfo = os.stat(datafile)
        if length == 0:
            length = statinfo.st_size # Define 'length' = Tamanho do arquivo

        print("file size", length)
        
        # Verifica a versão do Python. 
        if sys.version[0] == '3':
            value = 35 # Se for >= 3 le o cabeçalho em binário.
        else:
            value = '#' # Se for < 3 le o cabeçalho como string.

        # header
        lt = aerdatafh.readline()
        while lt and lt[0] == value:
            p += len(lt)
            k += 1
            lt = aerdatafh.readline() 
            if debug >= 2:
                print(str(lt))
            continue
        
        # variables to parse
        timestamps = []
        xaddr = []
        yaddr = []
        pol = []
        
        # read data-part of file
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
        
        print(xmask, xshift, ymask, yshift, pmask, pshift)    
        while p < length:
            addr, ts = struct.unpack(readMode, s)
            # parse event type
            if(camera == 'DVS128'):     
                x_addr = (addr & xmask) >> xshift # Endereço x -> bits de 1-7
                y_addr = (addr & ymask) >> yshift # Endereço y -> bits de 8-14
                a_pol = (addr & pmask) >> pshift  # Endereço polaridade -> bit 0            
                if debug >= 3: 
                    print("ts->", ts) 
                    print("x-> ", x_addr)
                    print("y-> ", y_addr)
                    print("pol->", a_pol)

                timestamps.append(ts)
                xaddr.append(x_addr)
                yaddr.append(y_addr)
                pol.append(a_pol)
                    
            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen        

        if debug > 0:
            try:
                print("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
                n = 5
                print("showing first %i:" % (n))
                print("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
            except:
                print("failed to print statistics")
        t, x, y, p = np.array(timestamps), np.array(xaddr), np.array(yaddr), np.array(pol)
        return t - t[0], x, y, p



    def matrix_active(x, y, pol,filtered=None):
    
        matrix = np.zeros([128, 128]) # Cria uma matriz de zeros 128x128 onde serão inseridos os eventos
        pol = (pol - 0.5) # Os eventos no array de Polaridade passam a ser -0.5 ou 0.5
        
        if(len(x) == len(y)): # Verifica se o tamanho dos arrays são iguais   
            for i in range(len(x)):
                val = 0
                #se a flag do filtro for true. Os eventos serão somados
                #para que eles sejam normalizados pelo maior valor de um acumulo de eventos
                #e depois retirados por um limiar de ~30%
                if filtered == None or filtered == False:
                    val = pol[i]
                elif filtered == True:
                    val = 1
                matrix[x[i], y[i]] += val # insere os eventos dentro da matriz de zeros
        else:
            print("error x,y missmatch")    

        if filtered:
            maxValue = matrix.max()
            matrix = matrix/maxValue
            #matrix[matrix <= 0.5] = 0
            #matrix[np.logical_and(matrix > 0.1, matrix <= 0.3)] = 0.1
            #matrix[matrix >= 0.5] = 1
            matrix = (matrix * 255) # Normaliza a matriz para 8bits -> 0 - 255
        else:
            idx = 0
            limiar = 0.5
            for i in matrix: # Limita os eventos em dentro do limiar
                for j, v in enumerate(i):
                    if v > limiar:
                        matrix[idx][j] = limiar
                    if v < (limiar-1):
                        matrix[idx][j] = (limiar-1)
                idx += 1
            if limiar != 1:
                matrix = (matrix * 255) + 127.5 # Normaliza a matriz para 8bits -> 0 - 255
            
        return matrix

    def fila(t, x, y, p):  
    
        tore=0
        T= 5000000
        T_linha=150
    
        n = 4
        #declaração das matrizes P+ e P-
        TORE_positivo = np.full((128,128,n),[0,0,0,0],list)
        TORE_negativo = np.full((128,128,n),[0,0,0,0],list)
        TORE_fp = np.full((128,128,n),[0,0,0,0],list)
        TORE_fn = np.full((128,128,n),[0,0,0,0],list)
        
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
                for num in range(n):
                    tore = max(min(math.log(t[i] - TORE_positivo[x[i],y[i]][num] +1),T),math.log(T_linha))
                    aux_t = np.delete(TORE_fp[x[i],y[i]],0)
                    aux_t = np.append(aux_t,tore)
                TORE_fp[x[i],y[i]] = aux_t
            
                    
            elif  p[i]==0:
                    #l=len(TORE_negativo[x[i],y[i]])
                    aux = np.delete(TORE_negativo[x[i],y[i]],0)
                    aux = np.append(aux,t[i])
                    TORE_negativo[x[i],y[i]] = aux
                    for num in range(n):
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

    def getFrameTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp, Ti):
        aux = 0
        t2 = timeArray[(timeArray > Ti) & (timeArray <= Ti + timeStamp)]
        x2 = xPosArray[aux : aux + len(t2)]
        y2 = yPosArray[aux : aux + len(t2)]
        p2 = polArray[aux : aux + len(t2)]
        aux += len(t2)
        img = matrix_active(x2, y2, p2)
        img = rotateMatrix(img)
        return img

    def getFramesTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp,filtered=None):
        totalImages = []
        i, aux = 0, 0
        images = []
        totalTore = []
        tores = []
        totalImages_tore = []
        
        images_tore = []
        
        while (i + timeStamp) < abs(timeArray[-1]):
            t2 = timeArray[(timeArray > i) & (timeArray <= i + timeStamp)]
            x2 = xPosArray[aux : aux + len(t2)]
            y2 = yPosArray[aux : aux + len(t2)]
            p2 = polArray[aux : aux + len(t2)]
            aux += len(t2)
            aux_t_tore = t2[0]
            t_tore = t2 - aux_t_tore
            # print(t_tore)
            tore_p,tore_n = aedatUtils.fila(t_tore,x2,y2,p2)
            img = aedatUtils.matrix_active(x2, y2, p2,filtered)
            rotacao = aedatUtils.rotateMatrix(img)
            images.append(img)
            tores.append([tore_p,tore_n])	
            i += timeStamp
        totalTore.extend(tores)
        totalTore = np.array(totalTore)
        totalImages.extend(images)
        totalImages = np.array(totalImages)

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
            matrix = (matrix * 255)+127.5 # Normaliza a matriz para 8bits -> 0 - 255
            matrix = aedatUtils.rotateMatrix(matrix)
            images_tore.append(matrix)
        totalImages_tore.extend(images_tore)
        totalImages_tore = np.array(totalImages_tore)
        return totalImages_tore


    def rotateMatrix(mat): 
        N = len(mat)
        # Consider all squares one by one 
        for x in range(0, int(N/2)): 
            
            # Consider elements in group    
            # of 4 in current square 
            for y in range(x, N-x-1): 
                
                # store current cell in temp variable 
                temp = mat[x][y] 
    
                # move values from right to top 
                mat[x][y] = mat[y][N-1-x] 
    
                # move values from bottom to right 
                mat[y][N-1-x] = mat[N-1-x][N-1-y] 
    
                # move values from left to bottom 
                mat[N-1-x][N-1-y] = mat[N-1-y][x] 
    
                # assign temp to left 
                mat[N-1-y][x] = temp 

        return mat
    

    

   
    
