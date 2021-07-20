import numpy as np
import copy
import os
import glob
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from matplotlib.patches import Rectangle
#from segmentationUtils import  segmentationUtils as su

def main():

    #Path to .aedat file
    path = 'WidowX_controll_dataset/Com fundo/Cup.aedat'
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)

    #time window of the frame (merging events)
    tI=200000 #200 ms
    fator_conversao_tempo = 1000000
    t_total_rec = t[-1]
    print(t_total_rec)

    qtde_frames = int(t_total_rec/tI)

    totalImages = []
    #get the t,p,x and y vectors and return a vector of frames agrouped in time intervals of tI
    totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

    #config for plotting the frames
    fig,ax = plt.subplots(1)
    handle = None
    imageVector = []

    pasta = "WidowX_controll_dataset/Sem fundo PNG/Cup/labels/Cup_{0}.txt"
    dir_name = "WidowX_controll_dataset/Sem fundo PNG/Cup/labels/"
    # caminhos = [os.path.join(pasta_, nome) for nome in os.listdir(pasta_)]
    # arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    # txt = [arq for arq in arquivos if arq.lower().endswith(".txt")]
    
    #print(list_of_files)

    #como sabemos que o movimento é sempre padronizado, devemos criar as bounding box conforme o movimento do widowX
    #esqueda - index:10 - tempo: 200ms | direita - index:20 - tempo: 200ms | centro - index:30 - tempo:200ms | sobe - index:62 - tempo:200ms - 
    #74 -centro
    # 103 aproxima
    # até o fim afasta
    #até 2s esquerda
    #de 2s até 4s - direita
    #de 4s até 6.6s - centro
    #de 6.6 a 8.6 - parado
    #de 6s até 12s - topo
    #de 12s até 15s - desce
    #de 15s até 20s - aproxima
    #de 20s até o fim - afasta
    #devemos determinar o tamanho do objeto na câmera
    width = 40
    height = 50
    x = 40
    y = 75
    z = 0
    x_aux = x
    y_aux = y
    z_aux = z
    width_aux = width
    height_aux = height
    temposEtapas = [0,2,4,6,8.2,10.4,12.4,14.4,20.8,22,24.6]
    #temposEtapas = [2,2,2,6,3,5,5]

    #as duas ultimas etapas de distância o aumento é em Z
    distanciaEtapas = [(x-5,y,z),(x+10,y,z),(x,y,z),(x,y,z),(x,y-6,z),(x,y-16,z),(x,y-4,z),(x,y-5,z+1),(x,y-9,z+1),(x,y-6,z-1)]
    velocidade = 0
    vetorPosicoes = []
    currentEtapa = "primeira etapa"
    lastEtapa = "primeira etapa"
    countEtapa = 1
    currentPos = distanciaEtapas[0]
    lastPos = (x,y,z)
    temp = (temposEtapas[1] - temposEtapas[0])
    for i in range(qtde_frames):
        tempo_atual = (tI/fator_conversao_tempo)*i
        print(tempo_atual)
        if(tempo_atual <= 24.5):
            if(tempo_atual>=temposEtapas[countEtapa-1] and tempo_atual < temposEtapas[countEtapa]):
                aux_dist_x = currentPos[0] - lastPos[0]
                aux_dist_y = currentPos[1] - lastPos[1]
                aux_dist_z = currentPos[2] - lastPos[2]
                x_aux = x_aux + aux_dist_x/temp - (aux_dist_z/temp)/2
                y_aux = y_aux + aux_dist_y/temp - (aux_dist_z/temp)/2
                width_aux = width_aux + (aux_dist_z/temp)
                height_aux = height_aux + (aux_dist_z/temp)
            else:
                print(tempo_atual)
                currentPos = distanciaEtapas[countEtapa]
                lastPos = distanciaEtapas[countEtapa - 1]
                temp = temposEtapas[countEtapa] - temposEtapas[countEtapa - 1]
                countEtapa = countEtapa + 1

            vetorPosicoes.append((x_aux,y_aux,width_aux,height_aux))

    print(vetorPosicoes)
    index_image = 0
    for f in totalImages:

        f = f.astype(np.uint8)
        imagem = copy.deepcopy(np.dstack([f,f,f]))
        
        index_image = index_image + 1
        print(index_image)
        if index_image < len(vetorPosicoes):
            aux_rect = vetorPosicoes[index_image]
            ax.add_patch( Rectangle((aux_rect[0],aux_rect[1]),
                            aux_rect[2], aux_rect[3],
                            fc ='none', 
                            ec ='g',
                            lw = 3) )

        if handle is None:
            handle = plt.imshow(imagem)
        else:
            handle.set_data(imagem)
        
        plt.pause(tI/1000000)
        plt.draw()
        ax.patches = []
        




if __name__ == "__main__":
	main()
