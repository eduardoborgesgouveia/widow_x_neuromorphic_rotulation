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
    path = 'WidowX_controll_dataset/Sem fundo/Cup.aedat'
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)

    #time window of the frame (merging events)
    tI=100000 #100 ms
    t_total_rec = t[-1]
    print(t_total_rec)

    qtde_frames = t_total_rec/tI

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
    index_image = 0
    for f in totalImages:

        f = f.astype(np.uint8)
        imagem = copy.deepcopy(np.dstack([f,f,f]))
        #imagem,r = su.vickers_watershed(np.dstack([f,f,f]))
        if(os.path.isfile(pasta.format(index_image))):
            path = pasta.format(index_image)
            f = open(path, "r")
            info = f.readline()
            aux_rect = info.split(" ")
            rect_ = aux_rect[1:len(aux_rect)]
            rect = [int(float(item)) for item in rect_]
            print(rect)
            ax.add_patch( Rectangle((rect[0],rect[1]),
                        rect[2], rect[3],
                        fc ='none', 
                        ec ='g',
                        lw = 10) )
        
        index_image = index_image + 1

        if handle is None:
            handle = plt.imshow(imagem)
        else:
            handle.set_data(imagem)
        
        plt.pause(tI/1000000)
        plt.draw()
        ax.patches = []




if __name__ == "__main__":
	main()
