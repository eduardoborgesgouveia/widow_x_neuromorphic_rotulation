import numpy as np
import copy
import matplotlib.pyplot as plt
from openAEDAT_Gustavo import aedatUtils
from PIL import Image 
import PIL 
import png


def main():

    #Path to .aedat file
    #path = 'Lapiseira.aedat'
    path = 'WidowX_controll_dataset/Com fundo/Cup.aedat'
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)

    #time window of the frame (merging events)
    tI=100000 #50 ms


    totalImages = []
    #get the t,p,x and y vectors and return a vector of frames agrouped in time intervals of tI
    totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

    #config for plotting the frames
    fig,axarr = plt.subplots(1)
    handle = None
    imageVector = []

    i = 0
    for f in totalImages:

        f = f.astype(np.uint8)
        imagem = copy.deepcopy(f)
        # imagem[imagem == 0] = 255
        # imagem[imagem == 127] = 0

        if handle is None:
            handle = plt.imshow(np.dstack([imagem,imagem,imagem]))
        else:
            handle.set_data(np.dstack([imagem,imagem,imagem]))

        #png.from_array([f,f,f], 'L').save(str(i) + ".png")
        i = i+1
        plt.pause(tI/1000000)
        plt.draw()




if __name__ == "__main__":
	main()
