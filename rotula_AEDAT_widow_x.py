import numpy as np
import copy
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from segmentationUtils import  segmentationUtils as su

def main():

    #Path to .aedat file
    path = 'WidowX_controll_dataset/Sem fundo/Cup.aedat'
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)

    #time window of the frame (merging events)
    tI=50000 #50 ms
    t_total_rec = t[-1]
    print(t_total_rec)

    qtde_frames = t_total_rec/tI

    totalImages = []
    #get the t,p,x and y vectors and return a vector of frames agrouped in time intervals of tI
    totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)

    #config for plotting the frames
    fig,axarr = plt.subplots(1)
    handle = None
    imageVector = []


    for f in totalImages:

        f = f.astype(np.uint8)
        #imagem = copy.deepcopy(np.dstack([f,f,f]))
        imagem,r = su.vickers_watershed(np.dstack([f,f,f]))

        if handle is None:
            handle = plt.imshow(imagem)
        else:
            handle.set_data(imagem)

        plt.pause(tI/1000000)
        plt.draw()




if __name__ == "__main__":
	main()
