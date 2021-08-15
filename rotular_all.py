import numpy as np
import copy
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
from openAEDAT_Leandra import aedatUtils as aedatUtilsLeandra
from openAEDAT_Gustavo import aedatUtils as aedatUtilsGustavo
from matplotlib.patches import Rectangle
import yaml
#from segmentationUtils import  segmentationUtils as su

def main():

    a_yaml_file = open("control_dataset.yaml")

    param = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    tempoFrame = param['general']['integration_time']
    for index,classe in param['class_name'].items():
        for tipoSuperficie in param['time_surface']:
            imagens = getImages(param['general'],param['class_path_aedat'][index],tipoSuperficie)
            tempoGravacao = tempoFrame*len(imagens)
            bb = getBoudingBox(param['class_parameters'][index],tempoFrame,tempoGravacao,param['general']['time_limit'])
            saveInfos(param['general'],bb,imagens,classe,index,tipoSuperficie)







def getImages(parametrosGerais, path, tipoSuperficie):
    tI = parametrosGerais['integration_time']
    #loading the values of the file
    #t is the time vector
    # x and y is the coordinates of the events
    # p is the polarity of the event (eg.: 1 or -1)
    t, x, y, p = aedatUtils.loadaerdat(path)
    totalImages = []
    if(tipoSuperficie == "SITS"):
        totalImages = aedatUtilsLeandra.getFramesTimeBased(t,p,x,y,tI)
    elif(tipoSuperficie == "Integral"):
        #get the t,p,x and y vectors and return a vector of frames agrouped in time intervals of tI
        totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI)
    elif tipoSuperficie == "TORE":
        totalImages = aedatUtilsGustavo.getFramesTimeBased(t,p,x,y,tI)
    return totalImages

def saveInfos(parametros_gerais, boundingBox, totalImages,nomeClasse,indexClasse, tipoSuperficie):
    '''
    c_x = (rect(1,1) + (rect(1,3)/2))/128;
    c_y = (rect(1,2) + (rect(1,4)/2))/128;
    label = strcat(int2str(indexClasse) + " ", sprintf('%.6f',(c_x)) + " ", sprintf('%.6f',(c_y)) + " ", sprintf('%.6f',(rect(1,3)/128)) + " ", sprintf('%.6f',(rect(1,4)/128)));
    '''
    percentTrain = parametros_gerais['percentage_train']
    path = parametros_gerais['path_to_save']
    i = 0
    indexRotulo = 1
    indexImage = 0
    for bb in boundingBox:
        f = totalImages[indexImage]
        f = f.astype(np.uint8)
        imagem = copy.deepcopy(np.dstack([f,f,f]))
        indexRotulo = indexRotulo + 1
        aux_rect = bb
        c_x = (aux_rect[0] + (aux_rect[2]/2))/128
        c_y = (aux_rect[1] + (aux_rect[3]/2))/128
        stringBase = "{0} {1} {2} {3} {4}"
        stringToSave = stringBase.format(indexClasse,c_x,c_y,(aux_rect[2]/128), (aux_rect[3]/128))
        if(i<=percentTrain*len(boundingBox)):
            tipo = "train"
        else:
            tipo = "test"
        if(not os.path.isdir(path + tipoSuperficie + "/" + tipo + "/labels/")):
            os.makedirs(path + tipoSuperficie + "/" + tipo + "/labels/")
        if(not os.path.isdir(path + tipoSuperficie + "/" + tipo + "/images/")):
            os.makedirs(path + tipoSuperficie + "/" + tipo + "/images/")
        cv.imwrite(path + tipoSuperficie + "/" + tipo + "/images/" + nomeClasse + "_"  + str(i) + ".png",imagem)
        f= open(path + tipoSuperficie + "/" + tipo + "/labels/" + nomeClasse + "_"  + str(i) + ".txt","w")
        f.write(stringToSave)
        f.close()
        i = i + 1
        indexImage = indexImage + 1

def getBoudingBox(parametros,tempoFrame,tempoGravacao,timeLimit):
    tI = tempoFrame
    fator_divisor_tempo = 200000/tempoFrame
    fator_conversao_tempo = 1000000
    qtde_frames = int(tempoGravacao/tempoFrame)
    width = parametros['width']
    height = parametros['height']
    x = parametros['x']
    y = parametros['y']
    z = parametros['z']
    x_aux = x
    y_aux = y
    z_aux = z
    width_aux = width
    height_aux = height
    temposEtapas = parametros['temposEtapas']
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
        if(tempo_atual <= timeLimit):
            if(tempo_atual>=temposEtapas[countEtapa-1] and tempo_atual < temposEtapas[countEtapa]):
                aux_dist_x = currentPos[0] - lastPos[0]
                aux_dist_y = currentPos[1] - lastPos[1]
                aux_dist_z = currentPos[2] - lastPos[2]
                x_aux = x_aux + aux_dist_x/(temp*fator_divisor_tempo) - (aux_dist_z/(temp*fator_divisor_tempo))/2
                y_aux = y_aux + aux_dist_y/(temp*fator_divisor_tempo) - (aux_dist_z/(temp*fator_divisor_tempo))/2
                width_aux = width_aux + (aux_dist_z/(temp*fator_divisor_tempo))
                height_aux = height_aux + (aux_dist_z/(temp*fator_divisor_tempo))
            else:
                print(tempo_atual)
                currentPos = distanciaEtapas[countEtapa]
                lastPos = distanciaEtapas[countEtapa - 1]
                temp = temposEtapas[countEtapa] - temposEtapas[countEtapa - 1]
                countEtapa = countEtapa + 1

            vetorPosicoes.append((x_aux,y_aux,width_aux,height_aux))
    return vetorPosicoes


if __name__ == "__main__":
	main()
