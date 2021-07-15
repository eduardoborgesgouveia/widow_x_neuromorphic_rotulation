import copy
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#from iou import iou
#from skimage.morphology import erosion, dilation, opening, closing, white_tophat,square,convex_hull_image
#from filterUtils import filterUtils
#from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils

class segmentationUtils:

    '''
    parameters:
        imagem - desired image
        minimumSizeBox - threshold for the size of the bounding box (in percentage)
        options - optional parameter who is a string with the desired options.
            avaiable options:
                '--neuromorphic' - is the declaration of neuromorphic image or else is a RGB image
    '''
    def watershed(imagem,options=None,detection =[], lastRoi = [],minimumSizeBox = 2,smallBBFilter = True,centroidDistanceFilter=False,mergeOverlapingDetectionsFilter=True,flagCloserToCenter = True):

        opt = []
        imagem = segmentationUtils.preWatershed(imagem)
        flagNeuromorphic = False
        occurrences = 0
        global imageDimensions
        #img = imagem.astype(np.uint8)
        imageDimensions = imagem.shape
        #imagem = filterUtils.median(imagem.astype(np.uint8),15)
        if options != None:
            options = "".join(options.split())
            opt = options.split('--')

        if opt.__contains__('neuromorphic'):
            flagNeuromorphic = True
            img = imagem.astype(np.uint8)
            #img = filterUtils.median(img)
            #img = cv.medianBlur(img,3)
            img[img == 255] = 0
            #img = filterUtils.avg(img)

            if len(img.shape) == 3:
                img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        else:
            img = cv.cvtColor(imagem,cv.COLOR_RGB2GRAY)
            equ = cv.equalizeHist(img)
            equ[equ<=230] = 0
            #cv.imwrite('res.png',res)
            #img = res
            #img = imagem


        ret, thresh = cv.threshold(equ,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
        # sure background area

        sure_bg = cv.dilate(opening,kernel,iterations=3)
        sure_bg = filterUtils.median(sure_bg.astype(np.uint8),27)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.09*dist_transform.max(),255,0)
        sure_fg = filterUtils.median(sure_fg.astype(np.uint8),27)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)

        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv.watershed(imagem,markers)
        #imagem[markers == -1] = [0,0,255]

        detections = segmentationUtils.makeRectDetection(markers,minimumSizeBox,smallBBFilter,centroidDistanceFilter,mergeOverlapingDetectionsFilter)
        #detections = segmentationUtils.getOnlyCloseToCenter(flagCloserToCenter,detections)
        imagem = segmentationUtils.drawRect(imagem,detections)
        #detections = segmentationUtils.getCoordinatesFromPoints(detections)
        return imagem, markers, detections#,sure_fg,sure_bg,unknown

    def vickers_watershed(img):
        #image = cv.resize(img,None,fx=0.2,fy=0.2,interpolation=cv.INTER_AREA)
        image = img
        shifted = cv.pyrMeanShiftFiltering(image, 21, 51)

        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255,
            cv.THRESH_BINARY | cv.THRESH_OTSU)[1]



        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20,
            labels=thresh)
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        contour = []
        circles = []
        max_r = 0.0
        index_max_r = 0
        i = 1
        r = 0
        # loop over the unique labels returned by the Watershed
        # algorithm
        if len(np.unique(labels)) < 300:
            for label in np.unique(labels):
                # if the label is zero, we are examining the 'background'
                # so simply ignore it
                if label == 0:
                    continue
                # otherwise, allocate memory for the label region and draw
                # it on the mask
                mask = np.zeros(gray.shape, dtype="uint8")
                mask[labels == label] = 255
                # detect contours in the mask and grab the largest one
                cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv.contourArea)
                ((x, y), r) = cv.minEnclosingCircle(c)
                #print(r)
                if r >= max_r:
                    max_r = r
                    index_max_r = i
                contour.append(c)
                circles.append(((x, y), r))
                i = i+1
            #print(index_max_r)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == np.unique(labels)[index_max_r]] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv.contourArea)
            ((x, y), r) = cv.minEnclosingCircle(c)
            # draw a circle enclosing the object
            x_r,y_r,w_r,h_r = cv.boundingRect(c)
            #((x, y), r) = cv.minEnclosingCircle(contour[i])
            #((x, y), r) = circles[i]
            cv.rectangle(image,(x_r,y_r),(x_r+w_r,y_r+h_r),(0,255,0),2)
            #cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            #cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
            #    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return image, r

    def preWatershed(img):
        shifted = cv.pyrMeanShiftFiltering(img, 21, 51)
        gray = cv.cvtColor(shifted,cv.COLOR_RGB2GRAY)
        equ = cv.equalizeHist(gray)

        equ[equ<=230] = 0


        res = np.hstack((gray,equ)) #stacking images side-by-side
        cv.imwrite('res.png',res)
        ret, thresh = cv.threshold(equ,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.09*dist_transform.max(),255,0)
        sure_fg = filterUtils.median(sure_fg.astype(np.uint8),27)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv.watershed(img,markers)
        img[markers == -1] = [153,204,50]
        return img


    def getOnlyCloseToCenter(flagCloserToCenter, detections):
        retorno = []
        if(flagCloserToCenter):
            for j in range(len(detections)):
                if (detections[j][7] == 'closerToCenter'):
                    retorno.append(detections[j])
        else:
            retorno = copy.deepcopy(detections)
        return retorno

    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def compare_images(imageA, imageB):
        # compute the mean squared error and structural similarity
        # index for the images
        m = segmentationUtils.mse(imageA, imageB)
        s = measure.compare_ssim(imageA, imageB)
        return m,s
    '''
    this method was make in order to receive a mask from multiple detection using the watershed method
    and make a rectangular bounding box ao redor of the detections.
    '''
    def makeRectDetection(mask,minimumSizeBox=2,smallBBFilter=True,centroidDistanceFilter=True,mergeOverlapingDetectionsFilter = True, last_detection = []):
        #make sure that the edges of the image is not being marked
        #print(mask)
        #mask[mask == 255] = 1
        mask[0,:] = 1
        mask[:,0] = 1
        mask[mask.shape[0]-1,:] = 1
        mask[:, mask.shape[1]-1] = 1
        #blank = np.ones_like(mask)
        #blank[markers == -1] = [153,204,50]
        #gray = cv.cvtColor(blank,cv.COLOR_RGB2GRAY)
        #cv.imshow('output',gray)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        unique = np.unique(mask)
        unique = unique[unique != 1]
        unique = unique[unique != 2]
        print(unique)

        objects = []
        if len(unique) < 40:
            for i in range(len(unique)):
                positions = np.where(mask == unique[i])
                x = min(positions[0])
                y = min(positions[1])
                lastX = max(positions[0])
                lastY = max(positions[1])
                width = lastX - x
                height = lastY - y
                #if the area of the detection is bigger then 20% of the image size (128 * 128 = 16384)
                #so if the bb area is larger then 0.2*16384 the bb need to be keep. Otherwise I ignore then
                if ((smallBBFilter) and (width * height)>((minimumSizeBox/100.0)*(mask.shape[0]*mask.shape[1]))):
                    objects.append([x, y, width, height])
                elif (not smallBBFilter):
                    objects.append([x, y, width, height])


        # print(len(objects))

        objects = segmentationUtils.getCentroid(objects)
        objects = segmentationUtils.getPointsFromCoordinates(objects)
        objects = segmentationUtils.filterDetections(objects,centroidDistanceFilter,mergeOverlapingDetectionsFilter)
        if len(last_detection) > 0:
            objects = segmentationUtils.closerToReference(objects, last_detection)
        else:
            objects = segmentationUtils.closerToReference(objects)

        return objects

    '''
        INPUT -> coordinates =
                     [x, y, width, height, centroidx, centroidy, distanceToCenter]
        OUTPUT -> coordinates =
                     [x, y, width, height, centroidx, centroidy, distanceToCenter, infoAboutCloserToCenter]
    '''
    def closerToReference(coordinates,reference = []):
        if len(reference)==0:
            reference = imageDimensions
        else:
            reference = reference[0]

        if(len(coordinates) > 0):
            minOfEachColumn = np.amin(coordinates,axis=0)
            minOfEachColumnCoord = np.where(coordinates == np.amin(coordinates,axis=0))
            # print('coordenadas dos valores mínimos de cada coluna do array de coordenadas: ',minOfEachColumn)
            # print('array de coordenadas: ',coordinates)
            limitOfDistance = ((math.sqrt(((reference[0]-reference[0]/2)**2)+((reference[1]-reference[1]/2)**2)))*0.2)
        for i in range(len(coordinates)):
            if coordinates[i][6] == minOfEachColumn[6]:
                coordinates[i].append('closerToCenter')
            else:
                coordinates[i].append('notCloserToCenter')

        return coordinates





    '''
        INPUT ->
            coordinates =
                 [x, y, width, height]
        OUTPUT ->
            coordinates =
                [x, y, width, height, centroidx, centroidy, distanceToCenter]
    '''
    def getCentroid(coordinates):
        for i in range(len(coordinates)):
            distanceToCenter = 0
            Cxa = coordinates[i][0] + coordinates[i][2]/2
            Cya = coordinates[i][1] + coordinates[i][3]/2
            coordinates[i].append(Cxa)
            coordinates[i].append(Cya)
            if(imageDimensions and imageDimensions[0] != 0 and imageDimensions[1] != 0):
                distanceToCenter = math.sqrt(((coordinates[i][4]-imageDimensions[0]/2)**2)+((coordinates[i][5]-imageDimensions[1]/2)**2))
            coordinates[i].append(distanceToCenter)
        return coordinates

    def filterDetections(detections,centroidDistanceFilter=True,mergeOverlapingDetectionsFilter = True):
        flag = True
        retorno = detections[:]
        while(flag):
            flag, pos = segmentationUtils.checkIntersec(retorno,centroidDistanceFilter,mergeOverlapingDetectionsFilter)
            retorno = segmentationUtils.mergeDetections(retorno,pos)
        return retorno

    def checkIntersec(coordinates,centroidDistanceFilter=True,mergeOverlapingDetectionsFilter = True):
        count = len(coordinates)
        print(count)
        register = 0
        maxDistance = math.sqrt(imageDimensions[0]**2 + imageDimensions[1]**2)
        if (centroidDistanceFilter or mergeOverlapingDetectionsFilter):
            for i in range(len(coordinates)):
                for j in range(len(coordinates)):
                    if j > i:
                        area = iou.bb_intersection_over_union(coordinates[i],coordinates[j])
                        distance = segmentationUtils.getDistance(coordinates[i],coordinates[j])
                        if (area > 0.0 and area != 1.0 and mergeOverlapingDetectionsFilter) or (centroidDistanceFilter and distance < 0.002*maxDistance):
                            return True, [i,j]

        return False, None

    def getDistance(boxA, boxB):
        distance = math.sqrt(((boxA[4]-boxB[4])**2)+((boxA[5]-boxB[5])**2))
        # print('distance: '+ str(distance))
        return distance

    def drawRect(img, detections,lineWidth=None):
        bbColor = 8
        detections = segmentationUtils.getCoordinatesFromPoints(detections)
        if lineWidth == None:
            lineWidth = round(0.001*img.shape[0])
        if len(img.shape) == 3:
            bbColor = [255,0,0]
        for i in range(len(detections)):
            img[detections[i][0],detections[i][1]] = bbColor
            img[detections[i][0]:detections[i][0]+lineWidth,detections[i][1]:(detections[i][1]+detections[i][3])] = bbColor
            img[detections[i][0]:(detections[i][0]+detections[i][2]),detections[i][1]:detections[i][1]+lineWidth] = bbColor
            img[(detections[i][0]+detections[i][2]):(detections[i][0]+detections[i][2])+lineWidth,detections[i][1]:(detections[i][1]+detections[i][3])] = bbColor
            img[detections[i][0]:(detections[i][0]+detections[i][2]),(detections[i][1]+detections[i][3]):(detections[i][1]+detections[i][3])+lineWidth] = bbColor
        return img

    def drawTarget(img,lineWidth=2):

        retorno = cv.drawMarker(img, (int(img.shape[1]/2),int(img.shape[0]/2)),(0,0,255),markerType=cv.MARKER_CROSS,markerSize=4, thickness=lineWidth, line_type=cv.LINE_AA)

        return retorno

    def drawMag(img):
        font = cv.FONT_HERSHEY_SIMPLEX
        retorno = cv.putText(img,text='10x',org=(int(img.shape[1]*0.03),int(img.shape[0]*0.08)),fontFace=font,thickness=4,lineType=cv.LINE_AA,color=(255,255,255),fontScale=5)
        return retorno

    def drawRef(img,dsFactor):
        #70 µm é equivalente a 388,88 pixels
        #com o dfFactor de 0.2 fica 77,77
        pixel = 364.9548
        threshold = 0.028*img.shape[1]

        font = cv.FONT_HERSHEY_SIMPLEX
        retorno = cv.putText(img,text='70 um',org=(int(img.shape[1]*0.86),int(img.shape[0]*0.87)),fontFace=font,thickness=4,lineType=cv.LINE_AA,color=(255,255,255),fontScale=5)
        retorno = cv.line(retorno,pt2=(int(img.shape[1] - pixel*dsFactor - threshold),int(img.shape[0]*0.9)),pt1= (int(img.shape[1] - threshold),int(img.shape[0]*0.9)),color=(255,255,255),thickness=3)

        retorno = cv.line(retorno,pt1=(int(img.shape[1] - pixel*dsFactor - threshold),int(img.shape[0]*0.88)),pt2= (int(img.shape[1] - pixel*dsFactor - threshold),int(img.shape[0]*0.92)),color=(255,255,255),thickness=3)
        retorno = cv.line(retorno,pt1=(int(img.shape[1] - threshold),int(img.shape[0]*0.88)),pt2= (int(img.shape[1] - threshold),int(img.shape[0]*0.92)),color=(255,255,255),thickness=3)
        return retorno


    '''
        If one or more rectangular detections has a IOU the bounding boxes are merged and
    became just one
    '''
    def mergeDetections(detections,pos):
        retorno = detections
        if (pos != None):
            coordinates = copy.deepcopy(detections)
            retorno = []
            X1 = max(coordinates[pos[0]][0],coordinates[pos[0]][2],coordinates[pos[1]][0],coordinates[pos[1]][2])
            X2 = min(coordinates[pos[0]][0],coordinates[pos[0]][2],coordinates[pos[1]][0],coordinates[pos[1]][2])
            Y1 = max(coordinates[pos[0]][1],coordinates[pos[0]][3],coordinates[pos[1]][1],coordinates[pos[1]][3])
            Y2 = min(coordinates[pos[0]][1],coordinates[pos[0]][3],coordinates[pos[1]][1],coordinates[pos[1]][3])
            width = X1 - X2
            height = Y1 - Y2

            coordWithCentroid = segmentationUtils.getCentroid([[X2, Y2, width, height]])

            coordinates.remove(detections[pos[0]])
            coordinates.remove(detections[pos[1]])

            retorno = coordinates
            retorno.append([X2, Y2, X1, Y1, coordWithCentroid[0][4],coordWithCentroid[0][5],coordWithCentroid[0][6]])
        return retorno

    def getPointsFromCoordinates(detections):
        objects = []
        for i in range(len(detections)):
            x1 = detections[i][0]
            y1 = detections[i][1]
            x2 = detections[i][0] + detections[i][2]
            y2 = detections[i][1] + detections[i][3]
            objects.append([x1, y1, x2, y2, detections[i][4], detections[i][5],detections[i][6]])
        return objects
    def getCoordinatesFromPoints(detections):
        objects = []
        for i in range(len(detections)):
            if detections[i][2] - detections[i][0] > 1 and detections[i][3] - detections[i][1] > 1:
                x1 = detections[i][0]
                y1 = detections[i][1]
                width = detections[i][2] - x1
                lenght = detections[i][3] - y1
                objects.append([x1, y1, width, lenght, detections[i][4], detections[i][5],detections[i][6],detections[i][7]])
        return objects

    #this function get the original image and extract the ROI
    def getROI(d, image):
        _ = 0
        if len(d) > 0:
            dim = (128, 128)
            biggerEdge = max(d[0][2],d[0][3])
            smallerEdge = min(d[0][2],d[0][3])
            delta = int((biggerEdge - smallerEdge)/2)
            if(d[0].index(smallerEdge) == 2):
                d[0][0] = 0 if (d[0][0]-delta) < 0 else d[0][0]-delta
            else:
                d[0][1] = 0 if (d[0][1]-delta) < 0 else d[0][1]-delta
            d[0][d[0].index(smallerEdge)] = biggerEdge
            crop_img = image[(d[0][0] + 1) : d[0][0] + d[0][2], (d[0][1] + 1) : d[0][1] + d[0][3]]
            interp_img = cv.resize(crop_img, dim, interpolation = cv.INTER_NEAREST)
            # crop_img = crop_img.reshape(1, 128, 128, 1)
            return crop_img, interp_img
        else:
            return image, image

    '''
    this method run a demo for watershed segmentation technique.
    this will plot 4 images:
        - 1 standard image (original)
        - 1 standard image (watershed segmentation)
        - 1 neuromorphic image (original | probabily 100 ms event agroupation)
        - 1 neuromorphic image (watershed segmentation + filter of avg and median)
    '''
    def watershed_demo():
        font = {'family': 'serif',
                'color':  'red',
                'weight': 'normal',
                'size': 8,
        }
        off_set_text = 3

        dim = (128,128)

        neuromorphicImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Detection/assets/testes/Mouse_22.png')
        nImage = copy.deepcopy(neuromorphicImage)
        standardImage = cv.imread('/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/Standard files/HV5_FOTO 1.jpg')

        watershedStandardImage, standardMask,standardDetection, standardOpening, standardSure_fg, standardSure_bg,standardMarkers = segmentationUtils.watershed(standardImage)

        neuromorphicImage = filterUtils.avg(neuromorphicImage)
        neuromorphicImage = filterUtils.median(neuromorphicImage)

        watershedNeuromorphicImage, neuromorphicMask,neuromorphicDetection, neurOpening, neuroSure_fg, neuroSure_bg,neuroMarkers = segmentationUtils.watershed(neuromorphicImage,'--neuromorphic')


        f, axarr = plt.subplots(2,3)
        axarr[0,0].set_title('neuromorphic image [original]')
        axarr[0,0].imshow(nImage)

        axarr[0,1].set_title('neuromorphic - mask')
        axarr[0,1].imshow(neuromorphicMask)

        crop_neuromorphic = nImage[neuromorphicDetection[0][0]+1:neuromorphicDetection[0][0]+neuromorphicDetection[0][2] , neuromorphicDetection[0][1]+1:neuromorphicDetection[0][1]+neuromorphicDetection[0][3]]

        axarr[0,2].set_title('croped bounding box')
        axarr[0,2].imshow(crop_neuromorphic)


        axarr[1,0].set_title('standard image [original]')
        axarr[1,0].imshow(standardImage)

        axarr[1,1].set_title('standard - mask')
        axarr[1,1].imshow(standardMask)

        crop_standard = standardImage[standardDetection[0][0]+(round(0.01*standardImage.shape[0])):standardDetection[0][0]+standardDetection[0][2] , standardDetection[0][1]+(round(0.01*standardImage.shape[0])):standardDetection[0][1]+standardDetection[0][3]]

        axarr[1,2].set_title('croped bounding box')
        axarr[1,2].imshow(crop_standard)


        model = classifierTools.openModel('model/model.json',
							              'model/model.h5')


        crop_img = cv.resize(crop_neuromorphic, dim, interpolation = cv.INTER_AREA)
        crop_img = cv.cvtColor(crop_img,cv.COLOR_RGB2GRAY)
        crop_img = crop_img.reshape(1, 128, 128, 1)
        resp, objectSet = classifierTools.predictObject(crop_img, model)
        predict = objectSet[resp][1]
        axarr[0,0].text(neuromorphicDetection[0][1], neuromorphicDetection[0][0]-off_set_text, predict, fontdict = font)

        plt.show()

if __name__ == "__main__":
	segmentationUtils.watershed_demo()
