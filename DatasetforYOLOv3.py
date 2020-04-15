import os
import cv2
import numpy as np
import math
import random
import AffineTransformation as AT

# important parameters
PATH = os.getcwd()
plates_path = os.getcwd()+"\\licensePlateDatasetGenerator\\data"
# scenes_path = os.getcwd()+"\\ResizedGrayUrbanScene_Data"============================
scenes_path = os.getcwd()+"\\UrbanScene_Data"
# database_path = os.getcwd()+"\\Database_FrenchLicensePlates"
database_path = os.getcwd()+"\\data"
if not os.path.exists(database_path):
    os.mkdir(database_path)
ratio = 15 #int
size_factor = 4 #float

os.chdir(plates_path)
plates = os.listdir()
os.chdir(scenes_path)
scenes = os.listdir()


def find_labels_for_yolo(imgshape, ntlc, nblc, ntrc, nbrc, new_center,category_number="0"):
    dw = 1./imgshape[1]
    dh = 1./imgshape[0]
    xmax, ymax = np.max([ntlc[0], nblc[0], ntrc[0], nbrc[0]]), np.max([ntlc[1], nblc[1], ntrc[1], nbrc[1]])
    xmin, ymin = np.min([ntlc[0], nblc[0], ntrc[0], nbrc[0]]), np.min([ntlc[1], nblc[1], ntrc[1], nbrc[1]])
    w = (xmax-xmin)*dw
    cx = new_center[0]*dw
    h = (ymax-ymin)*dh  
    cy = new_center[1]*dh
    
    return category_number+" "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h)
    
def find_labels_for_yolov3(count, imgpath, imgshape, ntlc, nblc, ntrc, nbrc, new_center,category_number="0"):
    xmax, ymax = np.max([ntlc[0], nblc[0], ntrc[0], nbrc[0]]), np.max([ntlc[1], nblc[1], ntrc[1], nbrc[1]])
    xmin, ymin = np.min([ntlc[0], nblc[0], ntrc[0], nbrc[0]]), np.min([ntlc[1], nblc[1], ntrc[1], nbrc[1]])
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    if xmin < 0:
        xmin=int(0)
    if ymin < 0:
        ymin=int(0)
    if xmax > imgshape[1]:
        xmax=imgshape[1]
    if ymax > imgshape[0]:
        ymax=imgshape[0]
        
    
    return " "+str(imgpath)+" "+str(imgshape[1])+" "+str(imgshape[0])+" "+category_number+" "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)

def new_corners(lpshape,sceneshape,M):
    ''' find corners in the new image'''
    lp1, lp2 = lpshape[0], lpshape[1]
    center = [lp2/2,lp1/2]
    tlc = [0,0]
    blc = [lp2,0]
    trc = [0,lp1]
    brc = [lp2,lp1]

    #new center of the plate
    new_center = [M[0,0]*center[0] + M[0,1]*center[1] + M[0,2],M[1,0]*center[0] + M[1,1]*center[1] + M[1,2]] 
    #new top left corner of the plate
    ntlc       = [M[0,0]*tlc[0] + M[0,1]*tlc[1] + M[0,2],M[1,0]*tlc[0] + M[1,1]*tlc[1] + M[1,2]] 
    #new bottom left corner of the plate
    nblc       = [M[0,0]*blc[0] + M[0,1]*blc[1] + M[0,2],M[1,0]*blc[0] + M[1,1]*blc[1] + M[1,2]] 
    #new top right corner of the plate
    ntrc       = [M[0,0]*trc[0] + M[0,1]*trc[1] + M[0,2],M[1,0]*trc[0] + M[1,1]*trc[1] + M[1,2]] 
    #new bottom right corner of the plate
    nbrc       = [M[0,0]*brc[0] + M[0,1]*brc[1] + M[0,2],M[1,0]*brc[0] + M[1,1]*brc[1] + M[1,2]] 

    return ntlc, nblc, ntrc, nbrc, new_center

def train_test_val(labels, path=database_path):
    count = 1
    train = open(path+'\\train.txt','w')
    test = open(path+'\\test.txt','w')
    val = open(path+'\\val.txt','w')
    count_train = 0
    count_test = 0
    count_val = 0
    for label in labels:
        if count<=8:
            train.write(str(count_train)+label+"\n")
            count_train+=1
        if count==9:
            test.write(str(count_test)+label+"\n")
            count_test+=1
        if count==10:
            val.write(str(count_val)+label+"\n")
            count_val+=1
            count=0
        count+=1

    train.close()
    test.close()
    val.close()



#  Each plate will be assigned to "ratio" random scenes
labels = []
count_yolo = 0 
for plate in plates:
    lp = cv2.imread(plates_path+"\\"+plate)
    lp_shape = np.ones(lp.shape)*255
    count = 0
    while count < ratio:
        count += 1
        scene_path = random.choice(scenes)
        scene = cv2.imread(scenes_path+"\\"+scene_path)
        scene = cv2.resize(scene,(int(128*size_factor),int(64*size_factor)))
        M, out_of_bounds, scale, center_to = AT.make_affine_transform(from_shape=lp.shape[:2], to_shape=scene.shape[:2], 
                                                    min_scale=0.2, max_scale=0.3,
                                                    scale_variation=1.5,
                                                    rotation_variation=1.2,
                                                    translation_variation=1.2
                                                    )
                                                    
        # finding labels
        ntlc, nblc, ntrc, nbrc, new_center = new_corners(lp.shape, scene.shape, M)
        label = find_labels_for_yolov3(count=count_yolo, imgpath="./data/plates/"+str(count_yolo)+".jpg", imgshape=scene.shape,
                                       ntlc=ntlc, nblc=nblc, ntrc=ntrc, nbrc=nbrc, 
                                       new_center=new_center,category_number="0")
        
       
        # find if the plate is in the image or not
        # checking if the plate is inside the rectangle defined by image shape - 1/2*plateshape
        lp1, lp2 = lp.shape[0]/2.3*scale, lp.shape[1]/4*scale
        sc1, sc2 = scene.shape[0], scene.shape[1]
        if (   (new_center[1]>(lp1))
            and(new_center[1]<(sc1-lp1))
            and(new_center[0]>(lp2))
            and(new_center[0]<(sc2-lp2))):
            out_of_bounds = False
        else:
            out_of_bounds = True
            continue
        if out_of_bounds:
            count-=1
        
        # if not out_of_bounds:
        #     print(label)
    

        # image treatment
            # rotating plate and creating mask from its shape
        rotationed_lp       = cv2.warpAffine(lp, M,(scene.shape[1], scene.shape[0]))
        rotationed_lp_shape = cv2.warpAffine(lp_shape, M,(scene.shape[1], scene.shape[0]))

        # Operation do combine the photos: apply mask to scene add the plate and convert to gray
        anti_shape = ((255 - rotationed_lp_shape)/255)
        combined_image = (scene*anti_shape).astype(np.uint8)+rotationed_lp
        allgray_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)

        #add noise
        noise = np.zeros(allgray_image.shape)
        cv2.randn(noise,mean=0,stddev=25)
        final_image = np.clip(allgray_image+noise, 0, 255).astype(np.uint8)

        if out_of_bounds:
            final_image_path = database_path+"\\"+plate[:-7]+"_0_"+scene_path[:-4]+".jpg"
        else:
            final_image_path = database_path+"\\"+plate[:-7]+"_1_"+scene_path[:-4]+".jpg"

        count_yolo+=1
        labels.append(label)
        cv2.imwrite(database_path+"\\"+str(count_yolo)+".jpg", final_image)

        # prints for debugs reasons
        # cv2.imshow('rotationed_lp',rotationed_lp)
        # cv2.imshow('lp_shape'+str(out_of_bounds),rotationed_lp_shape)
        # cv2.imshow('scene',scene)
        # cv2.imshow('anti_shape',anti_shape)
        # cv2.imshow('final_image'+str(1-int(out_of_bounds)),final_image)
        # cv2.waitKey(1000)
        print(label)

train_test_val(labels=labels, path=database_path)