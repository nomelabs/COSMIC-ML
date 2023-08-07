from timeit import default_timer as timer
start = timer()

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import random as rd
import string
import cv2
import os
from imantics import Polygons,BBox,Mask,Annotation
from shapely.geometry import Polygon
from pycocotools import mask as mk  
import json
import scipy.ndimage
import glob
import re
import pandas as pd
import random
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

#change working directory
os.chdir(r'C:\Users\iluvs\OneDrive\Desktop\Cosmic ML\COSMIC-ML')
print(os.getcwd())
export_loc = 'Synth Data v8.0/a' 
Artifical_img_size = 1024
#### Organize all backgrounds ####
BackgroundsDF = pd.DataFrame(columns= ['Name','Scale','Image','Artifical Scale'])
background_fldr_name = r"Clean Background Imgs\Usable Imgs"

background_rez_fldrs = os.listdir(background_fldr_name)
for folder in background_rez_fldrs:
    temp_backs_names = glob.glob(os.path.join(background_fldr_name,folder,'*'))
    for temp_back in temp_backs_names:
        temp_scale = folder
        temp_img = cv2.imread(temp_back , 0)
        temp_art_scale = (len(temp_img)/Artifical_img_size) * float(temp_scale)
        #plt.imshow(temp_img,cmap='gray')
        #plt.show()
        tempdf=pd.DataFrame([{'Name':temp_back, 'Scale':temp_scale , 'Image':temp_img , 'Artificial Scale':temp_art_scale}])
        BackgroundsDF = pd.concat([BackgroundsDF,tempdf],ignore_index=True)
SynthImgDF = pd.DataFrame(columns= ['Image Name','Resolution','Scale','Field of view (nm^2)','Total Cavities','Avg Size(nm)'])
    
#input('y')    



#### Setting up Table of COntrsat Functions #######
Table_defoc_paths = glob.glob(r'Mathematica Tables\*')
FunctionsDF = pd.DataFrame(columns= ['defoc','r0','I_min','Contrast Function'])
rhoList = np.linspace(0,1.5,151)
defoc_vals = os.listdir('Mathematica Tables')
defoc_vals = [ int(x) for x in defoc_vals ]
for c in range(len(Table_defoc_paths)):
    Table_paths = glob.glob(os.path.join(Table_defoc_paths[c],'*'))
    temp_defoc = int(Table_defoc_paths[c][19:])
    for d in range(len(Table_paths)):
        temp_table = np.loadtxt(Table_paths[d])
        temp_func = scipy.interpolate.interp1d(rhoList,temp_table,bounds_error=False,fill_value=temp_table[-1])
        temp_r0 = int(re.findall(r'\d+', Table_paths[d])[2])
        temp_Io_min = np.min(temp_table)
        tempdf=pd.DataFrame([{'defoc':temp_defoc,'r0':temp_r0 ,'I_min':temp_Io_min, 'Contrast Function':temp_func}])
        FunctionsDF = pd.concat([FunctionsDF,tempdf],ignore_index=True)
BackgroundsDF = BackgroundsDF.sample(frac=1).reset_index(drop=True) #randomize backgrounds

tz=timer()
print(tz-start)

#### Egg Cav Constants ####
low_AC_ran_min,low_AC_ran_max,low_BD_ran_min,low_BD_ran_max = 80,120, 10,30
med_AC_ran_min,med_AC_ran_max,med_BD_ran_min,med_BD_ran_max = 80,120, 20,50
high_AC_ran_min,high_AC_ran_max,high_BD_ran_min,high_BD_ran_max = 80,120, 30,70
neg_list = [-1,1]
cav_pixels = 70#100
corner = np.sqrt(8)
line = np.linspace(-corner,corner,cav_pixels)
dist_x = np.tile(line,(cav_pixels,1))
dist_y = np.transpose(dist_x)


Master_cav_r0_list = []
Master_ant_pixels_list = np.array([])
Master_cav_bbox_list = []
Master_cav_defoc_list = []

    
total_cavs_req = 200
total_cavs = 0
iteration = 0
img_count = 0
while total_cavs < total_cavs_req:
    for bckgrnd_val in range(len(BackgroundsDF)): #[0,5,10,15,20,25,30]:
        if total_cavs > total_cavs_req:
            break
        wrkng_backgrnd = BackgroundsDF.iloc[bckgrnd_val]
        og_res = float(wrkng_backgrnd['Scale'])
        background = wrkng_backgrnd['Image']
        background_name = wrkng_backgrnd['Name']
        avg_back_size = int((np.shape(background)[0] +np.shape(background)[1])/2)
        scale_factor = avg_back_size / Artifical_img_size
        
        defoc_polarity = -1#random.choices([1,-1],weights=(20,80))[0] # 1 represents and overfocused image and -1 an underfocused image
        print('Defoc Polarity:'+str(defoc_polarity))
        
        background = cv2.resize(background, dsize=(Artifical_img_size, Artifical_img_size), interpolation=cv2.INTER_CUBIC) #REMOVE, THIS IS JUST FOR SPEEDING UP
        copying_background = background
        back_clone = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        back_shape = np.shape(background)
        back_width = back_shape[0]
        back_height = back_shape[1]
        back_mean = np.mean(background)
        percentile = np.percentile(background , 50)
        flat_back = np.uint8(np.full(back_shape,percentile))  #back_mean
        flat_w_cavs = np.full(back_shape,1).astype(float)
        always_flat = np.full(back_shape,1)
        sloppy_clone = np.uint8(np.full(back_shape,percentile))
        back_res = og_res * scale_factor
        back_physical_len = back_res * Artifical_img_size
        back_fov = back_physical_len**2
        bright_p = np.percentile(background , 95)
        dark_p = np.percentile(background , 5)
            
        #plt.hist(background.ravel(),256,[0,256]); plt.show()
            
        cav_r0_min = int(np.round(7 * back_res)) #was 6
        cav_density_percent = rd.randrange(5,15)/100 
        cav_density_req =  np.size(background) * cav_density_percent 
        cav_size_med = rd.randrange(cav_r0_min,3*cav_r0_min,1) #18
        cav_size_1dev = .8 * cav_size_med  
        cav_med_area = (np.pi * cav_size_med**2) / back_res**2
        avg_cav_quant = int(cav_density_req / cav_med_area)
        cav_size_list = np.round(np.random.normal(cav_size_med, cav_size_1dev, size = avg_cav_quant))
        cav_size_list[cav_size_list > 40] = 40
        cav_size_list[cav_size_list < cav_r0_min] = cav_r0_min
        plt.hist(cav_size_list,int(np.max(cav_size_list)) , ((0,int(np.max(cav_size_list)))) )
        plt.show()
        avg_cav_size = np.mean(cav_size_list)
        
        
        perf_cav_defocs = []
        for c in cav_size_list:
            temp_perf_cav_defocs = ( 450 * np.log (c/50)) * rd.choice(np.arange(.75,1.25,.01)) #same for whole image
            perf_cav_defocs.append(temp_perf_cav_defocs)
            
        print(str(cav_density_percent*100)+'% cavities')
        print('resolution:',str(back_res))
        print('cav_r0_min:'+str(cav_r0_min))
        cav_density = 0 
        

        annotations = []
        cum_mask = np.zeros(back_shape,dtype=bool)
        t = 0
        for l in range(len(cav_size_list)):
       # while cav_density < cav_density_req :

            #### Egg Geom Gen #####
            img_full = False
            temp_cav_r0 = cav_size_list[l]
            temp_cav_defoc = np.random.normal(perf_cav_defocs[l] , 50)
            temp_cav_defoc = 50 * round(temp_cav_defoc/50)
            if temp_cav_defoc >-300 :
                temp_cav_defoc = -300
            if temp_cav_defoc <-2300 :
                temp_cav_defoc = -2300

            if temp_cav_r0 <= 5:
                AC_ran_min,AC_ran_max,BD_ran_min,BD_ran_max = low_AC_ran_min,low_AC_ran_max,low_BD_ran_min,low_BD_ran_max
            elif temp_cav_r0 > 5 and temp_cav_r0 <= 20:
                AC_ran_min,AC_ran_max,BD_ran_min,BD_ran_max = med_AC_ran_min,med_AC_ran_max,med_BD_ran_min,med_BD_ran_max
            elif temp_cav_r0 > 20:
                AC_ran_min,AC_ran_max,BD_ran_min,BD_ran_max = high_AC_ran_min,high_AC_ran_max,high_BD_ran_min,high_BD_ran_max
            A,C = rd.randrange(AC_ran_min,AC_ran_max)/100 , rd.randrange(AC_ran_min,AC_ran_max)/100
            B,D = rd.choice(neg_list)*rd.randrange(BD_ran_min,BD_ran_max)/100 , rd.choice(neg_list)*rd.randrange(BD_ran_min,BD_ran_max)/100
            
            egg_shape = np.zeros((cav_pixels,cav_pixels))
            it = np.nditer(egg_shape, flags=['multi_index'])
            for point in it:
                x_dist = dist_x[it.multi_index]
                y_dist = dist_y[it.multi_index]
                total_dist = np.sqrt(x_dist**2 + y_dist**2)
                radius = 1 + A*np.sin(x_dist*B) + C*np.sin(y_dist*D)
                if np.isclose(total_dist,radius,atol=.075):
                    egg_shape[it.multi_index] = 1
                else:
                    continue
            #plt.imshow(egg_shape)
            #plt.title("A:%s B:%s C:%s D:%s" % (str(A),str(B),str(C),str(D)) )
            #plt.show()
         
            egg_resize_len = int(4 * temp_cav_r0/back_res)    
            if egg_resize_len > cav_pixels:
                egg_shape_scale = egg_shape
            elif egg_resize_len <= cav_pixels:
                egg_shape_scale = cv2.resize(egg_shape , dsize = (egg_resize_len,egg_resize_len) , interpolation=cv2.INTER_CUBIC)
                egg_shape_scale = np.where(np.abs(egg_shape_scale) > 0.0 , 1 , 0)  #may need a better way to do this
        
            temp_egg_len = len(egg_shape_scale)
            contours = cv2.findContours(np.uint8(egg_shape_scale), 0, 1)
            poly_points = np.squeeze(contours[0])
            polygon = Polygon(poly_points)
            egg_center_x = int(polygon.centroid.coords.xy[0][0])
            egg_center_y = int(polygon.centroid.coords.xy[1][0])
            roll_x = (int(egg_center_y - temp_egg_len/2))
            roll_y = (int(egg_center_x - temp_egg_len/2))
            if roll_x*roll_y > 0:
                roll_x = -1*roll_x
                roll_y = -1*roll_y
            centered_egg = np.roll(egg_shape_scale , roll_y, axis=0 )
            centered_egg = np.roll(centered_egg , roll_x, axis=1 )
            contours = cv2.findContours(np.uint8(centered_egg), 0, 1)
            poly_points = np.squeeze(contours[0])
            polygon = Polygon(poly_points)
            poly_points= tuple(poly_points) #Tuples are faster to parse?
            egg_center_x = int(polygon.centroid.coords.xy[0][0])
            egg_center_y = int(polygon.centroid.coords.xy[1][0])
            
            egg_full = ndimage.binary_fill_holes(centered_egg).astype(float)
            for p in range(len(poly_points)):
                poly_y = int(poly_points[p][0])
                poly_x = int(poly_points[p][1])
                egg_full[poly_x][poly_y] = 2
                
            rho_map = np.zeros((temp_egg_len,temp_egg_len))
            it = np.nditer(egg_full, flags=['multi_index'])
            for val in it:
                if val == 2: #cavity edge
                    rho = 1  
                else:
                    x = it.multi_index[0]
                    y = it.multi_index[1]
                    dist_cent = np.sqrt((x-egg_center_x)**2 + (y-egg_center_y)**2)
                    
                    edge_distances = []
                    for p in range(int(len(poly_points)/2)):   #divide by 2 and the 2*p is so we skip every other point to save time
                        poly_y = int(poly_points[2*p][0])
                        poly_x = int(poly_points[2*p][1])
                        dist_poly = np.sqrt((x-poly_x)**2 + (y-poly_y)**2)
                        edge_distances.append(dist_poly)
                    closest_edge = np.min(edge_distances)
                    
                    if val == 1: #inside cavity
                        rho = dist_cent / (dist_cent+closest_edge)
                    if val == 0: #outside cavity 
                        rho = dist_cent / (dist_cent-closest_edge)
        
                    
                rho_map[it.multi_index] = rho #17.5seconds

            #### Egg Contrast Gen ####
            
    
            Con_func = FunctionsDF.loc[(FunctionsDF['r0']==temp_cav_r0) & (FunctionsDF['defoc']==temp_cav_defoc)]['Contrast Function'].item()
            I_min = FunctionsDF.loc[(FunctionsDF['r0']==temp_cav_r0) & (FunctionsDF['defoc']==temp_cav_defoc)]['I_min'].item()
            Cav_img = np.zeros((temp_egg_len,temp_egg_len))
            it = np.nditer(Cav_img, flags=['multi_index'])
            for val in it:
                rho = rho_map[it.multi_index]
                Cav_img[it.multi_index] = Con_func(rho)
                
            #plt.imshow(Cav_img,cmap='gray')
            #plt.show()
                
                
                #### Upscale if needed ####
            if egg_resize_len > cav_pixels:
                Cav_img = cv2.resize(Cav_img, dsize = (egg_resize_len,egg_resize_len) , interpolation=cv2.INTER_CUBIC)

                
                
                
            cavity = Cav_img
            normalized_cav = cavity
            #plt.imshow(normalized_cav,cmap='gray')
            #plt.title('r0='+str(temp_cav_r0)+' defoc='+str(temp_cav_defoc))
            #plt.colorbar()
            #plt.show()

            #normalized_cav = cavity#cavity*back_mean

            ##### Auto-Label ######
            cav_shape = np.shape(normalized_cav)
            cav_corner_val = normalized_cav[0,0]
            mask = np.full(cav_shape,0)
            mask_center = int(len(mask)/2)

            mask2=mask
            atol_val=.10
            atol_max = cav_corner_val - I_min
        
            while mask[mask_center,mask_center] == 0: #ie once the the center is labeled as cav exit loop         
                isclose = np.isclose(normalized_cav , I_min , atol=atol_val*atol_max)
                mask = ndimage.binary_fill_holes(isclose).astype(int)
                atol_val=atol_val+.05
            if mask[0,0] == 1: #super small cavs may still fail, then just make the bright spots lableled
                mask = np.where(normalized_cav>1.1 , 1, 0)
            

            blur_kern_size = int(np.round(.08 * len(mask)))
            mask=cv2.blur(mask,(blur_kern_size,blur_kern_size))
            labeled_mask, mask_labels = ndimage.label(mask) #This junk just find the center region
            cntr_mask_val = labeled_mask[mask_center,mask_center]  
            cav_mask = np.where(labeled_mask == cntr_mask_val, True, False)
            #plt.imshow(cav_mask)
            #plt.show()
            
            #FUTURE REF, other mask option, find darkest(min) of cavity, then find darkest pixel next to that, repeat until full circle or back at inital value?
            
            ### DQE Cav
            dqe = .6
            dqe_cav = (dqe * (normalized_cav - 1)) + 1
            #plt.imshow(dqe_cav)
            #plt.colorbar()
            #plt.show()
            
            #input('y')
            

            
            
            
            ######### MTF #########
            cav_mtf = cv2.cvtColor(np.uint8(dqe_cav), cv2.COLOR_GRAY2BGR)
                         
            zero = np.zeros((np.shape(cav_mtf)[0],np.shape(cav_mtf)[1],2))
            x_center = int((np.shape(cav_mtf)[0]-1)/2)
            y_center = int((np.shape(cav_mtf)[1]-1)/2)
            for x in range(0,np.shape(cav_mtf)[0]):
                for y in range(0,np.shape(cav_mtf)[1]):
                    zero[x][y] += np.sqrt((x-x_center)**2 + (y-y_center)**2)
            mtf_mask = np.zeros((np.shape(cav_mtf)[0],np.shape(cav_mtf)[1],2))
            w_cutoff = .5
            mtf_norm = (zero[int(x_center)][0])/(1)
            MTF_empty = zero/mtf_norm
            it = np.nditer(MTF_empty, flags=['multi_index'])
            for omega in it:
                if omega <= w_cutoff:
                    mtf_mask[it.multi_index] = np.exp(-omega/.2) #(Van Den Broek 2020 Paper, .2 was given value)
                else:
                    mtf_mask[it.multi_index] = 0
            mtf_dft = cv2.cvtColor(cav_mtf, cv2.COLOR_BGR2GRAY)       
            dft = cv2.dft(np.float32(mtf_dft),flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft) 
            fshift = dft_shift * mtf_mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            norm_img_back = cv2.normalize(img_back, None, np.min(dqe_cav), np.max(dqe_cav), cv2.NORM_MINMAX, cv2.CV_32F)            
            
            #plt.imshow(norm_img_back,cmap='gray')
            #plt.colorbar()
            #plt.show()
            
            ###Sloppy fix to weird mtf background problem
            norm_img_back = np.where(cav_mask==True,norm_img_back,dqe_cav)
            #plt.imshow(norm_img_back,cmap='gray')
            #plt.colorbar()
            #plt.show()
            
            
            ##### Noise #####
                
            noise = np.arange(.90,1.10,.01)
            it = np.nditer(norm_img_back, flags=['multi_index'])
            for I in it:
                dqe_cav[it.multi_index] = np.random.choice(noise)*I
                
            
            #####   Clone In    #####
                
            cav_size = len(cavity)
            cav_xwidth = np.shape(cavity)[0]
            cav_yheight = np.shape(cavity)[1]
            cav_xcen = rd.randrange(int(cav_size/2 +1),back_width-int(cav_size/2 +1))
            cav_ycen = rd.randrange(int(cav_size/2 +1),back_height-int(cav_size/2 +1))
            cav_upleft_x = cav_xcen - int(cav_xwidth/2)
            cav_upleft_y = cav_ycen - int(cav_yheight/2)
            
            back_median = np.percentile(background , 50)
            back_dark = np.percentile(background , 5)
            back_bright = np.percentile(background , 95)
            back_norm = background/back_median
            
            
            back_mask = np.full(np.shape(background), False)
            z=np.full(np.shape(cav_mask),True)
            back_mask[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight] = cav_mask
            sloppy_back_copy = background
            overlap_check = np.where(back_mask==True , cum_mask, False)
            
            whoops_count = 0
            while np.any(overlap_check == True):  #overlapping check is following loop
                cav_size = len(cavity)
                cav_xwidth = np.shape(cavity)[0]
                cav_yheight = np.shape(cavity)[1]
                cav_xcen = rd.randrange(int(cav_size/2 +1),back_width-int(cav_size/2 +1))
                cav_ycen = rd.randrange(int(cav_size/2 +1),back_height-int(cav_size/2 +1))
                cav_upleft_x = cav_xcen - int(cav_xwidth/2)
                cav_upleft_y = cav_ycen - int(cav_yheight/2)
                normalized_cav2 = cv2.normalize(normalized_cav, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #probably needs to better
                back_mask = np.full(np.shape(background), False)
                back_mask[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight] = cav_mask
                overlap_check = np.where(back_mask==True , cum_mask, False)
                whoops_count+=1
                #print('whoops, overlap!')
                if whoops_count >= 10:
                    img_full = True
                    #print('MEGA OOPS!')
                    break
            if img_full: #finish image if 10 attempts for 1 cavity fail
                break

            
            back_cav_area = background[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight]
            sloppy_cav_area = back_cav_area * norm_img_back
            if np.any(sloppy_cav_area > 254):
                darkest_p = np.min(sloppy_cav_area)
                sloppy_cav_area = cv2.normalize(sloppy_cav_area, None, darkest_p, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
                #print('too bright')
            temp_back = np.full(np.shape(background), 0)
            temp_back[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight] = sloppy_cav_area
               
            cav_test = norm_img_back/norm_img_back[0,0].astype(float)
            flat_w_cavs[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight] = flat_w_cavs[cav_upleft_x:cav_upleft_x+cav_xwidth , cav_upleft_y:cav_upleft_y+cav_yheight]  * cav_test
            #plt.imshow(flat_w_cavs,cmap='gray'),plt.colorbar(),plt.show(),input('y')
            
            background = np.where(back_mask==True , temp_back , background)
            flat_back_temp = background
            flat_back = flat_back_temp
            cum_mask += back_mask
            
            #plt.imshow(flat_back,cmap='gray')
            #plt.show()
            #plt.imshow(cum_mask),plt.colorbar,plt.show(),input('y')

            
            ####Find segmentation and bbox for indvidual cavity in full image
            poly = Polygons.from_mask(back_mask)
            fortran_ground_truth_binary_mask = ~np.asfortranarray(back_mask)
            encoded_ground_truth = mk.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mk.area(encoded_ground_truth)
            cav_density += ground_truth_area
            
            tuples=[]
            seg=poly.segmentation[0]
            for j in range(0,len(seg),2):
                tuples.append((seg[j] , seg[j+1]))
            temp_poly = Polygon(tuples)
            wrong_bbox = temp_poly.bounds
            x , y = int(wrong_bbox[0]) , int(wrong_bbox[1])
            x_len = int(wrong_bbox[2]) - int(wrong_bbox[0])
            y_len = int(wrong_bbox[3]) - int(wrong_bbox[1])
            avg_bbox_size = (y_len+x_len)/2
            best_bbox = [x , y , x_len , y_len]
            
            
            if defoc_polarity == 1:
                category_id_val = 1 #overfocused
            if defoc_polarity == -1:
                category_id_val = 2 #underfocused 
            
            ant = {"id":t,
                                "image_id":0,
                                "category_id":category_id_val,
                                "segmentation":[poly.segmentation[0]], #no idea why the [0] is needed
                                "area":int(ground_truth_area),
                                "bbox": best_bbox,   #old bbox =[cav_xcen-cav_size,cav_ycen-cav_size,cav_size,cav_size],
                                "iscrowd":0}
            annotations.append(ant)
            t +=1
            Master_cav_r0_list.append(temp_cav_r0)
            Master_cav_bbox_list.append(100 * avg_bbox_size/Artifical_img_size)
            Master_cav_defoc_list.append(temp_cav_defoc)
            
            print(f'\r{t} cavs have been added (out of %s)' % str(len(cav_size_list)), end='', flush=True)
        total_cavs += len(annotations)
        print('Total Cavs:'+str(total_cavs))
        
        temp_data_name = str( str(iteration) + '-' + str(bckgrnd_val) + '-' +''.join(rd.choices(string.ascii_uppercase + string.digits, k=6)))
        
        info = {"year":2020,
                  "version":"v2",
                  "description":"Rando Cav Integrator",
                  "contributor":"Matt Lynch",
                  "url":"https://www.nome.engin.umich.edu/",
                  "date_created":"9/13/2020"}
        images = [{"id":0,
                     "width":back_width,
                     "height":back_height,
                     "file_name":"%s.png" % (temp_data_name),
                     "license":'X',
                     "date_captured":"X"}]
        licenses = [{"id":0,
                        "name":"Unknown",
                        "url":""}]
        categories = [ {"id":1,"name":"overfocused cav","supercategory":"Cavity"},{"id":2,"name":"underfocused cav","supercategory":"Cavity"}] #Ryan add
            
        all_data = {"info":info,
                        "images":images,
                        "annotations":annotations,
                        "licenses":licenses,
                        "categories":categories}
            
        jsonfile = "%s/Annotations/%s.json" % (str(export_loc) , temp_data_name)
        #jsonfile = "%s/Annotations/iter%s-img%s.json" % (str(export_loc) , str(iteration), str(bckgrnd_val))
        with open(jsonfile,'w') as outfile:
            json.dump(all_data, outfile)
            
        
        flat_back_clone = cv2.cvtColor(np.uint8(flat_back), cv2.COLOR_GRAY2BGR)
        plt.imshow(flat_back_clone, cmap="gray")
        plt.show()
    

            
                    ####### Final Cloning #######    
            
        total_mask_fill_val = np.random.choice(np.arange(85,255,1))  #1/3 to 2/3rds of 255, higher means more visable cavs
        total_mask = np.uint8(np.full(np.shape(flat_back_clone), total_mask_fill_val ))
        big_clone_mask = np.uint8(np.where(cum_mask==True,255,0))
        big_clone_mask=np.dstack([big_clone_mask]*3)
        plt.imshow(big_clone_mask)
        plt.show()
            
        plt.imshow(back_clone, cmap="gray")
        plt.show()
            
        #back_clone = cv2.normalize(back_clone, None, 0*back_mean, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
        back_clone[back_clone>255] = 255
        back_clone[back_clone<0] = 0

        
        ### THIS IS ONLY FOR TEST 3!!!!!!!!!
        #img_back_convert = cv2.normalize(mtf_dft, None, 0*back_mean, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        #img_back_convert = cv2.normalize(img_back, None, 0*back_mean, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
        #testz = np.where(flat_back == l
        #le, percentile , img_back_convert)
        #plt.imshow(img_back_convert, cmap="gray")
        #plt.show()
        #cv2.imwrite("img_back_convert.png", img_back_convert)
            
        #flat_back_clone =cv2.cvtColor(img_back_convert, cv2.COLOR_GRAY2BGR)
        Synthetic_n = cv2.seamlessClone(np.uint8(flat_back), back_clone, total_mask, (int(back_clone.shape[1]/2),int(back_clone.shape[0]/2)), cv2.NORMAL_CLONE)
        Synthetic2 = cv2.seamlessClone(np.uint8(flat_back), back_clone, total_mask, (int(back_clone.shape[1]/2),int(back_clone.shape[0]/2)), cv2.MIXED_CLONE)

        Synth2_blur = cv2.GaussianBlur(Synthetic2 , (3,3), cv2.BORDER_DEFAULT)
        
        plt.imshow(Synthetic_n, cmap="gray")
        plt.show()

        cv2.imwrite("%s/Images/cutnpaste/%s.png" % (str(export_loc) , (temp_data_name)), flat_back)         
        cv2.imwrite("%s/Images/clone_n/%s.png" % (str(export_loc) , (temp_data_name)), Synthetic_n)
        cv2.imwrite("%s/Images/clone_m/%s.png" % (str(export_loc) , (temp_data_name)), Synthetic2)
        cv2.imwrite("%s/Images/clone_m_blur/%s.png" % (str(export_loc) , (temp_data_name)), Synth2_blur)
        flat_w_cavs_norm = cv2.normalize(flat_w_cavs, None , 0 , 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite("%s/Images/flat_back/%s.png" % (str(export_loc) , (temp_data_name)), flat_w_cavs_norm)
        #np.save("%s/Images/flat_back/%s" % (str(export_loc) , (temp_data_name)), flat_w_cavs)        

        bbox_mask = np.full(np.shape(Synth2_blur),False)
        for x in range(len(annotations)):
            x , y , x_len , y_len = annotations[x]['bbox']
            bbox_mask[y:y+y_len,x:x+x_len,:] = True
                    
            
        empty_back_spots = np.where(bbox_mask==True,Synth2_blur,0)
        cv2.imwrite("%s/Images/clone_m_blur_empty/%s.png" % (str(export_loc) , (temp_data_name)), empty_back_spots)


        #cv2.imwrite('Artifical Test v4.7/images/method2/ac-iteration%s-image%s-2.png' % (str(iteration),str([bckgrnd_val])), Blended)

                
        total_img_ant_pixels = sum(sum(cum_mask))
        Master_ant_pixels_list = np.append(Master_ant_pixels_list , total_img_ant_pixels)

        BackgroundsDF = pd.concat([BackgroundsDF,pd.DataFrame([{'Name':temp_back, 'Scale':temp_scale , 'Image':temp_img , 'Artificial Scale':temp_art_scale}])],ignore_index=True)
        SynthImgDF = pd.concat([SynthImgDF,pd.DataFrame([{'Image Name':temp_data_name , 'Resolution':Artifical_img_size , 'Scale':back_res , 'Field of view (nm^2)':back_fov , 'Total Cavities':len(annotations) , 'Avg Size(nm)':avg_cav_size}])],ignore_index=True)
        

        print(str(iteration) +'-'+ background_name[33:]+' is done')
        end = timer()
        print(end - start)
        print('cavs per second is:'+ str(total_cavs/(end-start)))
        img_count +=1
    #break#Makes every background only used once
    iteration +=1
end_gen = timer()

plt.hist(Master_cav_r0_list,bins=np.linspace(1,50,50))
plt.title('Database Cavity Size distribution')
plt.xlabel('Radius (nm)')
plt.ylabel('Count')
plt.savefig('%s/Metadata/Total Cav Distribution' % (str(export_loc)))
plt.show() 

plt.hist(Master_cav_bbox_list, bins=np.arange(0,50,1) ) 
plt.title('Database BBox Size distribution')
plt.xlabel('BBox Size (% of Image)')
plt.ylabel('Count')
plt.savefig('%s/Metadata/Total BBox Relative Size Distribution' % (str(export_loc)))
plt.show() 


Img_ant_percent = Master_ant_pixels_list/(Artifical_img_size**2)
DB_ant_pixel = sum(Master_ant_pixels_list)
txt_info  = str( 'Total Cavities: %s \n' % len(Master_cav_r0_list) +
                'Average Cavity Size: %s nm \n' % np.round(np.mean(Master_cav_r0_list),2) +
                'Total Images: %s \n' % img_count +
                'Average Cavities Per Image: %s \n' %  np.round(len(Master_cav_r0_list)/img_count,1) +
                '\n' +
                'Total True Positive Pixels: %s \n' % DB_ant_pixel+
                '%s%s of all Database Pixels \n' % (100*np.round( DB_ant_pixel / (img_count*Artifical_img_size**2),1) , '%') +
                'Average BBox Size: %s (Percent of Image) \n' % np.round(np.mean(Master_cav_bbox_list),2) +
                '\n' +
                'Time To Generate Database: %s seconds \n' % np.round(end_gen - start)+
                'Average time per Cavity: %s seconds \n' % np.round((end_gen - start)/len(Master_cav_r0_list),3))

with open("%s/Metadata/DB Info.txt" % (export_loc), "w") as text_file:
     text_file.write(txt_info)
SynthImgDF.to_csv('%s/Metadata/Per Img Data.csv' % (str(export_loc)))


plt.scatter(Master_cav_r0_list,Master_cav_defoc_list)
plt.show()




end = timer()
print(end - start)

