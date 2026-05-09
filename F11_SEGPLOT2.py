from __future__ import print_function
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def segplot(pathm, lim, image, predmask, grmask, trMeanR, trMeanG, trMeanB, indx):
    # print('image shape',image.shape, 'output shape', predmask.shape, 'groundtruth shape', grmask.shape)
       
    image[:,:,0] = image[:,:,0] + trMeanR
    image[:,:,1] = image[:,:,1] + trMeanG
    image[:,:,2] = image[:,:,2] + trMeanB
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    rgbname="rgb_{}.png".format(indx)
    cv2.imwrite(os.path.join(pathm, rgbname),np.uint8(255*image))
    
    """
    rgb = np.random.randint(255, size=(lim,lim,3),dtype=np.uint8)
    name='rndrgb.png'
    cv2.imwrite(os.path.join(pathm, name),rgb)
    img = cv2.imread(os.path.join(pathm, name))
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #print('gray shape:',gray.shape)
    #gray = image[:,:,0]
    
    h, s, v = cv2.split(hsv_image)
    """
       
    #h = np.zeros((lim,lim), np.uint8)
    #s = np.zeros((lim, lim), np.uint8)
    #v = np.zeros((lim, lim), np.uint8)
    
    """
    gmask = grmask+predmask
    predmask = np.squeeze(predmask)
    grmask = np.squeeze(grmask)
    """
    
    v= image[:,:,0]/4 + np.squeeze(predmask)/2 + np.squeeze(grmask)/4
    #print(np.squeeze(grmask+predmask).dtype)
    s = np.minimum(np.squeeze(grmask+predmask),np.ones((lim, lim), np.float32))
    """
    for i in range(0, len(gmask)):
        newg = np.squeeze(gmask[i])
        s[i]=np.minimum(newg, 1)
    """    
    h= 0.75-np.squeeze(grmask)/2
    
    # print(s[55,76])
    
    h = h*179
    v = v*255
    s = s*255

    
    h = h.astype(np.uint8)
    #h = np.squeeze(h)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)
    #v = np.squeeze(v)
    
    #v = v[0]
    #print(h[0,0])
    #print(s[0,0])
    #print(v[0,0])
    #print('h shape',h.shape, 's shape', s.shape, 'v shape', v.shape)
    
    hsv_image = cv2.merge([h, s, v])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    outname="segmentation_image_{}.png".format(indx)
    cv2.imwrite(os.path.join(pathm, outname),out) 
    # image = image.astype(np.uint8)


    maskname="mask_{}.png".format(indx)
    cv2.imwrite(os.path.join(pathm, maskname),grmask*255) 
    # plt.imsave(os.path.join(pathm, 'test_image.png'),image)
    # plt.imsave(os.path.join(pathm, 'test_image_R.png'),image[:,:,0],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_image_G.png'),image[:,:,1],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_image_B.png'),image[:,:,2],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_pred_mask.png'),np.squeeze(predmask))
    # plt.imsave(os.path.join(pathm, 'ground_truth_mask.png'),np.squeeze(grmask))
                







