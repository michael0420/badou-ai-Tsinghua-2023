import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcess:
    def __init__(self,image_path):
        self.image=cv2.imread(image_path)
        self.image_plt=plt.imread(imagePath)
        self.image_width=self.image.shape[:2][0]
        self.image_height=self.image.shape[:2][1]
        self.image_gray=np.zeros([self.image_width,self.image_height],self.image.dtype)
        self.image_gray_plt=np.zeros([self.image_width,self.image_height],self.image_plt.dtype)
        self.image_thresholding=np.zeros([self.image_width,self.image_height])
        self.image_thresholding_plt = np.zeros([self.image_width, self.image_height])
        self.image_gray_ok=False
        self.image_thresholding_ok=False

    def handleGray(self):
        for i in range(self.image_width):
            for j in range(self.image_height):
                m=self.image[i,j]
                n=self.image_plt[i,j]
                self.image_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
                self.image_gray_plt[i,j]=n[0]*0.3+n[1]*0.59+n[2]*0.11
                self.image_thresholding[i,j]=round(self.image_gray[i,j]/255,2)
                self.image_thresholding_plt[i,j]=round(self.image_gray_plt[i,j],2)
                if self.image_thresholding[i, j]<=0.5:
                    self.image_thresholding[i,j]=0
                else:
                    self.image_thresholding[i,j]=1

                if self.image_thresholding_plt[i, j]<=0.5:
                    self.image_thresholding_plt[i,j]=0
                else:
                    self.image_thresholding_plt[i,j]=1
        self.image_gray_ok=True
    def gray(self):
        self.image_gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image_gray_plt=cv2.cvtColor(self.image_plt,cv2.COLOR_BGR2GRAY)
        self.image_gray_ok = True
    def thresholding(self):
        self.image_thresholding_ok=True
        return self.image_thresholding
    def thresholdingPlt(self):
        if self.image_thresholding_plt[0,0]>1 or self.image_thresholding_plt[0,0]==0:
            print("Please make gray first")
        else:
            self.image_thresholding_ok=True
            return self.image_thresholding_plt
    def cv2show(self):
        if (self.image_gray_ok==True and self.image_thresholding_ok==True):
            cv2.imshow("The Origin Picture",self.image)
            cv2.imshow("The Gray picture",self.image_gray)
            cv2.imshow("The Thresholding Picture",self.image_thresholding)
            cv2.waitKey()
        elif(self.image_gray_ok==True and self.image_thresholding_ok==False):
            cv2.imshow("The Origin Picture", self.image)
            cv2.imshow("The Gray picture", self.image_gray)
            cv2.waitKey()
        elif(self.image_gray_ok==False and self.image_thresholding_ok==False):
            cv2.imshow("The Origin Picture", self.image)
            cv2.waitKey()
    def pltshow(self):
        if (self.image_gray_ok==True and self.image_thresholding_ok==True):
            plt.subplot(1,3,1)
            plt.imshow(self.image_plt)
            plt.subplot(1,3,2)
            plt.imshow(self.image_gray_plt,cmap="gray")
            plt.subplot(1,3,3)
            plt.imshow(self.image_thresholding_plt,cmap="gray")
            plt.show()
        elif(self.image_gray_ok==True and self.image_thresholding_ok==False):
            plt.subplot(1, 2, 1)
            plt.imshow(self.image_plt)
            plt.subplot(1, 2, 2)
            plt.imshow(self.image_gray_plt,cmap="gray")
            plt.show()
        elif(self.image_gray_ok==False and self.image_thresholding_ok==False):
            plt.subplot(1, 1, 1)
            plt.imshow(self.image_plt)
            plt.show()



if __name__=="__main__":
    imagePath="G:\CV Studies Files\Week One\study\lenna.png"
    imageProcess=ImageProcess(imagePath)
    imageProcess.handleGray()
    imageProcess.gray()
    imageProcess.thresholding()
    imageProcess.cv2show()
    imageProcess.thresholdingPlt()
    imageProcess.pltshow()


