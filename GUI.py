# -*- coding: utf-8 -*-
from tkinter import filedialog
from tkinter import Tk, Label, Button
from tkinter.messagebox import showerror

from PIL import Image
from PIL import ImageTk
import numpy as np

from keras.models import model_from_json

import matplotlib.pyplot as plt
import os

class GUI:
    path = None
    orig = None
    predicted = None
    tkRoot = None
    btn = None
    model = None
    
    def __init__(self):
        
        # Model reconstruction from JSON file
        with open('model.json', 'r') as f:
            self.model = model_from_json(f.read())
        
        # Load weights into the new model
        self.model.load_weights('model_weights.h5')
        
        self.root = Tk()
        self.root.winfo_toplevel().title("U-Net data segmentation") 
        
        img = ImageTk.PhotoImage(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
        
        self.panelA = Label(image=img)
        self.panelA.image = img
        self.panelA.pack(side="left", padx=10, pady=10)
 
        self.panelB = Label(image=img)
        self.panelB.image = img
        self.panelB.pack(side="right", padx=10, pady=10)
        
        self.btn = Button(self.root, text="Select an image", command=self.newImage)
        self.btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        self.root.mainloop()


    
    def newImage(self):
        self.path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        
        if not self.path:
            showerror(title='Error', message='Select a file')
            return
        
        try:
            img = (Image.open(self.path))
            origImg = ImageTk.PhotoImage(img.resize((512, 512)))
            predict = np.expand_dims(np.asarray(img.resize((128, 128))), axis=0)
            predicted = self.model.predict(predict)
            predicted = np.argmax(predicted[0], axis=2)
            plt.imsave('predicted.png', predicted)
            predictedImg = ImageTk.PhotoImage(Image.open('predicted.png').resize((512, 512)))
            os.remove('predicted.png')
    
            self.panelA.configure(image=origImg)
            self.panelB.configure(image=predictedImg)
            self.panelA.image = origImg
            self.panelB.image = predictedImg
        except:
            showerror(title='Error', message='Invalid image')



            

start = GUI()
        
        
        
        
