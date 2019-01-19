# -*- coding: utf-8 -*-
from tkinter import filedialog
from tkinter import Tk, Label, Button
from tkinter.messagebox import showerror

from PIL import Image
from PIL import ImageTk
import numpy as np

class GUI:
    path = None
    orig = None
    predicted = None
    tkRoot = None
    btn = None
    
    def __init__(self):
        self.root = Tk()
        self.root.winfo_toplevel().title("U-Net data segmentation") 
        
        img = ImageTk.PhotoImage(Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)))
        
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
            img = (Image.open(self.path).resize((256,256)))
            
            origImg = ImageTk.PhotoImage(img)
            predictedImg = ImageTk.PhotoImage(img)
            
            self.panelA.configure(image=origImg)
            self.panelB.configure(image=predictedImg)
            self.panelA.image = origImg
            self.panelB.image = predictedImg
        except:
            showerror(title='Error', message='Invalid image')

            

start = GUI()
        
        
        
        
