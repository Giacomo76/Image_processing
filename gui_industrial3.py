# python gui_industrial3.py

from imutils.video import VideoStream
#import RPi.GPIO as GPIO
from tkinter import * 
from tkinter import filedialog 
import cv2 
import imutils
import numpy as np
from PIL import Image, ImageTk 
import datetime 
import tkinter as tk
from tkinter import ttk
import argparse
import signal
import time
import sys

# Soglia delle distanze da visualizzare
DISTANZA_MIN = 2
DISTANZA_MAX = 50
# Fattore di conversione: 1 pixel corrisponde a 0.05 mm
conversion_factor = 0.05 # mm per pixel
video_source = 0  # Videocamera USB predefinita

PINK = "#e2979c"
RED = "#e7305b"
GREEN = "#9bdeac"
YELLOW = "#f7f5dd"
BLUE = "#678ac2"
FONT_NAME = "Courier"
ANTRACITE = "#293133"

# Create the main application window 
root = Tk() 
root.title("Industrial Vision Control for T10-Glass dome ") 
root.geometry("1800x1000") 
root.configure(bg="black",padx=5,pady=5)

roi_top = tk.IntVar(value=100)
roi_bottom = tk.IntVar(value=200)
roi_left = tk.IntVar(value=100)
roi_right = tk.IntVar(value=200)

def accendi_illuminatore(self):
    # Accendi l'illuminatore
    GPIO.output(ILLUMINATOR_PIN, GPIO.HIGH)
    
def spegni_illuminatore(self):
    # Accendi l'illuminatore
    GPIO.output(ILLUMINATOR_PIN, GPIO.LOW)
       
def calcola_tangente(contour, idx):
    """Calcola il vettore tangente in un punto specifico del contorno."""
    prev_idx = (idx - 1) % len(contour)
    next_idx = (idx + 1) % len(contour)
    tangente = contour[next_idx][0] - contour[prev_idx][0]
    return tangente / np.linalg.norm(tangente)       

# Function to close the window 
def close_window():
    
    root.destroy()
    
    # img = ImageTk.PhotoImage(Image.open(filename))
    # img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)   
# function to update the canvas widget
def display():
    global displayed_img, img_cv, img
    if displayed_img:
        canvas.delete(displayed_img)
    displayed_img = None
    if img_cv is not None:
        img = ImageTk.PhotoImage(Image.fromarray(img_cv))
        #img=imutils.resize(img, width=200)
        displayed_img = canvas.create_image(0, 0, image=img, anchor=NW)
        canvas.image = img
 
# action for Button widget
def load():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *")])
    if filename:
        global displayed_img, img_cv, img
        img_cv= cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        img_cv=imutils.resize(img_cv, width=200)
        display()
 
# Create labels 
fng_image = Image.open("C:\\Users\\jAck\Documents\\Python Scripts\\esempi\\T10\\fng.jpg")
fng_image = fng_image.resize((140, 60), Image.ANTIALIAS)
logo_fng = ImageTk.PhotoImage(fng_image)
company_logo =  Label(root,image=logo_fng , borderwidth=2, relief="ridge")  
company_logo.grid(row=0,column=0,padx=5,pady=5, sticky="nw")
company_logo.image = logo_fng
company_logo2 =  Label(root,image=logo_fng , borderwidth=2, relief="ridge")  
company_logo2.grid(row=0,column=2,padx=5,pady=5, sticky="nw")
company_logo2.image = logo_fng
# Position image
#company_logo.place(x=0, y=0)

company_label = Label(root, text="Industrial Vision Control for T10-Glass dome dimensions", borderwidth=5, relief="ridge", font=("Arial Black", 24),fg="light grey", bg=ANTRACITE)
company_label.grid(row=0,column=0,columnspan=3,padx=230,pady=5, sticky="ew") 
version_label = Label(root, text="Version 1.0",font=("Arial Black", 10), fg="light grey", bg="black") 
version_label.grid(row=1,column=2,padx=0,pady=0,sticky="e")
vision_label = Label(root, text="Vision System", font=("Helvetica", 16,"bold"), fg="light grey", bg="black")
vision_label.grid(row=2,column=0,columnspan=3,padx=0,pady=5, sticky="ew")

# Create a frame to hold the canvases 
frame = Frame(root,width=800,height=800,relief="ridge",borderwidth=1, bg="black")
frame.grid(row=3,column=0,padx=10,pady=0)

# Create canvases 
video_canvas1 = Canvas(frame, borderwidth=2, width=200, height=400, bg= ANTRACITE)#320  240
video_canvas1.grid(row=3,column=0,padx=5,pady=5)
video_canvas1.create_text(80, 10, text="Smoothing", font=("Helvetica", 10),fill="light grey", anchor="ne")
video_canvas2 = Canvas(frame, borderwidth=2, width=200, height=400, bg=ANTRACITE)
video_canvas2.grid(row=3,column=1,padx=5,pady=5)
video_canvas2.create_text(80, 10, text="Threshold", font=("Helvetica", 10),fill="light grey", anchor="ne")
video_canvas3 = Canvas(frame, borderwidth=2, width=200, height=400, bg=ANTRACITE)
video_canvas3.grid(row=3,column=2,padx=5,pady=5)
video_canvas3.create_text(110, 10, text="Edge detection", font=("Helvetica", 10),fill="light grey", anchor="ne")
video_canvas4 = Canvas(frame, borderwidth=2, width=640, height=450, bg=ANTRACITE)
video_canvas4.grid(row=3,column=3,padx=5,pady=5)
video_canvas4.create_text(120, 10, text="T10 Analisi cupola ", font=("Helvetica", 10),fill="light grey", anchor="ne")
# create a canvas to display the image
canvas = Canvas(frame, borderwidth=2 ,width=200, height=400, bg=ANTRACITE)
canvas.grid(row=3, column=4, padx=5, pady=5)

######################## Smoothing Panel
smoothing_frame = tk.LabelFrame(frame, text="Smoothing", bg="black",font=("Helvetica", 12,"bold"), fg="grey", padx=5,pady=5)
smoothing_frame.grid(row=0, column=0, padx=15,pady=15)
smooth_method = tk.StringVar(value='None')
smooth_box = ttk.Combobox(smoothing_frame, textvariable=smooth_method, values=['None', 'GaussianBlur', 'Blur', 'BoxFilter', 'MedianBlur', 'Bilateral'])
smooth_box.pack()                   
kernel_slider = tk.Scale(smoothing_frame, from_=1, to=31, resolution=2, label='Kernel (odd only)', orient='horizontal')
kernel_slider.set(3)
kernel_slider.pack()
######################## Thresholding Panel
thresholding_frame = tk.LabelFrame(frame, text="Thresholding", bg="black",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
thresholding_frame.grid(row=0, column=1, padx=15,pady=15)
thresh_method = tk.StringVar(value='None')
thresh_box = ttk.Combobox(thresholding_frame, textvariable=thresh_method, values=['None','Normal','Inverse','Gaussian','GaussianInv','Otsu'])
thresh_box.pack()
thresh_min = tk.Scale(thresholding_frame, from_=0, to=255, label='Thresh Min', orient='horizontal')
thresh_min.set(100)
thresh_min.pack()
thresh_max = tk.Scale(thresholding_frame, from_=0, to=255, label='Thresh Max', orient='horizontal')
thresh_max.set(200)
thresh_max.pack()
######################## Edge Detection Panel
edge_frame = tk.LabelFrame(frame, text="Edge Detection", bg="black",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
edge_frame.grid(row=0, column=2, padx=15,pady=15)
edge_method = tk.StringVar(value='None')
edge_box = ttk.Combobox(edge_frame, textvariable=edge_method, values=['None', 'Sobel_1','Sobel_Magnitude','Sobel_Magnitude_2','Sobel_Orientation','Sobel_2','Sobel_3','Scharr', 'Prewitt', 'Canny'])
edge_box.pack()
edge_min = tk.Scale(edge_frame, from_=0, to=255, label='Edge Min', orient='horizontal')
edge_min.set(50)
edge_min.pack()
edge_max = tk.Scale(edge_frame, from_=0, to=255, label='Edge Max', orient='horizontal')
edge_max.set(150)
edge_max.pack()
edge_kernel = tk.Scale(edge_frame, from_=1, to=31, resolution=2, label='Edge Kernel (odd only)', orient='horizontal')
edge_kernel.set(3)
edge_kernel.pack()

######################### Slider ROI
roi_frame = tk.LabelFrame(frame, text="ROI window", bg="black",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
roi_frame.grid(row=0, column=3, padx=15,pady=15, sticky="nsw")#ew
tk.Label(roi_frame, text="ROI Top").grid(row=0, column=0, sticky="nsew")
tk.Scale(roi_frame, from_=0, to=480, orient="horizontal", variable=roi_top).grid(row=0, column=1)

tk.Label(roi_frame, text="ROI Bottom").grid(row=1, column=0, sticky="nsew")
tk.Scale(roi_frame, from_=0, to=480, orient="horizontal", variable=roi_bottom).grid(row=1, column=1)
        
tk.Label(roi_frame, text="ROI Left").grid(row=2, column=0, sticky="nsew")
tk.Scale(roi_frame, from_=0, to=640, orient="horizontal", variable=roi_left).grid(row=2, column=1)

tk.Label(roi_frame, text="ROI Right").grid(row=3, column=0, sticky="nsew")
tk.Scale(roi_frame, from_=0, to=640, orient="horizontal", variable=roi_right).grid(row=3, column=1)

# ill_frame = tk.LabelFrame(frame, text="Illuminatore",bg="blue",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
# ill_frame.grid(row=0, column=3, padx=5,pady=5, sticky="nse")
# illuminatore_button = tk.Button(ill_frame, text="Illuminatore ON", command=accendi_illuminatore)
# illuminatore_button.pack()#(fill=tk.X, padx=10, pady=10) 
# illuminatore_button = tk.Button(ill_frame, text="Illuminatore OFF", command=spegni_illuminatore)
# illuminatore_button.pack()#(fill=tk.X, padx=10, pady=10)

frame2 = Frame(root,width=800,height=800,relief="ridge",borderwidth=1, bg="black")
frame2.grid(row=4,column=0,padx=10,pady=10, sticky="nw")
# General Controls Panel
general_frame = tk.LabelFrame(frame2, text="General Controls",bg="black",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
general_frame.grid(row=0, column=0, padx=5,pady=5)#, sticky="nw")

brightness = tk.Scale(general_frame, from_=-100, to=100, label='Brightness', orient='horizontal')
brightness.set(0)
brightness.pack(side='left')
contrast = tk.Scale(general_frame, from_=-100, to=100, label='Contrast', orient='horizontal')
contrast.set(0)
contrast.pack(side='left')
exposure = tk.Scale(general_frame, from_=-100, to=100, label='Exposure', orient='horizontal')
exposure.set(0)
exposure.pack(side='left')
balance = tk.Scale(general_frame, from_=-100, to=100, label='Balance', orient='horizontal')
balance.set(0)
balance.pack(side='left')

# Illuminatore  Panel
ill_frame = tk.LabelFrame(frame2, text="Comando illuminatore",bg="black",font=("Helvetica", 12,"bold"),fg="grey", padx=5,pady=5)
ill_frame.grid(row=0, column=1, padx=5,pady=5)#, sticky="nw")
illuminatore_button = tk.Button(ill_frame, text="Illuminatore ON", command=accendi_illuminatore)
illuminatore_button.pack(side='left')#(fill=tk.X, padx=10, pady=10) 
illuminatore_button = tk.Button(ill_frame, text="Illuminatore OFF", command=spegni_illuminatore)
illuminatore_button.pack(side='right')#(fill=tk.X, padx=10, pady=10)

# Create a frame to hold the configuration options
entry_frame = Frame(frame2, width = 200, height = 600, bg="black")
entry_frame.grid(row=0, column=2, padx=5, pady=5)
mybutton= Button(entry_frame, width=20, height=2, borderwidth=2, relief="ridge", text='Load Reference Image', command=load) 
mybutton.grid(row=0, column=2, columnspan=1, padx=5,pady=5)#,sticky="nw") 

# create a close button
close_button = Button(root, width=20, height=2, borderwidth=2, relief="ridge", text="Esci", command=close_window) 
close_button.grid(row=4, column=0, columnspan=1, padx=5, pady=5, sticky="ne")

# labels = ["No. of pieces accepted:", "Delay time:", "No. of pieces rejected:"] 
# entries = [Entry(entry_frame, borderwidth=2, relief="ridge")for _ in range(3)]
# for i, label_text in enumerate(labels):
    # label = Label(entry_frame, text=label_text, font=("Helvetica", 18), fg="black", bg="blue")
    # label.grid(row=i+1, column=0, padx=5, pady=5, sticky="e")
    # entry = entries[i]
    # entry.grid(row=i+1, column=1, padx=5, pady=5, sticky="w")
#entry_frame.grid(row=6, column=0, padx=5, pady=5)  
 
def apply_smoothing(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = kernel_slider.get()
    if k % 2 == 0:
        k += 1
    method = smooth_method.get()
    if method == 'None':
        return img       
    elif method == 'GaussianBlur':
        return cv2.GaussianBlur(img, (k, k), 0)
    elif method == 'Blur':
        return cv2.blur(img, (k, k))
    elif method == 'BoxFilter':
        return cv2.boxFilter(img, -1, (k, k))            
    elif method == 'MedianBlur':
        return cv2.medianBlur(img, k)
    elif method == 'Bilateral':
        return cv2.bilateralFilter(img, k, 50, 50)  
    
    return img

def apply_thresholding(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = thresh_method.get()
    tmin = thresh_min.get()
    tmax = thresh_max.get()
    if method == 'None':
        return img
    elif method == 'Normal':
        _, result = cv2.threshold(img, tmin, tmax, cv2.THRESH_BINARY)
    elif method == 'Inverse':
        _, result = cv2.threshold(img, tmin, tmax, cv2.THRESH_BINARY_INV)        
    elif method == 'Gaussian':
         result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4.0)
    elif method == 'GaussianInv':
         result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4.0)             

    elif method == 'Otsu':
        _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)#cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

#cv2.adaptiveThreshold(	src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]	)

def apply_edge_detection(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = edge_method.get()
    tmin = edge_min.get()
    tmax = edge_max.get()
    k = edge_kernel.get()
    if k % 2 == 0:
        k += 1
    if method == 'None':
        return img            
    elif method == 'Sobel_1':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        result = cv2.convertScaleAbs(cv2.magnitude(dx, dy))
    elif method == 'Sobel_Magnitude':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        # dx_abs = np.uint8(np.absolute(dx))
        # dy_abs = np.uint8(np.absolute(dy))        
        result = cv2.convertScaleAbs(np.sqrt((dx ** 2) + (dy ** 2)))
    elif method == 'Sobel_Magnitude_2':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        sobel = np.sqrt(dx**2 + dy**2)
        result = np.uint8(255 * sobel / np.max(sobel))             
    elif method == 'Sobel_Orientation':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)              
        result = cv2.convertScaleAbs(np.arctan2(dx, dy) * (180 / np.pi) % 180)
    elif method == 'Sobel_2':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        dx = cv2.convertScaleAbs(dx)
        dy = cv2.convertScaleAbs(dy)
        result = cv2.addWeighted(dx, 0.5,dy, 0.5, 0)    
    elif method == 'Sobel_3':
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        dx_abs = np.uint8(np.absolute(dx))
        dy_abs = np.uint8(np.absolute(dy)) 
        result =  cv2.bitwise_or(dx_abs, dy_abs)        
   
    elif method == 'Scharr':
        dx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        dy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        result = cv2.convertScaleAbs(cv2.magnitude(dx, dy))
    elif method == 'Prewitt':
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        dx = cv2.filter2D(img, -1, kernelx)
        dy = cv2.filter2D(img, -1, kernely)
        result = cv2.convertScaleAbs(cv2.addWeighted(dx, 0.5, dy, 0.5, 0))
    elif method == 'Canny':
        result = cv2.Canny(img, tmin, tmax)
    
    return result

def apply_general_controls(img):
    _brightness = brightness.get()
    _contrast = contrast.get()
    img = cv2.convertScaleAbs(img, alpha= 1 + _contrast / 100.0, beta= _brightness)
    return img
   
def apply_dome_detector(frame,threshold):
    
    #frame = cv2.flip(frame, 0)
    (H, W) = frame.shape[:2]
    cx = W // 2
    cy = H // 2
    #cv2.rectangle(frame, (cx, cy), (cx + w, cy + h), (0, 255, 0),2)    

    # Seleziona la ROI 
    #x, y, w, h = 50, 100, 200, 270               # ROI generica
    #x, y, w, h = cx-(cx//2), cy-(cy//2), cx, cy  # ROI centrata sempre al frame

    
    #roi = threshold[y:y+h, x:x+w]
    
    top = roi_top.get()
    bottom = roi_bottom.get()
    left = roi_left.get()
    right = roi_right.get()
    
    #roi = threshold[top:bottom, left:right]
    y = top
    x = left
    w = right-left
    h = bottom - top
    roi = threshold[y:y+h, x:x+w]
    
    

    # Trova i contorni
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Escludi il contorno esterno basandoti sull'area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < (w * h) * 0.8]
    #contour1 = [0] + np.array([[0, 0]])
    #contour2 = [0] + np.array([[0, 0]])
     
    if len(contours) >= 2:
        # Seleziona i primi due contorni validi
        contour1 = contours[0] + np.array([[x, y]])
        contour2 = contours[1] + np.array([[x, y]])
        pts1 = contour1.reshape(-1, 2)
        pts2 = contour2.reshape(-1, 2)
        
        # Calcola la matrice delle distanze
        diff = pts1[:, None, :] - pts2[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Identifica le distanze che rientrano nell'intervallo richiesto
        mask = (distances >= DISTANZA_MIN) & (distances <= DISTANZA_MAX)
        
        # Inizializzazione Contatore
        contatore_segmenti = 0
        lunghezze_segmenti = []
        segmenti_con_lunghezze = []  # Lista per memorizzare (pt1, pt2, lunghezza)
        
        # Disegna solo i segmenti che sono tangenti
        for idx1, idx2 in zip(*np.where(mask)):
            pt1, pt2 = pts1[idx1], pts2[idx2]
            
            # Calcolo delle tangenti
            tangente1 = calcola_tangente(contour1, idx1)
            tangente2 = calcola_tangente(contour2, idx2)
            
            # Calcolo del vettore del segmento
            segmento = pt2 - pt1
            segmento_norm = segmento / np.linalg.norm(segmento)
            
            
            # Verifica se il segmento è tangente
            if np.isclose(np.dot(tangente1, segmento_norm), 0, atol=0.1) and np.isclose(np.dot(tangente2, segmento_norm), 0, atol=0.1):
                cv2.line(frame, tuple(pt1), tuple(pt2), (255, 255, 0), 1)  # Azzurro = solo segmenti tangenti
                contatore_segmenti += 1
                
                #Calcola e memorizza la lunghezza del segmento
                lunghezza = np.linalg.norm(segmento)
                lunghezze_segmenti.append((contatore_segmenti, lunghezza))
                segmenti_con_lunghezze.append((pt1, pt2, lunghezza))
                
            # Stampa le lunghezze dei segmenti trovati
            # for idx, lunghezza in lunghezze_segmenti:
                
                # #Calcolo delle distanze in millimetri
                # min_dist_mm = min_dist * conversion_factor
                # max_dist_mm = max_dist * conversion_factor
                # print(f"Segmento {idx}: Lunghezza = {lunghezza:.2f} pixel")
                
                
        
        # Trova il segmento più lungo e più corto
        if segmenti_con_lunghezze:
            segmento_max = max(segmenti_con_lunghezze, key=lambda x: x[2])
            segmento_max_mm = segmento_max[2] * conversion_factor
            segmento_min = min(segmenti_con_lunghezze, key=lambda x: x[2])
            segmento_min_mm = segmento_min[2] * conversion_factor
            
            # Disegna il segmento più lungo
            cv2.line(frame, tuple(segmento_max[0]), tuple(segmento_max[1]), (255, 0, 0), 2)
            cv2.putText(frame, f"Max: {segmento_max[2]:.2f}", (20,40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f"Max: {segmento_max[2]:.2f}", tuple(segmento_max[0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)                

         
            # Disegna il segmento più corto             
            cv2.line(frame, tuple(segmento_min[0]), tuple(segmento_min[1]), (0, 255, 0), 2)
            cv2.putText(frame, f"Min: {segmento_min[2]:.2f}", tuple(segmento_min[0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Min: {segmento_min[2]:.2f}", (20,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
            print(f"Numero di segmenti tangenti trovati: {contatore_segmenti}")
            print(f"Min: {segmento_min[2]:.2f}")
            print(f"Max: {segmento_max[2]:.2f}")
            print(f"Min_mm : {segmento_min_mm:.2f}")
            print(f"Max_mm : {segmento_max_mm:.2f}")    
            # cv2.putText(frame, f"Segmenti tangenti: {contatore_segmenti}", 
                # (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"{contatore_segmenti}", 
                (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                
        # Disegna i contorni rilevati
        cv2.drawContours(frame, [contour1], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [contour2], -1, (0, 255, 255), 1)
        
    cv2.putText(frame, f"Segmenti tangenti: ",(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Visualizza l'area ROI e gli ASSI X/Y
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
    
    cv2.line(frame, (0, int(frame.shape[0]/2)), (frame.shape[1],int(frame.shape[0]/2)), (255, 0, 0), 2) # REF. ASSE X
    cv2.line(frame, (int(frame.shape[1]/2), 0), (int(frame.shape[1]/2),frame.shape[0]), (255, 0, 0), 2) # REF. ASSE Y
    #cv2.rectangle(frame, (cx-(cx//2), cy-(cy//2)), (cx+(cx//2), cy+(cy//2)), (255, 255, 255),1)         # Mirino rettangolare
    
    #cv2.line(frame, (cx, cy),(int(cx-1),int(cy-1)), (0, 0, 255), 5) # REF. ASSE X
    #cv2.rectangle(frame, (cx-(w//2), cy-(h//2)), (cx + w, cy + h), (0, 0, 255),2)
    #cv2.imshow('Segmenti Tangenti', frame)

    return frame   

def update_image():
    try:
        global displayed_img, img_cv, img

        img = ImageTk.PhotoImage(Image.fromarray(img_cv))
    except AttributeError:
        print("")
    
# Function to update the video canvas with webcam feed
def update_video_canvas():

    ret, frame = cap.read()
    #vs = VideoStream(usePiCamera=False).start()
    #frame = vs.read()
    #frame = imutils.resize(frame, width=640)
    #frame = cv2.resize(frame,(40,40),interpolation= cv2.INTER_LANCZOS4)
    # imgn1 = cv2.resize(img1,(128,128),cv2.INTER_NEAREST)
    # imgbl1 = cv2.resize(img1,(128,128),cv2.INTER_LINEAR)
    # imgbc1= cv2.resize(img1,(128,128),cv2.INTER_CUBIC)
    # imgl1= cv2.resize(img1,(128,128),cv2.INTER_LANCZOS4)
        
    frame = cv2.flip(frame, 1)
    #while True:
    if ret :
        #frame = vs.read()
        #frame = cv2.flip(frame, 1)
        #frame = imutils.resize(frame, width=640)
        frame = apply_general_controls(frame)
        # Convert frame to grayscale 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #clahe = cv2.createCLAHE(clipLimit=args["clip"],tileGridSize=(args["tile"], args["tile"]))
        clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8, 8))
        equalized = clahe.apply(gray_frame)
        
        #equalized = cv2.equalizeHist(gray_frame)
        # Push the timestamp into the frame
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # dt = datetime.datetime.now()
        # st = dt.strftime("%d"+ "/"+ "%m"+ "/"+ "%y" +"  "+ "%H"+ ":" +"%M"+ ":" +"%S")
        
        smooth = apply_smoothing(equalized)
        
        #smooth = apply_smoothing(gray_frame)
        smooth= imutils.resize(smooth, width=400)
        # Convert to a format suitable for Tkinter 
        smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
        # smooth_rgb = cv2.putText(smooth_rgb, st, (10,50),font,0.8,(255,0,0)) 
        smooth_photo = ImageTk.PhotoImage(image=Image.fromarray(smooth_rgb)) 
        # Display edge detection in the second canvas 
        video_canvas1.create_image(0, 30, image=smooth_photo, anchor=NW) 
        video_canvas1.photo = smooth_photo
      
        thresh = apply_thresholding(smooth) #smooth
        # Convert to a format suitable for Tkinter 
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        thresh_photo = ImageTk.PhotoImage(image=Image.fromarray(thresh_rgb)) 
        # Display edge detection in the second canvas 
        video_canvas2.create_image(0, 30, image=thresh_photo, anchor=NW)  #n, ne, e, se, s, sw, w, nw, or center 
        video_canvas2.photo = thresh_photo

        edge = apply_edge_detection(smooth)#thresh
        # Convert to a format suitable for Tkinter 
        edge_rgb = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
        contours_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        edge_rgb[edge > edge_min.get()] = (0, 0, 255)

        edge_photo = ImageTk.PhotoImage(image=Image.fromarray(edge_rgb)) 
        # Display edge detection in the second canvas 
        video_canvas3.create_image(0, 30, image=edge_photo, anchor=NW) 
        video_canvas3.photo = edge_photo    
        
        #Detector viene inizializzato quando nella pipeline sono già presenti la conversione in grayscale,smooth_method, threshold ed edging
        detector = apply_dome_detector(frame,edge)
        detector_rgb = cv2.cvtColor(detector, cv2.COLOR_BGR2RGB)
        detector_photo = ImageTk.PhotoImage(image=Image.fromarray(detector_rgb))
        # Display detector in the fourth canvas
        video_canvas4.create_image(0, 30, image=detector_photo, anchor= NW)
        video_canvas4.photo = detector_photo
        
        
        # # Push the timestamp into the frame 
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # dt = str(datetime.datetime.now()) 
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # frame_rgb = cv2.putText(frame_rgb, dt, (10,50),font,1,(255,0,0)) 
        # # Display original video in the first canvas 
        # orig_photo = ImageTk.PhotoImage(image=Image. fromarray(frame_rgb)) 
        # video_canvas2.create_image(0, 30, image=orig_photo, anchor=NW) 
        # video_canvas2.photo = orig_photo 
        video_canvas1.after(10, update_video_canvas)
        
# # Open the webcam 
cap = cv2.VideoCapture(video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

# vs = VideoStream(src=0,usePiCamera=False).start()
# time.sleep(2.0)

# Call the update_video_canvas function to start displaying the video feed 
update_video_canvas()

# intialize the global variables
img_cv = None
displayed_img = None
img = None

# Start the tkinter main loop 
root.mainloop() 

# Release the webcam and close OpenCV when the window is closed 
cap.release() 
cv2.destroyAllWindows()
        
        
 
        
        
    
    
    

