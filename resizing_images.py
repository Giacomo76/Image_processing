# python resizing_images.py

import os
import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
 
def ridimensiona_immagine(img, larghezza_max, altezza_max):
    # Ottieni dimensioni originali
    altezza, larghezza = img.shape[:2]
    
    # Calcola il rapporto di scala
    rapporto = min(larghezza_max / larghezza, altezza_max / altezza)
    
    # Nuove dimensioni
    nuova_larghezza = int(larghezza * rapporto)
    nuova_altezza = int(altezza * rapporto)
    
    # Ridimensiona mantenendo le proporzioni
    return cv2.resize(img, (nuova_larghezza, nuova_altezza), interpolation=cv2.INTER_AREA)

def ridimensiona_cartella(cartella_input, cartella_output, larghezza_max, altezza_max):
    if not os.path.exists(cartella_output):
        os.makedirs(cartella_output)
    
    # Scansiona la cartella
    for filename in os.listdir(cartella_input):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            percorso_file = os.path.join(cartella_input, filename)
            immagine = cv2.imread(percorso_file)

            if immagine is None:
                print(f"Errore caricamento immagine: {filename}")
                continue

            immagine_ridimensionata = resize(immagine, width=larghezza_max, height=altezza_max, inter=cv2.INTER_AREA)
            #ridimensiona_immagine(immagine, larghezza_max, altezza_max)
            percorso_output = os.path.join(cartella_output, filename)
            cv2.imwrite(percorso_output, immagine_ridimensionata)
            print(f"Immagine salvata: {percorso_output}")


cartella_input = "immagini_originali"   # Cartella con le immagini originali
cartella_output = "immagini_ridimensionate"  # Cartella di destinazione
larghezza_max = 128   # Imposta la larghezza massima desiderata
altezza_max = 128    # Imposta l'altezza massima desiderata

# Esegui
ridimensiona_cartella(cartella_input, cartella_output, larghezza_max, altezza_max)
