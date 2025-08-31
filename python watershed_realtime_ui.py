# cd C:\Users\jAck\Documents\Python Scripts\esempi\T10\VideoCapture with threading and multiprocessing
#python watershed_realtime_ui.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Affidabile acquisizione video con threading + multiprocessing per segmentazione Watershed (OpenCV + NumPy + Tkinter)

Requisiti:
- Python 3.8+
- opencv-python (cv2)
- numpy
- pillow (per visualizzare i frame in Tkinter)

Esecuzione:
    python3 watershed_realtime_ui.py

Tasti rapidi:
    q       -> Esci
    s       -> Screenshot del frame elaborato in ./captures/
    space   -> Pausa/Resume acquisizione

Note:
- Il thread di acquisizione cattura i frame dalla camera senza bloccare la UI.
- Un processo separato esegue la pipeline di segmentazione watershed (CPU-bound).
- Le code (multiprocessing.Queue) gestiscono il passaggio dei frame e dei risultati.
- I parametri principali sono regolabili in tempo reale tramite slider Tkinter.
"""

import os
import sys
import time
import queue
import ctypes
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    # Pillow è consigliato per mostrare i frame in Tkinter
    from PIL import Image, ImageTk
except Exception as e:
    print("Errore: è necessario installare Pillow (pip install pillow) per la visualizzazione Tkinter.", file=sys.stderr)
    raise

import multiprocessing as mp
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# -------------------- Config --------------------

DEFAULT_CAMERA_INDEX = 0
CAPTURE_WIDTH = 640#1280
CAPTURE_HEIGHT = 480#720
CAPTURE_FPS = 30
use_FLIP_X : bool = False
INPUT_QUEUE_MAX = 4      # Evita accumulo di latenza
OUTPUT_QUEUE_MAX = 4

SCREENSHOT_DIR = "./captures"

# ------------------------------------------------

@dataclass
# class WatershedParams:
    # blur_ksize: int = 3           # kernel gaussiano (dispari)
    # morph_ksize: int = 3          # kernel morfologico (dispari)
    # dist_ratio: float = 0.7       # soglia su distanza normalizzata [0..1]
    # use_clahe: bool = False       # migliora contrasto
    # remove_small: int = 50        # rimuove componenti piccole (px)
    # markers_on_edges: bool = True # colora i bordi del watershed

class Sobel:
    morph_ksize: int = 3          # kernel morfologico (dispari)
    dist_ratio: float = 0.7       # soglia su distanza normalizzata [0..1]
    use_clahe: bool = False       # migliora contrasto
    remove_small: int = 50        # rimuove componenti piccole (px)
    markers_on_edges: bool = True # colora i bordi del watershed
    blur_ksize: int = 3           # kernel gaussiano (dispari)    


# class PreProc:
    # use_Flip: bool = False
    
    


def odd(k: int) -> int:
    """Converte in valore dispari >= 1"""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1

# def watershed_process_loop(in_q: mp.Queue, out_q: mp.Queue, ctrl_q: mp.Queue):
    # """
    # Processo separato che esegue la segmentazione watershed.
    # Riceve (frame_bgr, params_dict, frame_id) e ritorna (vis_bgr, mask, markers, frame_id, timings).
    # ctrl_q: coda di controllo per shutdown.
    # """
    # def to_gray(bgr):
        # return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # def remove_small_components(binary, min_size):
        # if min_size <= 1:
            # return binary
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # mask = np.zeros_like(binary, dtype=np.uint8)
        # for i in range(1, num_labels):
            # if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # mask[labels == i] = 255
        # return mask

    # last_params = None
    # while True:
        # # Controllo shutdown non bloccante
        # try:
            # msg = ctrl_q.get_nowait()
            # if msg == "STOP":
                # break
        # except queue.Empty:
            # pass

        # try:
            # item = in_q.get(timeout=0.1)
        # except queue.Empty:
            # continue

        # frame_bgr, params_dict, frame_id, t_sent = item
        # t0 = time.time()
        # params = WatershedParams(**params_dict)

        # # Pre-processing
        # gray = to_gray(frame_bgr)

        # if params.use_clahe:
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # gray = clahe.apply(gray)

        # k_blur = odd(params.blur_ksize)
        # if k_blur > 1:
            # gray_blur = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
        # else:
            # gray_blur = gray

        # # Threshold + morfologia
        # _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # k_morph = odd(params.morph_ksize)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        # opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

        # # Rimozione componenti piccole
        # opening = remove_small_components(opening, params.remove_small)

        # # Sure background
        # sure_bg = cv2.dilate(opening, kernel, iterations=2)

        # # Distance Transform -> Sure foreground
        # dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # # Normalizza per sogliare in [0,1]
        # dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        # _, sure_fg = cv2.threshold(dist_norm, params.dist_ratio, 1.0, cv2.THRESH_BINARY)
        # sure_fg = (sure_fg * 255).astype(np.uint8)

        # # Unknown area
        # unknown = cv2.subtract(sure_bg, sure_fg)

        # # Marker labelling
        # n_markers, markers = cv2.connectedComponents(sure_fg)
        # markers = markers + 1  # background -> 1
        # markers[unknown == 255] = 0

        # # Watershed
        # markers_ws = markers.copy()
        # cv2.watershed(frame_bgr, markers_ws)

        # # Visualizzazione: colora segmenti
        # vis = frame_bgr.copy()
        # if params.markers_on_edges:
            # vis[markers_ws == -1] = (0, 0, 255)  # bordi in rosso

        # # Opzionale: pseudo-color dei cluster
        # # m > 1 sono oggetti
        # m = markers_ws.copy()
        # m[m <= 1] = 0
        # m_edges = (markers_ws == -1).astype(np.uint8) * 255

        # # Genera colormap casuale ma deterministica fino a 256 etichette
        # num_labels = int(m.max())
        # rng = np.random.default_rng(42)
        # palette = rng.integers(0, 255, size=(max(256, num_labels + 1), 3), dtype=np.uint8)
        # seg_rgb = palette[np.clip(m, 0, palette.shape[0]-1)]
        # seg_rgb[m == 0] = (0, 0, 0)

        # overlay = cv2.addWeighted(vis, 0.65, seg_rgb, 0.35, 0)
        # overlay[m_edges == 255] = (0, 0, 255)

        # t1 = time.time()
        # timings = {
            # "q_latency_ms": int((t0 - t_sent) * 1000),
            # "proc_ms": int((t1 - t0) * 1000),
            # "n_markers": int(num_labels),
        # }

        # # Ridimensiona per evitare trasferimenti enormi (facoltativo)
        # # (qui manteniamo la risoluzione originale)

        # # Invio risultato
        # try:
            # out_q.put_nowait((overlay, sure_fg, markers_ws.astype(np.int32), frame_id, timings))
        # except queue.Full:
            # # Scarta se UI non consuma in tempo (backpressure)
            # pass

    # # Uscita pulita
    # while not in_q.empty():
        # try:
            # in_q.get_nowait()
        # except Exception:
            # break
            
def sobel_process_loop(in_q: mp.Queue, out_q: mp.Queue, ctrl_q: mp.Queue):
    """
    Processo separato che esegue la segmentazione watershed.
    Riceve (frame_bgr, params_dict, frame_id) e ritorna (vis_bgr, mask, markers, frame_id, timings).
    ctrl_q: coda di controllo per shutdown.
    """
    def to_gray(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # def remove_small_components(binary, min_size):
        # if min_size <= 1:
            # return binary
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # mask = np.zeros_like(binary, dtype=np.uint8)
        # for i in range(1, num_labels):
            # if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # mask[labels == i] = 255
        # return mask

    # last_params = None
    while True:
        # Controllo shutdown non bloccante
        try:
            msg = ctrl_q.get_nowait()
            if msg == "STOP":
                break
        except queue.Empty:
            pass

        try:
            item = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_bgr, params_dict, frame_id, t_sent = item
        t0 = time.time()
        params = Sobel(**params_dict)

        # Pre-processing
        gray = to_gray(frame_bgr)
        
        #Reduce noise / Smoothen the image
        #Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        # blurred = cv.GaussianBlur(frame,(3,3),0)
        # #Grayscale convert
        # gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    
        # compute gradients along the x and y axis, respectively

        #****CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if params.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        k_blur = odd(params.blur_ksize)
        if k_blur > 1:
            gray_blur = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
        else:
            gray_blur = gray
        
        #*****SOBEL METODO 1
        sobel_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(255 * sobel / np.max(sobel))
        contours_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        contours_img[sobel > 50] = (0, 255, 0)
        #*****SOBEL METODO 2
        # gX = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0)
        # gY = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1)   
        # # the gradient magnitude images are now of the floating point data
        # # type, so we need to take care to convert them back a to unsigned
        # # 8-bit integer representation so other OpenCV functions can operate
        # # on them and visualize them
        # gX = cv2.convertScaleAbs(gX)
        # gY = cv2.convertScaleAbs(gY)
        # # combine the gradient representations into a single image
        # combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

        t1 = time.time()
        timings = {
            "q_latency_ms": int((t0 - t_sent) * 1000),
            "proc_ms": int((t1 - t0) * 1000)
            #"n_markers": int(num_labels),
        }

        # Ridimensiona per evitare trasferimenti enormi (facoltativo)
        # (qui manteniamo la risoluzione originale)

        # Invio risultato
        try:
            out_q.put_nowait((contours_img, frame_id, timings))#(overlay, sure_fg, markers_ws.astype(np.int32)
        except queue.Full:
            # Scarta se UI non consuma in tempo (backpressure)
            pass

    # Uscita pulita
    while not in_q.empty():
        try:
            in_q.get_nowait()
        except Exception:
            break

class CameraCapture(threading.Thread):
    """Thread affine per acquisizione affidabile da cv2.VideoCapture"""
    def __init__(self, index=0, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, stop_event: threading.Event= None):
        super().__init__(daemon=True)
        self.index = index
        self.width = width
        self.height = height
        self.flip_cam = False
        self.fps = fps
        self.stop_event = stop_event or threading.Event()
        self.cap = None
        self.last_frame = None
        self.ok = False
        self._lock = threading.Lock()
        self._paused = False
        #frame = cv2.flip(frame,0)
    
    def run(self):
        
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        # Settaggi (best-effort)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.flags = False

        # Warmup
        time.sleep(0.2)

        while not self.stop_event.is_set():
            if self._paused:
                time.sleep(0.05)
                continue
#flip code: A flag to specify how to flip the array; 
#   0 means flipping around the x-axis and positive value 
#   1 means flipping around y-axis. 
#   Negative value (for example, -1) means flipping around both axes
                

            ok, frame = self.cap.read()
            #flags = PreProc(**params_dict)
            if use_FLIP_X :
                frame = cv2.flip(frame,1)

            
            if not ok or frame is None:
                # tenta recovery
                time.sleep(0.05)
                continue

            with self._lock:
                self.last_frame = frame
                self.ok = True

        # cleanup
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self.ok, None if self.last_frame is None else self.last_frame.copy()

    def pause(self, flag: bool):
        self._paused = flag


class App:
    def __init__(self, root):
        self.root = root
        root.title("Watershed Realtime – Threaded Capture + Multiprocessing")
        try:
            root.iconbitmap(default="myIco.ico")
        except Exception:
            pass

        # State
        self.stop_event = threading.Event()
        self.capture = CameraCapture(index=DEFAULT_CAMERA_INDEX, stop_event=self.stop_event)
        self.capture.start()

        self.input_q = mp.Queue(maxsize=INPUT_QUEUE_MAX)
        self.output_q = mp.Queue(maxsize=OUTPUT_QUEUE_MAX)
        self.ctrl_q = mp.Queue(maxsize=2)

        self.proc = mp.Process(target=sobel_process_loop, args=(self.input_q, self.output_q, self.ctrl_q), daemon=True)
        self.proc.start()

        # FPS & Latency
        self.fps_deque = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_id = 0
        self.paused = False

        # Params
        self.params = Sobel()  #WatershedParams()
        #self.flag = False
        self._build_ui(root)

        # Loop di aggiornamento UI
        self.update_loop()

        # Key bindings
        root.bind("<q>", lambda e: self.on_close())
        root.bind("<Escape>", lambda e: self.on_close())
        root.bind("<space>", self.toggle_pause)
        root.bind("<s>", self.save_screenshot)

    def _build_ui(self, root):
        # Layout
        main = ttk.Frame(root, padding=6)
        main.pack(fill="both", expand=True)

        # Left: video
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew")

        # Right: controls
        right = ttk.LabelFrame(main, text="Parametri", padding=6)
        right.grid(row=0, column=1, sticky="nsew", padx=(8,0))

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        # Canvas per immagine
        self.canvas = tk.Label(left)
        self.canvas.pack(fill="both", expand=True)

        # Info bar
        self.info_var = tk.StringVar(value="Inizializzazione...")
        info_label = ttk.Label(left, textvariable=self.info_var, anchor="w")
        info_label.pack(fill="x", pady=(4,0))

        # Controls
        def add_slider(parent, text, from_, to_, init, command, step=1, row=None):
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=3)
            label = ttk.Label(frame, text=text, width=20)
            label.pack(side="left")
            var = tk.DoubleVar(value=init)
            scale = ttk.Scale(frame, from_=from_, to=to_, orient="horizontal", variable=var, command=lambda v: command(var))
            scale.pack(side="left", fill="x", expand=True, padx=6)
            val_label = ttk.Label(frame, text=f"{init}")
            val_label.pack(side="right")
            def _update_label(v):
                val = var.get()
                if step >= 1:
                    val = int(round(val / step) * step)
                val_label.config(text=f"{val}")
                command(var)
            var.trace_add("write", lambda *args: _update_label(var))
            return var

        # Blur ksize
        self.var_blur = add_slider(right, "Blur ksize (odd)", 1, 15, self.params.blur_ksize, self.on_params_changed, step=1)
        # Morph ksize
        self.var_morph = add_slider(right, "Morph ksize (odd)", 1, 21, self.params.morph_ksize, self.on_params_changed, step=1)
        # Distance ratio
        self.var_dist = add_slider(right, "Distance ratio", 0.1, 0.9, self.params.dist_ratio, self.on_params_changed, step=0.01)

        # Remove small
        frame_rs = ttk.Frame(right)
        frame_rs.pack(fill="x", pady=3)
        ttk.Label(frame_rs, text="Remove small (px)", width=20).pack(side="left")
        self.var_remove_small = tk.IntVar(value=self.params.remove_small)
        spin_rs = ttk.Spinbox(frame_rs, from_=0, to=1000, textvariable=self.var_remove_small, width=7, command=self.on_params_changed)
        spin_rs.pack(side="left", padx=6)
        
        # FLIP
        self.var_flip = tk.BooleanVar(value=use_FLIP_X)
        chk_flip = ttk.Checkbutton(right, text="Flip X", variable=self.var_flip, command=self.on_params_changed)
        chk_flip.pack(anchor="w", pady=3)

        # CLAHE
        self.var_clahe = tk.BooleanVar(value=self.params.use_clahe)
        chk_clahe = ttk.Checkbutton(right, text="Use CLAHE", variable=self.var_clahe, command=self.on_params_changed)
        chk_clahe.pack(anchor="w", pady=3)
        
        

        # Markers edges
        self.var_edges = tk.BooleanVar(value=self.params.markers_on_edges)
        chk_edges = ttk.Checkbutton(right, text="Show edges", variable=self.var_edges, command=self.on_params_changed)
        chk_edges.pack(anchor="w", pady=3)

        # Buttons
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill="x", pady=(8,0))
        ttk.Button(btn_frame, text="Screenshot (s)", command=self.save_screenshot).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Pausa/Resume (space)", command=lambda: self.toggle_pause(None)).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Esci (q)", command=self.on_close).pack(side="left", padx=2)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(right, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(8,0))

        # Tk styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_params_changed(self, *_):
        self.params.blur_ksize = int(self.var_blur.get())
        # self.params.morph_ksize = int(self.var_morph.get())
        # self.params.dist_ratio = float(self.var_dist.get())
        self.params.use_clahe = bool(self.var_clahe.get())
        use_FLIP_X = bool(self.var_flip.get())
        # self.params.markers_on_edges = bool(self.var_edges.get())
        # self.params.remove_small = int(self.var_remove_small.get())

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.capture.pause(self.paused)
        self.status_var.set("Pausa" if self.paused else "In esecuzione")

    def save_screenshot(self, event=None):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        # Prova a prendere l'ultima immagine mostrata
        if hasattr(self, "_last_vis"):
            path = os.path.join(SCREENSHOT_DIR, f"ws_{ts}.png")
            cv2.imwrite(path, self._last_vis)
            self.status_var.set(f"Screenshot salvato: {path}")
        else:
            self.status_var.set("Nessun frame disponibile per screenshot.")

    def update_loop(self):
        # 1) ottieni ultimo frame dal thread di cattura
        ok, frame_bgr = self.capture.get_latest()
        if ok and not self.paused:
            # 2) invia al processo di elaborazione (se possibile)
            params_dict = self.params.__dict__.copy()
            try:
                self.input_q.put_nowait((frame_bgr, params_dict, self.frame_id, time.time()))
                self.frame_id += 1
            except queue.Full:
                # se piena, scarta per non accumulare ritardo
                pass

        # 3) recupera risultati dal processo
        try:
            for _ in range(2):  # consuma al massimo 2 risultati per ciclo UI
                #vis_bgr, sure_fg, markers_ws, fid, timings = self.output_q.get_nowait()
                vis_bgr, fid, timings = self.output_q.get_nowait()
                
                self._last_vis = vis_bgr  # per screenshot
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(vis_rgb)
                imtk = ImageTk.PhotoImage(image=im)
                self.canvas.configure(image=imtk)
                self.canvas.image = imtk  # reference

                # FPS
                now = time.time()
                self.fps_deque.append(1.0 / max(1e-6, (now - self.last_time)))
                self.last_time = now
                fps = sum(self.fps_deque) / max(1, len(self.fps_deque))

                self.info_var.set(f"Frame {fid} | FPS UI: {fps:4.1f} | Proc: {timings['proc_ms']} ms | LatencyQ: {timings['q_latency_ms']}")
                #self.info_var.set(f"Frame {fid} | FPS UI: {fps:4.1f} | Proc: {timings['proc_ms']} ms | LatencyQ: {timings['q_latency_ms']} ms | Markers: {timings['n_markers']}")
        
        except queue.Empty:
            pass

        # 4) ripianifica
        self.root.after(50, self.update_loop)

    def on_close(self):
        try:
            self.status_var.set("Chiusura in corso...")
        except Exception:
            pass
        
        self.stop_event.set()
        
        try:
            self.ctrl_q.put_nowait("STOP")
        except Exception:
            pass

        # attende thread e processo
        try:
            self.capture.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.proc.is_alive():
                self.proc.join(timeout=1.0)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass

        self.root.after(10, self.root.destroy)
        self.root.destroy()


def main():
    mp.set_start_method("spawn", force=True)
    root = tk.Tk()
    app = App(root)
    root.geometry("1200x700")
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
