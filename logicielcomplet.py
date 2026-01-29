# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 11:07:43 2026

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:23:33 2025

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:21:36 2025

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
TEM Fringe Analyzer — UI claire & moderne (version finale)
- Thème clair (stylesheet modernisé)
- Matplotlib style custom (axes sobres + cohérent avec UI)
- Gabor/énergie calculée mais non affichée (Step 5 masqué)
- Step 6: 2 grandes images (IFFT / Franges détectées)
- Ctrl+S: export des images (FFT, Masque, IFFT, Franges détectées)
"""

import sys, os, csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from skimage.morphology import skeletonize
from skimage import img_as_bool


# 🎨 Matplotlib modernisé
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#AAAAAA",
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "legend.frameon": False,
    "legend.fontsize": 8
})

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QCheckBox,
    QDockWidget, QTextEdit, QLineEdit, QFormLayout,
    QMessageBox, QGroupBox, QRadioButton, QStackedWidget,QPushButton,
    QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLineEdit, QFormLayout, QFileDialog,
    QSlider, QComboBox, QSpinBox, QSizePolicy
)


from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Circle, Rectangle

from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# ================================
# Constants
# ================================
MIN_ZONE_AREA = 300
ENERGY_PCTL   = 80
KSIZE_GABOR   = 31
SIGMA_GABOR   = 7
GAMMA_GABOR   = 0.5
SMOOTH_ALPHA  = 0.30
LIVE_SCALE    = 0.6
DEFAULT_SCALE_NM = 20.0
TIMER_MS      = 30

# ================================
# Utilities
# ================================
def line_profile(img, pA, pB, n=300):
    xs = np.linspace(pA[0], pB[0], n)
    ys = np.linspace(pA[1], pB[1], n)
    coords = np.vstack((ys, xs))
    return map_coordinates(img, coords, order=1, mode='nearest').astype(np.float32)

def make_wedge_mask(rows, cols, crow, ccol, angle_center, angle_opening):
    Y, X = np.ogrid[:rows, :cols]
    dy, dx = Y - crow, X - ccol
    angles = np.degrees(np.arctan2(dy, dx))
    mask = np.zeros((rows, cols), np.uint8)
    delta = ((angles - angle_center + 180) % 360) - 180
    mask[np.abs(delta) < angle_opening/2] = 1
    delta_op = ((angles - (angle_center+180) + 180) % 360) - 180
    mask[np.abs(delta_op) < angle_opening/2] = 1
    R2 = (Y - crow)**2 + (X - ccol)**2
    mask[R2 <= 3**2] = 0
    return mask

def orientation_pca(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2: return 0.0
    _, eigenvecs = cv2.PCACompute(pts, mean=np.array([]))
    vx, vy = eigenvecs[0]
    return np.arctan2(vy, vx)


def skeletonize_roi(roi_gray: np.ndarray):
    """
    Méthode spatiale indépendante FFT :
    binarisation + nettoyage + squelettisation
    """
    if roi_gray is None:
        raise ValueError("ROI absente")
    if roi_gray.ndim != 2:
        raise ValueError("ROI doit être en niveaux de gris")

    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 2
    )
    th = 255 - th

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    skel_bool = skeletonize(img_as_bool(th))
    skel = (skel_bool * 255).astype(np.uint8)

    return th, skel



# ================================
# App State
# ================================
class State:
    def __init__(self):
        self.img_path = None
        self.img_color = None
        self.img = None
        self.img_full = None
        self.scale_points = []
        self.scale_percent = 20
        self.nm_per_px = None
        self.roi_rect = None
        self.roi = None
        self.live_enabled = True
        self.live_scale = LIVE_SCALE
        self.roi_live = None
        self.live_factor = 1.0
        self.angle = 0.0
        self.angle_target = 0.0
        self.opening = 10.0
        self.r_limit = None
        self.r_inner = None
        self.square_half = None
        self.f_full = None
        self.fshift_full = None
        self.fft_mag_full = None
        self.f_live = None
        self.fshift_live = None
        self.fft_mag_live = None
        self.mask = None
        self.img_back = None
        self.lambda_px = None
        self.lambda_nm = None
        self.energy = None
        self.energy_mask = None
        self.results = []
        self.out_vis = None
      
               

# ================================
# Main Window
# ================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 TEM Fringe Analyzer — Interface claire")
        self.resize(1640, 980)
        self.S = State()

        self.lambda_min = 0.35
        self.lambda_max = 0.65

        # 🌞 Thème clair modernisé
      
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
                color: #1e293b;
                font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
            }
        
            QLabel {
                color: #0f172a;
                font-size: 14px;
                font-weight: 600;
            }
        
            QPushButton {
                background-color: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 6px 12px;
                color: #1e293b;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #cbd5e1; }
            QPushButton:pressed { background-color: #94a3b8; color: white; }
        
            QSlider::groove:horizontal {
                background: #e2e8f0;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                width: 16px;
                border-radius: 8px;
                margin: -4px 0;
            }
        
            QDockWidget::title {
                background: #f1f5f9;
                padding-left: 10px;
                font-weight: bold;
            }
        
            QGroupBox {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: #ffffff;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                font-weight: 700;
            }

            
        
            QTextEdit {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                color: #334155;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)

        
        

        # UI containers
        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)
        self.console = QTextEdit(); self.console.setReadOnly(True)
        dock_console = QDockWidget("Console / Résultats", self)
        dock_console.setWidget(self.console); self.addDockWidget(Qt.BottomDockWidgetArea, dock_console)

        # Build steps
        self.page0 = self.build_step0_scale()
        self.page1 = self.build_step1_roi()
        self.page2 = self.build_step2_fft()
        self.page3 = self.build_step3_cropping()
        self.page4 = self.build_step4_ifft_lambda()
        # step5 n'affiche rien (compute_energy en coulisse)
        self.page6 = self.build_step6_detect()
        self.page7 = self.build_step7_plots()
        for p in [self.page0,self.page1,self.page2,self.page3,self.page4,self.page6,self.page7]:
            self.stack.addWidget(p)
            
        self.page_alt = self.build_alt_skeleton()
        self.stack.addWidget(self.page_alt)


        # Controls dock
        self.build_controls_dock()
        self.console.append("💡 Étape 0: charge une image ----> Définition de l'echelle ----> Entrée=valider.")

        # Angle smoothing timer
        self.tmr = QTimer(self); self.tmr.setInterval(TIMER_MS)
        self.tmr.timeout.connect(self.tick); self.tmr.start()

    # ----------- Helpers -----------
    def _warn(self, m):
        QMessageBox.warning(self, "Attention", m)
        self.console.append(f"⚠️ {m}")
    def _info(self, m):
        self.console.append(m)

    # ----------- Full reset -----------
    def reset_everything(self):
        if hasattr(self, 'rs') and self.rs is not None:
            try: self.rs.set_active(False)
            except Exception: pass
            self.rs = None

        self.S = State()

        for fig_attr, canvas_attr in [
            ('fig0','canvas0'),
            ('fig1','canvas1'),
            ('fig2','canvas2'),
            ('fig3','canvas3'),
            ('fig4','canvas4'),
            ('fig6','canvas6'),
            ('fig7a','canvas7a'),
            ('fig7b','canvas7b'),
        ]:
            fig = getattr(self, fig_attr, None)
            canvas = getattr(self, canvas_attr, None)
            if fig is not None:
                for ax in fig.axes: ax.cla()
            if canvas is not None:
                canvas.draw_idle()
                if canvas_attr not in ('canvas0','canvas1'):
                    canvas.setVisible(False)

        if hasattr(self,'grp_wedge'): self.grp_wedge.setEnabled(False)
        if hasattr(self,'sld_angle'): self.sld_angle.setValue(0)
        if hasattr(self,'sld_open'): self.sld_open.setValue(10)
        if hasattr(self,'chk_live'): self.chk_live.setChecked(True)
        if hasattr(self,'btn_roi_mode'): self.btn_roi_mode.setChecked(False)

        self.stack.setCurrentIndex(0)
        self.console.clear()
        self.console.append("♻️ Tout réinitialisé (nouvelle image)")
        self.console.append("💡 Étape 0: charge une image ----> Définition de l'echelle ----> Entrée=valider.")

    # ----------- Dock (nav + global controls) -----------
    def build_controls_dock(self):
        dock = QDockWidget("⚙️ Contrôles", self)
        w = QWidget(); v = QVBoxLayout(w)
       
        grp_nav = QGroupBox("📂 Chargement & navigation")
        grp_nav.setStyleSheet("""
        QGroupBox {
            border: 2px solid #0284c7;
            border-radius: 12px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            color: #0284c7;
            font-weight: 700;
        }
        """)
        
        vnav = QVBoxLayout(grp_nav)
        
        btn_load = QPushButton("📂 Charger image…")
        btn_load.clicked.connect(self.reload_from_dock)  
        vnav.addWidget(btn_load)
        
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← Précédent")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_next = QPushButton("Suivant →")
        self.btn_next.clicked.connect(self.next_step)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        
        vnav.addLayout(nav)
        v.addWidget(grp_nav)  # ✅ une seule fois


       
    
        # Live preview controls
        grp_live = QGroupBox("⚡ Aperçu live")
        vlive = QVBoxLayout(grp_live)
        
        grp_live.setStyleSheet(
            "QGroupBox::title { color: #0284c7; font-weight: 700; }"
        )

        
        self.chk_live = QCheckBox("Aperçu live (rapide)")
        self.chk_live.setChecked(True)
        self.chk_live.stateChanged.connect(self.on_toggle_live)
        vlive.addWidget(self.chk_live)
        
        vlive.addWidget(QLabel("Échelle d’aperçu (0.3 – 1.0)"))
        self.sld_live = QSlider(Qt.Horizontal)
        self.sld_live.setRange(30, 100)
        self.sld_live.setValue(int(LIVE_SCALE * 100))
        self.sld_live.valueChanged.connect(self.on_live_scale_change)
        vlive.addWidget(self.sld_live)
        
        v.addWidget(grp_live)

    
        # Wedge controls (steps 2–3)
        self.grp_wedge = QGroupBox("🧭 Wedge FFT — Étapes 2–3")
        g = QVBoxLayout(self.grp_wedge)
        
        self.grp_wedge.setStyleSheet("""
        QGroupBox {
            border: 2px solid #1d4ed8;
            border-radius: 12px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            color: #1d4ed8;
            font-weight: 700;
        }
        """)


        
        g.addWidget(QLabel("Orientation (°)"))
        self.sld_angle = QSlider(Qt.Horizontal)
        self.sld_angle.setRange(-180, 180)
        self.sld_angle.valueChanged.connect(lambda v: self.set_angle_target(float(v)))
        g.addWidget(self.sld_angle)
        
        g.addWidget(QLabel("Ouverture (°) — min 5°"))
        self.sld_open = QSlider(Qt.Horizontal)
        self.sld_open.setRange(5, 180)
        self.sld_open.valueChanged.connect(self.on_open_change)
        g.addWidget(self.sld_open)
        
        self.grp_wedge.setEnabled(False)
        v.addWidget(self.grp_wedge)

        
        # Seuil d’énergie (percentile)
        
        v.addSpacing(20)  # espace vertical avant le bloc (plus aéré)

        # --- Seuil d’énergie (percentile Gabor) ---
        grp_gabor = QGroupBox("🔬 Analyse Gabor — filtrage d₀₀₂")
        vg = QVBoxLayout(grp_gabor)
        
        grp_gabor.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4338ca;
                border-radius: 12px;
            }
            QGroupBox::title {
                color: #4338ca;
                font-weight: 700;
            }
            """)

        

        # --- Slider énergie (CRÉATION UNIQUE) ---
        # --- Analyse Gabor ---
        vg.addWidget(QLabel("Seuil d’énergie (percentile Gabor)"))
        
        # Slider énergie (CRÉATION UNIQUE)
        self.sld_energy = QSlider(Qt.Horizontal)
        self.sld_energy.setRange(60, 95)
        self.sld_energy.setValue(ENERGY_PCTL)
        self.sld_energy.valueChanged.connect(self.on_energy_percentile_change)
        vg.addWidget(self.sld_energy)
        
        self.lbl_energy_val = QLabel(f"Valeur actuelle : {ENERGY_PCTL}%")
        self.lbl_energy_val.setStyleSheet(
                    "color: #dc2626; font-weight: 700;"
                                                        )
        vg.addWidget(self.lbl_energy_val)
        
        vg.addSpacing(8)
        vg.addWidget(QLabel("Filtrage d₀₀₂ (nm) — distance entre franges"))
        
        # --- Ligne Min / Max (DANS le groupe Gabor) ---
        row_lambda = QHBoxLayout()
        
        lbl_min = QLabel("Min :")
        lbl_max = QLabel("Max :")
        lbl_min.setStyleSheet("color: #1e3a8a; font-weight: 600;")
        lbl_max.setStyleSheet("color: #1e3a8a; font-weight: 600;")
        
        self.edt_lambda_min = QLineEdit("0.35")
        self.edt_lambda_max = QLineEdit("0.65")
        self.edt_lambda_min.setMaximumWidth(60)
        self.edt_lambda_max.setMaximumWidth(60)
        
        row_lambda.addWidget(lbl_min)
        row_lambda.addWidget(self.edt_lambda_min)
        row_lambda.addSpacing(6)
        row_lambda.addWidget(lbl_max)
        row_lambda.addWidget(self.edt_lambda_max)
        
        vg.addLayout(row_lambda)
        
        v.addWidget(grp_gabor)

 
    
        # --- Export global (CSV + images) ---
        
        v.addSpacing(20)  # espace vertical de 10 px
        #v.addWidget(QLabel("Filtrage d_002 (nm) [distance entre franges pour chaque empilement]"))
   
        exp = QHBoxLayout()
        btn_export_all = QPushButton("💾 Exporter résultats & images")
        btn_export_all.clicked.connect(self.export_all)
        btn_to_plots = QPushButton("Voir courbes & histogrammes")
        btn_to_plots.clicked.connect(lambda: self.stack.setCurrentWidget(self.page7))
        exp.addWidget(btn_export_all); exp.addWidget(btn_to_plots); v.addLayout(exp)
        
        btn_open_detect = QPushButton("🖼️ Ouvrir image détection")
        btn_open_detect.clicked.connect(self.open_detection_image)
        v.addWidget(btn_open_detect)

        v.addSpacing(20)
        btn_alt = QPushButton("🧬 Squelettisation (ROI)")
        btn_alt.clicked.connect(self.go_to_skeleton_tab)
        v.addWidget(btn_alt)
        
                
    
        v.addStretch(1)
        dock.setWidget(w); self.addDockWidget(Qt.LeftDockWidgetArea, dock)
    
    
    def reload_from_dock(self):
        """Recharge une image depuis le dock (haut à gauche) avec recentrage automatique."""
        from PyQt5.QtCore import QTimer
        import matplotlib.pyplot as plt
        plt.close('all')
    
        # 🧭 Revenir à l’étape 0
        self.stack.setCurrentWidget(self.page0)
        self.reset_everything()
    
        # 🖼️ Charger l’image
        self.load_image()
    
        # 🪄 Recentrage différé automatique
        def finalize_display():
            try:
                if self.S.img_full is not None:
                    h, w = self.S.img_full.shape[:2]
                    self.ax0.set_xlim(0, w)
                    self.ax0.set_ylim(h, 0)
                    self.ax0.set_aspect('equal', adjustable='box')
                    self.fig0.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                    self.canvas0.draw_idle()
            except Exception as e:
                print("⚠️ finalize_display reload:", e)
    
        QTimer.singleShot(150, finalize_display)
        QTimer.singleShot(600, finalize_display)
            
    def open_detection_image(self):
        """🖼️ Ouvre l'image de détection la plus récente avec le logiciel par défaut du système (Windows/macOS/Linux)."""
        import platform
        import subprocess
        import cv2
    
        if self.S.img_path is None:
            self._warn("Aucune image chargée.")
            return
    
        # Dossier d’export de l’image courante
        base = os.path.splitext(os.path.basename(self.S.img_path))[0]
        folder = os.path.join(os.path.dirname(self.S.img_path), f"{base}_exports")
        img_path = os.path.join(folder, f"{base}_step6_ifft_detect.png")
    
        # Vérifie que la dernière détection existe
        # Vérifie que la dernière image de détection existe ou peut être rechargée
        if not os.path.exists(img_path):
            self._warn("Aucune image de détection trouvée. Lance d’abord l’étape 6 ou fais un export.")
            return
        
        # ✅ Recharge l’image PNG exportée (toujours à jour)
        img = cv2.imread(img_path)
        if img is None:
            self._warn("Impossible de charger l’image de détection.")
            return
        
        # 🔁 Écrase le fichier pour être sûr que Windows recharge bien (évite le cache)
        cv2.imwrite(img_path, img)

    
        import tempfile, shutil
        
        try:
            system = platform.system()
        
            # 🔁 Crée une copie temporaire pour éviter le cache Windows
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            shutil.copy2(img_path, tmp.name)
        
            # Ouvre cette copie temporaire
            if system == "Windows":
                os.startfile(tmp.name)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", tmp.name])
            else:  # Linux
                subprocess.run(["xdg-open", tmp.name])
        
            self._info(f"🖼️ Copie temporaire ouverte (rafraîchie) : {tmp.name}")
        
        except Exception as e:
            self._warn(f"⚠️ Impossible d’ouvrir l’image : {e}")

    
    def on_energy_percentile_change(self, val):
        """Met à jour le percentile pour la carte d’énergie."""
        self.lbl_energy_val.setText(f"Actuel : {val}%")
        self.S.energy_percentile = val
        # Recalcule si une IFFT existe
        if self.S.img_back is not None:
            self.compute_energy(auto=False)
            self._info(f"⚙️ Énergie recalculée avec percentile = {val}%")
    


    def export_all(self):
        """Exporte les résultats CSV et toutes les images."""
        self._info("🚀 Export global lancé...")
        self.export_csv()
        self.export_images()
        self._info("✅ Export global terminé.")

    def prev_step(self):
        i = self.stack.currentIndex()
        if i>0: self.stack.setCurrentIndex(i-1)

    def next_step(self):
        i = self.stack.currentIndex()
        if i<self.stack.count()-1:
            if i==0 and self.S.nm_per_px is None: self._warn("Mesure d'échelle non validée."); return
            if i==1 and self.S.roi is None: self._warn("ROI non validée."); return
            self.stack.setCurrentIndex(i+1)
    
    def build_step0_scale(self):
        page = QWidget()
        v = QVBoxLayout(page)
    
        # --- Figure d'affichage (pour montrer l'image après chargement) ---
        self.fig0 = Figure(figsize=(6, 6))
        self.ax0 = self.fig0.add_subplot(111)
        self.canvas0 = FigureCanvas(self.fig0)
        v.addWidget(self.canvas0)
    
        # --- Bouton "Charger une image" (affiché avant chargement) ---
        self.btn_load_img = QPushButton("📂 Charger une image")
        self.btn_load_img.setFixedSize(200, 60)
        self.btn_load_img.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                background-color: #0078d7;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3399ff;
            }
        """)
        self.btn_load_img.clicked.connect(self.load_image)
        v.addWidget(self.btn_load_img, alignment=Qt.AlignCenter)
    
        # --- Texte explicatif (en dessous de l'image, caché au départ) ---
        self.lbl0 = QLabel(
            "🧭 <b>Étape 0 — Définition de l’échelle</b><br><br>"
            "Après avoir chargé l'image :<br>"
            "👉 Cliquer sur les deux extrémités de la barre d’échelle :<br>"
            "• Premier clic = début (min)<br>"
            "• Deuxième clic = fin (max)<br><br>"
            "➡️ Entrer ensuite la valeur réelle de la barre (en nm).<br>"
            "➡️ Appuier sur <b>Entrée</b> ou cliquer sur <b>« Valider l’échelle »</b> pour confirmer."
        )
        self.lbl0.setStyleSheet("""
            QLabel {
                color: #1e293b;
                font-size: 13px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        self.lbl0.setAlignment(Qt.AlignCenter)
        self.lbl0.hide()  # 🔸 caché tant qu'on n'a pas chargé d'image
        v.addWidget(self.lbl0)
    
        # --- Champ et boutons pour l’échelle (cachés au départ) ---
        form = QFormLayout()
        self.edt_scale_nm = QLineEdit()
        self.edt_scale_nm.setPlaceholderText("ex: 20 (défaut)")
        form.addRow("Valeur barre (nm)", self.edt_scale_nm)
        self.frm_scale = QWidget()
        self.frm_scale.setLayout(form)
        self.frm_scale.hide()
        v.addWidget(self.frm_scale)
    
        row = QHBoxLayout()
        self.btn_reset_pts = QPushButton("Réinitialiser points")
        self.btn_reset_pts.clicked.connect(self.reset_scale_points)
        self.btn_scale_ok = QPushButton("Valider l'échelle")
        self.btn_scale_ok.clicked.connect(self.confirm_scale)
        row.addWidget(self.btn_reset_pts)
        row.addWidget(self.btn_scale_ok)
        self.row_scale = QWidget()
        self.row_scale.setLayout(row)
        self.row_scale.hide()
        v.addWidget(self.row_scale)
    
        # --- Connexion du clic sur la figure ---
        self.canvas0.mpl_connect('button_press_event', self.on_scale_click)
    
        return page

    def load_image(self):
        """Charge une nouvelle image et revient proprement à l’étape 0, centrée et bien cadrée."""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image",
            "", "Images (*.png *.jpg *.tif *.tiff *.dm3 *.dm4)"
        )
        if not fname:
            return
    
        # 🔁 Réinitialiser l’état complet
        self.reset_everything()
    
        imgc = cv2.imread(fname)
        if imgc is None:
            self._warn("Impossible de charger l'image.")
            return
    
        gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    
        # Sauvegarde état
        self.S.img_path = fname
        self.S.img_full = gray.copy()
        self.S.img = gray.copy()
        self.S.img_color = imgc.copy()
    
        h, w = gray.shape[:2]
        # 🧭 Forcer retour à l’étape 0 (utile si l’utilisateur recharge depuis une autre étape)
        self.stack.setCurrentWidget(self.page0)

        self._info(f"✅ Image chargée : {os.path.basename(fname)} | {w}x{h} px")
    
        # --- Préparer affichage étape 0 ---
        #self.stack.setCurrentWidget(self.page0)
    
        # 🔧 Forcer Qt à stabiliser les layouts avant le dessin
        self.centralWidget().layout().activate()
        QApplication.processEvents()
    
        # --- Affichage image dans l’axe matplotlib ---
        self.ax0.clear()
        self.ax0.imshow(self.S.img_full, cmap="gray", origin="upper")
        self.ax0.set_title("Clique sur la barre d’échelle", fontsize=10)
        self.ax0.axis("off")
        self.ax0.set_aspect("equal", adjustable="box")
        self.ax0.set_xlim(0, w)
        self.ax0.set_ylim(h, 0)
        self.fig0.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    
        # 🔁 Premier affichage immédiat
        self.canvas0.draw_idle()
    
        # 🪄 Correction de centrage différée (Qt met parfois 100–300 ms à stabiliser)
        from PyQt5.QtCore import QTimer
    
        def finalize_display():
            try:
                if self.stack.currentWidget() is self.page0:
                    self.ax0.set_xlim(0, w)
                    self.ax0.set_ylim(h, 0)
                    self.ax0.set_aspect("equal", adjustable="box")
                    self.fig0.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                    self.canvas0.draw_idle()
            except Exception as e:
                print("⚠️ finalize_display:", e)
    
        QTimer.singleShot(150, finalize_display)
        QTimer.singleShot(600, finalize_display)
    
        # 🧹 Nettoyage des points d’échelle
        self.S.scale_points = []
        # ✅ Afficher les éléments de l’étape 0 (si masqués)
        if hasattr(self, "btn_load_img"): 
            self.btn_load_img.hide()  # on cache le bouton central après le chargement
        if hasattr(self, "lbl0"): 
            self.lbl0.show()
        if hasattr(self, "frm_scale"): 
            self.frm_scale.show()
        if hasattr(self, "row_scale"): 
            self.row_scale.show()
        if hasattr(self, "edt_scale_nm"): 
            self.edt_scale_nm.show()
        if hasattr(self, "btn_scale_ok"): 
            self.btn_scale_ok.show()
        if hasattr(self, "btn_reset_pts"): 
            self.btn_reset_pts.show()

        self._info("💡 Étape 0 : charge une image ----> Définition de l’échelle ----> Entrée = valider.")

    def show_scale_preview(self):
        """Affiche une version redimensionnée de l'image pour la calibration (échelle)."""
        imgc = self.S.img_color
        if imgc is None:
            return
    
        h, w = imgc.shape[:2]
        sp = self.S.scale_percent
        nw = int(w * sp / 100)
        nh = int(h * sp / 100)
    
        disp = cv2.resize(imgc, (nw, nh), interpolation=cv2.INTER_AREA)
        self._scale_preview = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    
        self.ax0.clear()
        self.ax0.imshow(self._scale_preview)
        self.ax0.axis('off')
        self.ax0.set_title("Clique sur la barre d’échelle", fontsize=10)
    
        # 🔍 Agrandit un peu la figure
        self.fig0.set_size_inches(7, 7)
        self.fig0.tight_layout(pad=0.5)
        self.canvas0.draw_idle()
    
        self.S.scale_points = []


    def reset_scale_points(self):
        """Réinitialise les points de la barre d’échelle."""
        self.S.scale_points = []
        self.redraw_scale_points()
        self._info("♻️ Points réinitialisés.")


    def on_scale_click(self, event):
        """Récupère les clics sur l’image réelle pour définir la barre d’échelle."""
        if event.inaxes != self.ax0:
            return
        if not hasattr(self.S, "scale_points"):
            self.S.scale_points = []
    
        if event.xdata is not None and event.ydata is not None:
            self.S.scale_points.append((event.xdata, event.ydata))
            self._info(f"🟢 Point {len(self.S.scale_points)} ajouté: ({event.xdata:.1f}, {event.ydata:.1f})")
            self.redraw_scale_points()


    
    
    def redraw_scale_points(self):
        """Redessine l'image avec les points de l'échelle si présents."""
        if getattr(self.S, "img", None) is None:
            return
    
        self.ax0.clear()
        self.ax0.imshow(self.S.img, cmap='gray')
        self.ax0.axis('off')
    
        if getattr(self.S, "scale_points", None):
            pts = np.array(self.S.scale_points)
            self.ax0.plot(pts[:, 0], pts[:, 1], 'ro', ms=6)
            if len(pts) == 2:
                self.ax0.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], 'r-', lw=2, alpha=0.8)
    
        self.canvas0.draw_idle()
    
    
    def confirm_scale(self):
        """Calcule la conversion px → nm sur l’image réelle."""
        if self.S.img_color is None:
            self._warn("Aucune image.")
            return
        if len(self.S.scale_points) != 2:
            self._warn("Clique exactement 2 points.")
            return
    
        txt = self.edt_scale_nm.text().strip()
        scale_nm = float(txt) if txt != "" else DEFAULT_SCALE_NM
    
        (x1, y1), (x2, y2) = self.S.scale_points
        bar_len_px_full = float(np.hypot(x2 - x1, y2 - y1))  # ✅ distance sur image réelle
    
        if bar_len_px_full <= 0:
            self._warn("Longueur de barre nulle.")
            return
    
        nm_per_px = scale_nm / bar_len_px_full
        self.S.nm_per_px = nm_per_px
    
        self._info(f"📏 Échelle validée : 1 px = {nm_per_px:.5f} nm "
                   f"(barre {scale_nm} nm, {bar_len_px_full:.1f} px)")
    
        # Texte d’aide sous l’image
        self.ax0.text(
            0.5, -0.08,
            f"↔ {scale_nm:.1f} nm sur {bar_len_px_full:.1f} px → 1 px = {nm_per_px:.4f} nm",
            transform=self.ax0.transAxes,
            ha='center', va='top', fontsize=10, color='green'
        )
        self.canvas0.draw_idle()
    
        # Passage étape suivante
        self.stack.setCurrentIndex(1)
        self.draw_step1_image()


    # ----------- Step 1: ROI -----------
    def build_step1_roi(self):
        page = QWidget()
        v = QVBoxLayout(page)
    
        # --- Image d’abord (en haut) ---
        self.fig1 = Figure(figsize=(7, 6))
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        v.addWidget(self.canvas1)
    
        # --- Texte explicatif en bas de l’image ---
        lbl_explication = QLabel(
            "🧭 <b>Étape 1 — Définition de la zone d’analyse (ROI)</b><br><br>"
            "👉 Cliquer sur <b>« Sélection ROI (activer) »</b>, puis tracer la zone d’intérêt sur l’image :<br>"
            "• Cliquer et faire glisser avec le bouton gauche pour définir la zone.<br>"
            "• Relâcher le bouton pour fixer la sélection.<br><br>"
            "➡️ Cliquer ensuite sur <b>« Valider ROI »</b> pour confirmer la sélection."
            "<b>« Valider ROI »</b> pour confirmer."
        )
        lbl_explication.setStyleSheet("""
            QLabel {
                color: #1e293b;
                font-size: 13px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        lbl_explication.setAlignment(Qt.AlignCenter)
        v.addWidget(lbl_explication)
    
        # --- Boutons tout en bas ---
        row = QHBoxLayout()
        self.btn_roi_mode = QPushButton("Sélection ROI (activer)")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.clicked.connect(self.toggle_roi)
        self.btn_roi_ok = QPushButton("Valider ROI")
        self.btn_roi_ok.clicked.connect(self.confirm_roi)
        row.addWidget(self.btn_roi_mode)
        row.addWidget(self.btn_roi_ok)
        v.addLayout(row)
    
        self.rs = None
        return page


    def draw_step1_image(self):
        if self.S.img_full is None:
            return
        self.ax1.clear()
        # Toujours afficher l’image originale pour tracer la ROI
        self.ax1.imshow(self.S.img_full, cmap='gray')
        self.ax1.set_aspect('equal', 'box')
        self.ax1.set_title("Trace ROI (glisser)")

        # Si une ROI existe déjà, on la dessine
        if self.S.roi_rect is not None:
            x0, y0, x1, y1 = self.S.roi_rect
            self.ax1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         fill=False, edgecolor='lime', lw=2))
        self.canvas1.draw_idle()

    def toggle_roi(self, checked):
        if self.S.img is None:
            self._warn("Charge une image."); self.btn_roi_mode.setChecked(False); return
        if checked:
            self.btn_roi_mode.setText("Sélection ROI (active)")
            if self.rs is not None:
                try: self.rs.set_active(False)
                except Exception: pass
                self.rs = None
            self.rs=RectangleSelector(
                self.ax1, self.on_roi_select, useblit=True, button=[1],
                minspanx=10, minspany=10, spancoords='pixels', interactive=True)
        else:
            self.btn_roi_mode.setText("Sélection ROI (activer)")
            if self.rs is not None:
                try: self.rs.set_active(False)
                except Exception: pass
                self.rs=None
        self.canvas1.draw_idle()

    def on_roi_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None: return
        x0,y0=int(eclick.xdata), int(eclick.ydata); x1,y1=int(erelease.xdata), int(erelease.ydata)
        x0,x1=sorted([x0,x1]); y0,y1=sorted([y0,y1]); self.S.roi_rect=(x0,y0,x1,y1)
        self.draw_step1_image()

    def confirm_roi(self):
        if self.S.img_full is None:
            self._warn("Aucune image chargée.")
            return

        # ROI depuis RectangleSelector ; sinon image entière
        if self.S.roi_rect is None:
            h, w = self.S.img_full.shape
            self.S.roi_rect = (0, 0, w, h)

        x0, y0, x1, y1 = self.S.roi_rect
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            self._warn("ROI trop petite.")
            return

        roi = self.S.img_full[y0:y1, x0:x1].copy()
        self.S.roi = roi
        self.S.img = roi
        self.S.img_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        self._info(f"✅ ROI validée : {roi.shape[1]}x{roi.shape[0]} px (image de travail)")

        # Préparation live
        if self.S.live_enabled and 0.3 <= self.S.live_scale < 1.0:
            self.S.roi_live = cv2.resize(
                roi,
                (int(roi.shape[1] * self.S.live_scale),
                 int(roi.shape[0] * self.S.live_scale)),
                interpolation=cv2.INTER_AREA
            )
            self.S.live_factor = float(roi.shape[1] / self.S.roi_live.shape[1])
        else:
            self.S.roi_live = roi.copy()
            self.S.live_factor = 1.0

        # Aller à l’étape 2
        self.stack.setCurrentIndex(2)
        self.grp_wedge.setEnabled(True)
        self.reset_crops()
        self.prepare_fft_caches()
        self.refresh_step2(force_full_redraw=True)
        
    
    def update_roi_live_only(self):
        """Met à jour uniquement roi_live + live_factor sans relancer confirm_roi()."""
        if self.S.roi is None:
            return
    
        roi = self.S.roi
        if self.S.live_enabled and 0.3 <= self.S.live_scale < 1.0:
            self.S.roi_live = cv2.resize(
                roi,
                (int(roi.shape[1] * self.S.live_scale),
                 int(roi.shape[0] * self.S.live_scale)),
                interpolation=cv2.INTER_AREA
            )
            self.S.live_factor = roi.shape[1] / self.S.roi_live.shape[1]
        else:
            self.S.roi_live = roi.copy()
            self.S.live_factor = 1.0
    
        # Rebuild FFT cache preview si possible
        if self.S.roi_live is not None:
            fL = np.fft.fft2(self.S.roi_live)
            self.S.f_live = fL
            self.S.fshift_live = np.fft.fftshift(fL)
            self.S.fft_mag_live = np.log1p(np.abs(self.S.fshift_live))


    # ----------- FFT caches -----------
    def prepare_fft_caches(self):
        if self.S.roi is None: return
        f=np.fft.fft2(self.S.roi); fshift=np.fft.fftshift(f); mag=np.log1p(np.abs(fshift))
        self.S.f_full=f; self.S.fshift_full=fshift; self.S.fft_mag_full=mag
        rlive=self.S.roi_live
        fL=np.fft.fft2(rlive); fshiftL=np.fft.fftshift(fL); magL=np.log1p(np.abs(fshiftL))
        self.S.f_live=fL; self.S.fshift_live=fshiftL; self.S.fft_mag_live=magL

    # ----------- Step 2: FFT & live wedge -----------
    def build_step2_fft(self):
        page=QWidget(); v=QVBoxLayout(page)
        self.lbl2=QLabel("Étape 2 — Sélection de l’orientation dans la FFT\n\n"
        "👉 Déplace la souris dans la FFT (panneau gauche) :\n"
        "   • l’orientation du wedge suit la position de la souris\n"
        "   • clic gauche = verrouille l’angle choisi\n"
        "   • touche R = réinitialise l’angle\n\n"
        "➡️ Ajuste aussi l’ouverture avec le curseur à gauche si nécessaire.")
        v.addWidget(self.lbl2)
        self.fig2 = Figure(figsize=(15, 4), facecolor="white")  # plus large et un peu plus haute
        self.ax2 = [
            self.fig2.add_subplot(1, 3, 1),
            self.fig2.add_subplot(1, 3, 2),
            self.fig2.add_subplot(1, 3, 3)]

        self.canvas2=FigureCanvas(self.fig2); v.addWidget(self.canvas2)
        self.canvas2.mpl_connect('motion_notify_event', self.on_fft_motion)
        self.canvas2.mpl_connect('button_press_event', self.on_fft_click)
        row=QHBoxLayout()
        btn_reset=QPushButton("Réinitialiser wedge"); btn_reset.clicked.connect(self.reset_wedge)
        btn_apply_full = QPushButton("Appliquer en pleine résolution")
        btn_apply_full.clicked.connect(lambda: self.stack.setCurrentWidget(self.page3))
        row.addWidget(btn_reset); row.addWidget(btn_apply_full); v.addLayout(row)
        self.im2_fft=self.im2_mask=self.im2_ifft=None; self.txt2_info=None
        return page

    def set_angle_target(self, a): self.S.angle_target=float(a)
    def on_open_change(self, v): self.S.opening=float(v)
    
    def on_toggle_live(self, state):
        self.S.live_enabled = (state == Qt.Checked)
        self.update_roi_live_only()
        if self.stack.currentWidget() is self.page2:
            self.refresh_step2(force_full_redraw=True)
        if self.stack.currentWidget() is self.page3:
            self.refresh_step3(force_full_redraw=True)
    
    def on_live_scale_change(self, v):
        self.S.live_scale = max(0.3, min(1.0, v/100.0))
        self.update_roi_live_only()
        if self.stack.currentWidget() is self.page2:
            self.refresh_step2(force_full_redraw=True)
        if self.stack.currentWidget() is self.page3:
            self.refresh_step3(force_full_redraw=True)

    def reset_wedge(self): self.S.angle=0.0; self.S.angle_target=0.0; self.sld_angle.setValue(0)
    def on_fft_motion(self, event):
        if self.stack.currentWidget() is not self.page2: return
        if self.S.roi_live is None or event.inaxes!=self.ax2[0] or event.xdata is None: return
        rows,cols=self.S.roi_live.shape; cx,cy=cols//2, rows//2
        dx=event.xdata-cx; dy=event.ydata-cy
        self.S.angle_target=float(np.degrees(np.arctan2(dy, dx)))
    def on_fft_click(self, event):
        if self.stack.currentWidget() is not self.page2: return
        if self.S.roi_live is None or event.inaxes!=self.ax2[0] or event.xdata is None: return
        rows,cols=self.S.roi_live.shape; cx,cy=cols//2, rows//2
        dx=event.xdata-cx; dy=event.ydata-cy
        self.S.angle=float(np.degrees(np.arctan2(dy, dx))); self.S.angle_target=self.S.angle; self.sld_angle.setValue(int(round(self.S.angle)))

    def tick(self):
        da=(self.S.angle_target-self.S.angle+180)%360-180
        self.S.angle+=SMOOTH_ALPHA*da
        if self.stack.currentWidget() is self.page2:
            self.refresh_step2()
        elif self.stack.currentWidget() is self.page3:
            self.refresh_step3()

    def build_mask_live(self, rows, cols, crow, ccol):
        mask=make_wedge_mask(rows, cols, crow, ccol, self.S.angle, self.S.opening)
        Y,X=np.ogrid[:rows,:cols]; R=np.hypot(X-ccol, Y-crow)
        if self.S.r_limit is not None: mask[R>self.S.r_limit/self.S.live_factor]=0
        if self.S.r_inner is not None: mask[R<self.S.r_inner/self.S.live_factor]=0
    
        return mask

        
    def refresh_step2(self, force_full_redraw=False):
        if self.S.roi_live is None:
            return
    
        rows, cols = self.S.roi_live.shape
        crow, ccol = rows // 2, cols // 2
        mask = self.build_mask_live(rows, cols, crow, ccol)
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(self.S.fshift_live * mask)))
    
        if self.im2_fft is None or force_full_redraw:
            for ax in self.ax2: ax.clear()
    
            # (1) FFT + masque
            self.im2_fft = self.ax2[0].imshow(self.S.fft_mag_live, cmap='gray', interpolation='nearest')
            self.im2_mask_overlay = self.ax2[0].imshow(mask * 255, cmap='gray', alpha=0.35, interpolation='nearest')
            self.ax2[0].axis('off')
            self.ax2[0].set_title("FFT + masque", fontsize=10)
    
            # (2) Masque
            self.im2_mask = self.ax2[1].imshow(mask * 255, cmap='gray', interpolation='nearest')
            self.ax2[1].axis('off')
            self.ax2[1].set_title("Masque (aperçu)", fontsize=10)
    
            # (3) IFFT
            self.im2_ifft = self.ax2[2].imshow(img_back, cmap='gray', interpolation='nearest')
            self.ax2[2].axis('off')
            self.ax2[2].set_title("IFFT (aperçu)", fontsize=10)
    
        else:
            self.im2_mask_overlay.set_data(mask * 255)
            self.im2_mask.set_data(mask * 255)
            self.im2_ifft.set_data(img_back)
    
        # --- Titre global au-dessus ---
        self.fig2.suptitle(
            f"Angle : {self.S.angle:.1f}°   |   Ouverture : {self.S.opening:.1f}°",
            fontsize=12,
            color="#1d4ed8",
            fontweight="bold",
            y=0.98
        )
    
        self.fig2.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        self.canvas2.setVisible(True)
        self.canvas2.draw_idle()


    # ----------- Step 3: Croppings -----------
    def build_step3_cropping(self):
        page = QWidget(); v = QVBoxLayout(page)
        self.lbl3 = QLabel("Étape 3 — Application de rognages dans la FFT\n\n"
        "👉 Choisis un mode de rognage :\n"
        "   • Extérieur cercle = coupe les hautes fréquences\n"
        "   • Intérieur cercle = coupe les basses fréquences\n"
        "   • Intérieur carré = masque une zone centrale carrée\n"
        "   • Aucun = pas de rognage (wedge seul)\n\n"
        "👉 Ensuite, clique et fais glisser (drag) dans la FFT (aperçu à gauche)\n"
        "pour définir le rayon ou la taille du carré.\n\n"
        "➡️ R = réinitialise les rognages.")
        v.addWidget(self.lbl3)
        self.fig3 = Figure(figsize=(10,4))
        self.ax3 = [self.fig3.add_subplot(1,2,1), self.fig3.add_subplot(1,2,2)]
        self.canvas3 = FigureCanvas(self.fig3); v.addWidget(self.canvas3)
        row_modes = QHBoxLayout()
        
        self.rb_ext = QRadioButton("Extérieur")
        self.rb_int = QRadioButton("Intérieur cercle")
        self.rb_none = QRadioButton("Aucun")
        self.rb_ext.setChecked(True)
        
        for rb in (self.rb_ext, self.rb_int, self.rb_none):
            row_modes.addWidget(rb)

        v.addLayout(row_modes)
        row = QHBoxLayout()
        btn_reset = QPushButton("Réinitialiser rognages"); btn_reset.clicked.connect(self.reset_crops)
        btn_apply = QPushButton("Appliquer en pleine résolution"); btn_apply.clicked.connect(self.go_to_step4)
        row.addWidget(btn_reset); row.addWidget(btn_apply); v.addLayout(row)
        self.canvas3.mpl_connect('button_press_event', self.on_crop_press)
        self.canvas3.mpl_connect('motion_notify_event', self.on_crop_move)
        self.canvas3.mpl_connect('button_release_event', self.on_crop_release)
        self.im3_fft = self.im3_mask = None
        return page

    def go_to_step4(self):
        """Étape 3 → Étape 4 : applique wedge + rognages en pleine résolution"""
        self.apply_full_res_and_compute()   # calcul plein-rés
        self.draw_step4_panels()
        self.stack.setCurrentWidget(self.page4)

    def reset_crops(self):
        self.S.r_limit = None
        self.S.r_inner = None
        self.S.square_half = None
        self.S.mask = None
        self.im3_fft = None
        self.im3_mask = None
        if self.stack.currentWidget() is self.page3:
            self.refresh_step3(force_full_redraw=True)
        self._info("♻️ Rognages remis à zéro (wedge seul).")

    def refresh_step3(self, force_full_redraw=False):
        if self.S.roi_live is None:
            return
        rows, cols = self.S.roi_live.shape
        crow, ccol = rows // 2, cols // 2
        mask = self.build_mask_live(rows, cols, crow, ccol)

        if self.im3_fft is None or force_full_redraw:
            for ax in self.ax3: ax.clear()
            self.im3_fft = self.ax3[0].imshow(self.S.fft_mag_live, cmap='gray', interpolation='nearest')
            self.im3_mask_overlay = self.ax3[0].imshow(mask * 255, cmap='gray', alpha=0.35, interpolation='nearest')
            self.ax3[0].axis('off')
            self.ax3[0].set_title("FFT + masque (aperçu)")
            self.im3_mask = self.ax3[1].imshow(mask * 255, cmap='gray', interpolation='nearest')
            self.ax3[1].axis('off'); self.ax3[1].set_title("Masque (aperçu)")
        else:
            self.im3_mask_overlay.set_data(mask * 255)
            self.im3_mask.set_data(mask * 255)

        # Nettoyage patches et redessin guides
        for p in list(self.ax3[1].patches):
            try: p.remove()
            except Exception: pass
        if self.S.r_limit is not None:
            self.ax3[1].add_patch(Circle((ccol, crow), self.S.r_limit / self.S.live_factor, fill=False, edgecolor='r', lw=2))
        if self.S.r_inner is not None:
            self.ax3[1].add_patch(Circle((ccol, crow), self.S.r_inner / self.S.live_factor, fill=False, edgecolor='b', lw=2))
        if self.S.square_half is not None:
            s = self.S.square_half / self.S.live_factor
            self.ax3[1].add_patch(Rectangle((ccol - s, crow - s), 2 * s, 2 * s, fill=False, edgecolor='c', lw=2))

        self.canvas3.setVisible(True)
        self.canvas3.draw_idle()

    def on_crop_press(self, event):
        if self.stack.currentWidget() is not self.page3: return
        self._dragging=True; self.update_crop_from_event(event)
    def on_crop_move(self, event):
        if self.stack.currentWidget() is not self.page3 or not getattr(self,'_dragging',False): return
        self.update_crop_from_event(event)
    def on_crop_release(self, event): self._dragging=False

    def update_crop_from_event(self, event):
        if self.S.roi_live is None or event.inaxes!=self.ax3[0] or event.xdata is None:
            return
        rows, cols = self.S.roi_live.shape
        cx, cy = cols // 2, rows // 2
        dx = event.xdata - cx
        dy = event.ydata - cy
        r = int(max(1, np.hypot(dx, dy)))

        if self.rb_ext.isChecked():
            self.S.r_limit = int(r * self.S.live_factor)
        elif self.rb_int.isChecked():
            self.S.r_inner = int(r * self.S.live_factor)
        
        # ⚡ rafraîchit uniquement Step 3
        self.refresh_step3(force_full_redraw=True)

    # ----------- Masque plein-résolution factorisé -----------
    def build_fullres_mask(self):
        """Construit le masque wedge + rognages en pleine résolution."""
        if self.S.roi is None:
            return None
        rows, cols = self.S.roi.shape
        crow, ccol = rows // 2, cols // 2
        mask = make_wedge_mask(rows, cols, crow, ccol, self.S.angle, self.S.opening)

        # Appliquer rognages
        Y, X = np.ogrid[:rows, :cols]
        R = np.hypot(X - ccol, Y - crow)
        if self.S.r_limit is not None:
            mask[R > self.S.r_limit] = 0
        if self.S.r_inner is not None:
            mask[R < self.S.r_inner] = 0
    
        return mask

    # ----------- Calcul plein-rés (unique) -----------
    def apply_full_res_and_compute(self):
        """Calcule IFFT + λ + énergie avec wedge + rognages (plein-rés)."""
        if self.S.roi is None:
            self._warn("Définis d'abord la ROI.")
            return

        mask = self.build_fullres_mask()
        if mask is None:
            self._warn("Masque indisponible.")
            return

        self.S.mask = mask

        if self.S.fshift_full is None:
            self.prepare_fft_caches()

        # IFFT filtrée
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(self.S.fshift_full * mask)))
        self.S.img_back = img_back

        # λ auto (robuste)
        rows, cols = self.S.roi.shape
        crow, ccol = rows // 2, cols // 2
        mag_for_peak = np.log1p(np.abs(self.S.fshift_full) * mask)
        mag_for_peak[crow-5:crow+5, ccol-5:ccol+5] = 0
        y_peak, x_peak = np.unravel_index(np.argmax(mag_for_peak), mag_for_peak.shape)
        dy, dx = y_peak - crow, x_peak - ccol
        f_val = max(1e-6, float(np.hypot(dy / rows, dx / cols)))  # clamp
        lambda_px = 1.0 / f_val
        self.S.lambda_px = lambda_px
        self.S.lambda_nm = lambda_px * (self.S.nm_per_px if self.S.nm_per_px else 1.0)

        self._info(f"λ auto ≈ {self.S.lambda_nm:.2f} nm  (≈ {self.S.lambda_px:.2f} px)")

        # Énergie (calculée, pas affichée)
        self.compute_energy(auto=True)

    # ----------- Step 4: IFFT + λ auto -----------
    def build_step4_ifft_lambda(self):
        page = QWidget(); v = QVBoxLayout(page)
        v.addWidget(QLabel("Étape 4 — IFFT & λ auto (masque appliqué)"))
        self.fig4 = Figure(figsize=(12, 8))
        self.ax4 = [
            self.fig4.add_subplot(2, 2, 1),
            self.fig4.add_subplot(2, 2, 2),
            self.fig4.add_subplot(2, 2, 3),
            self.fig4.add_subplot(2, 2, 4),
        ]
        self.canvas4 = FigureCanvas(self.fig4); v.addWidget(self.canvas4)
        self.canvas4.setVisible(False)
        btn = QPushButton("Recalculer IFFT + λ + Gabor (auto)")
        btn.clicked.connect(self.recompute_ifft_step4)
        v.addWidget(btn)
        return page

    def draw_step4_panels(self):
        """Affiche FFT / masque / IFFT (énergie calculée en coulisse)."""
        if self.S.roi is None or self.S.mask is None or self.S.img_back is None:
            self._warn("Aucun calcul disponible — clique sur Recalculer.")
            return
        for ax in self.ax4: ax.clear()
        self.ax4[0].imshow(self.S.fft_mag_full, cmap='gray'); self.ax4[0].set_title("FFT"); self.ax4[0].axis('off')
        self.ax4[1].imshow(self.S.mask*255, cmap='gray'); self.ax4[1].set_title("Masque wedge final"); self.ax4[1].axis('off')
        self.ax4[2].imshow(self.S.img_back, cmap='gray'); self.ax4[2].set_title("IFFT filtrée"); self.ax4[2].axis('off')
        self.ax4[3].imshow(self.S.img_back, cmap='gray'); self.ax4[3].set_title("IFFT (énergie calculée en arrière-plan)"); self.ax4[3].axis('off')
        self.canvas4.setVisible(True); self.canvas4.draw_idle()

 
    def recompute_ifft_step4(self):
        """Recalcule IFFT + λ + énergie, puis passe directement à l'étape 6."""
        if self.S.roi is None:
            self._warn("Définis d'abord la ROI.")
            return
    
        # Recalcule IFFT, λ et énergie
        self.apply_full_res_and_compute()
        self.draw_step4_panels()
    
        # Passe directement à l’étape 6 (détection)
        self.stack.setCurrentWidget(self.page6)
        self._info("➡️ Passage automatique à l’étape 6 pour la détection.")


  # ----------- Énergie (calcul seulement) -----------
    def compute_energy(self, auto=True):
        if self.S.img_back is None:
            if not auto: self._warn("IFFT manquante — passe par l'étape 4.")
            return
        lambd = float(self.S.lambda_px if self.S.lambda_px else 12.0)
        theta = np.deg2rad(self.S.angle)
        kernel = cv2.getGaborKernel((KSIZE_GABOR, KSIZE_GABOR), SIGMA_GABOR, theta, lambd, GAMMA_GABOR, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(self.S.img_back.astype(np.float32), cv2.CV_32F, kernel)
        energy = cv2.normalize(np.abs(filtered), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #thr = np.percentile(energy, ENERGY_PCTL)
        pctl = getattr(self.S, "energy_percentile", ENERGY_PCTL)
        thr = np.percentile(energy, pctl)
        _, mask_energy_u8 = cv2.threshold(energy, int(thr), 255, cv2.THRESH_BINARY)
        

        # ⚡ masque binaire 0/1
        mask_energy = (mask_energy_u8 > 0).astype(np.uint8)
        kernel_m = np.ones((3, 3), np.uint8)
        mask_energy = cv2.morphologyEx(mask_energy, cv2.MORPH_OPEN, kernel_m, iterations=1)
        mask_energy = cv2.morphologyEx(mask_energy, cv2.MORPH_CLOSE, kernel_m, iterations=1)
        self.S.energy = energy
        self.S.energy_mask = mask_energy
        if auto:
            self._info(f"⚡ Énergie calculée automatiquement (percentile {ENERGY_PCTL}).")
   
    def build_step6_detect(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("Étape 6 — Détection zones & mesures"))
    
        # 🖼️ Figure agrandie, plus visible
        self.fig6 = Figure(figsize=(20, 10), facecolor="white")  # plus grande que ta version actuelle
        gs = self.fig6.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.05, wspace=0.05)
    
        # --- Disposition : ROI en haut, IFFT + Franges détectées en bas ---
        ax_roi = self.fig6.add_subplot(gs[0, :])   # haut = ROI pleine largeur
        ax_ifft = self.fig6.add_subplot(gs[1, 0])  # bas gauche = IFFT
        ax_frg  = self.fig6.add_subplot(gs[1, 1])  # bas droite = Franges détectées
    
        self.ax6 = [ax_roi, ax_ifft, ax_frg]
        self.canvas6 = FigureCanvas(self.fig6)
        v.addWidget(self.canvas6)
        self.canvas6.setVisible(False)
    
        # --- Boutons d’action ---
        row = QHBoxLayout()
        btn_detect = QPushButton("🔍 Détecter zones & mesurer")
        btn_detect.clicked.connect(self.run_detection)
        row.addWidget(btn_detect)
    
        btn_next = QPushButton("→ Étape 7 : Courbes & histogrammes")
    
        def go_to_step7():
            self.stack.setCurrentWidget(self.page7)
            self.refresh_plots()
            self._info("📊 Passage à l’étape 7 (courbes & histogrammes mis à jour).")
    
        btn_next.clicked.connect(go_to_step7)
        row.addWidget(btn_next)
        v.addLayout(row)
    
        return page


    def run_detection(self):
        if self.S.energy_mask is None or self.S.img_back is None:
            self._warn("Énergie/masque manquant (étape 4).")
            return

        # Lire λ min/max une seule fois
        try:
            lam_min = float(self.edt_lambda_min.text())
        except ValueError:
            lam_min = 0.35; self.edt_lambda_min.setText("0.35")
        try:
            lam_max = float(self.edt_lambda_max.text())
        except ValueError:
            lam_max = 0.65; self.edt_lambda_max.setText("0.65")

        contours, _ = cv2.findContours((self.S.energy_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = cv2.cvtColor(
            cv2.normalize(self.S.img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )

        results = []
        nm_per_px = float(self.S.nm_per_px if self.S.nm_per_px else 1.0)
        all_fringe_lengths = []

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_ZONE_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Orientation locale
            ang = orientation_pca(cnt)
            perp = ang + np.pi / 2

            seg_length = max((h / 2 if abs(np.cos(ang)) > abs(np.sin(ang)) else w / 2), 40)
            dxs, dys = np.cos(perp), np.sin(perp)
            pA = (int(cx - seg_length * dxs), int(cy - seg_length * dys))
            pB = (int(cx + seg_length * dxs), int(cy + seg_length * dys))

            # Profil
            prof = line_profile(self.S.roi, pA, pB, n=300)
            prof = np.abs(prof - prof.mean())
            prof = gaussian_filter1d(prof, sigma=2)

            seg_len_px = float(np.hypot(pB[0] - pA[0], pB[1] - pA[1]))
            px_per_sample = max(1e-6, seg_len_px / 300.0)

            lambda_px = float(self.S.lambda_px if self.S.lambda_px else 12.0)
            lambda_samples = lambda_px * 300.0 / seg_len_px if seg_len_px > 0 else 12.0
            min_dist = max(3, int(0.6 * lambda_samples))

            peaks, _ = find_peaks(prof, distance=min_dist, prominence=prof.std() * 0.5)

            widths_nm = []
            if len(peaks) > 0:
                widths_res = peak_widths(prof, peaks, rel_height=0.5)
                widths_px = widths_res[0] * px_per_sample
                widths_nm = list(widths_px * nm_per_px)

            if len(peaks) >= 2:
                dists_px = np.diff(peaks) * px_per_sample
                lambda_local = float(np.mean(dists_px) * nm_per_px)
            else:
                lambda_local = 0.0

            # Vérification des bornes λ
            if not (lam_min <= lambda_local <= lam_max):
                continue

            n = len(peaks)

            # Largeur projetée (direction des franges)
            dx_f, dy_f = np.cos(ang), np.sin(ang)
            pts = cnt.reshape(-1, 2)
            proj = pts @ np.array([dx_f, dy_f])
            L_largeur_nm = float((proj.max() - proj.min()) * nm_per_px)

            L_hauteur_nm = (n - 1) * lambda_local if n >= 2 else 0.0
            L_largeur_moy_emp = L_largeur_nm 

            # Dessiner contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            #cv2.drawContours(out, [box], 0, (0, 0, 255), 2)
            
            
            # Dessiner un point rouge au centre de la zone détectée
            cx, cy = int(np.mean(cnt[:, 0, 0])), int(np.mean(cnt[:, 0, 1]))
            cv2.circle(out, (cx, cy), 2, (0, 0, 255), -1)

            # Dessiner un point rouge au centre de la zone détectée
            


            # Sauvegarde des longueurs de franges (projection robuste)
            if n > 0:
                xs = np.linspace(pA[0], pB[0], 300)
                ys = np.linspace(pA[1], pB[1], 300)
                mask_rect = np.zeros(self.S.img_back.shape, dtype=np.uint8)
                cv2.fillPoly(mask_rect, [box.reshape((-1, 1, 2))], 255)

                for pk in peaks:
                    p = (int(xs[pk]), int(ys[pk]))
                    pA_long = (int(p[0] - 1000 * dx_f), int(p[1] - 1000 * dy_f))
                    pB_long = (int(p[0] + 1000 * dx_f), int(p[1] + 1000 * dy_f))

                    line_mask = np.zeros(self.S.img_back.shape, dtype=np.uint8)
                    cv2.line(line_mask, pA_long, pB_long, 255, 1)
                    clipped = cv2.bitwise_and(line_mask, mask_rect)
                    ys_c, xs_c = np.where(clipped > 0)

                    if len(xs_c) > 0:
                        coords = np.column_stack([xs_c, ys_c]).astype(np.float32)
                        d = np.array([dx_f, dy_f], dtype=np.float32)
                        proj = coords @ d
                        length_px = float(proj.max() - proj.min())
                        length_nm = float(length_px * nm_per_px)
                    else:
                        length_nm = L_largeur_nm / n if n > 0 else 0.0

                    if length_nm > 0:
                        all_fringe_lengths.append(length_nm)

            cv2.putText(out, f"{n} | {lambda_local:.2f} nm", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            results.append({
                'zone': (int(x), int(y), int(w), int(h)),
                'n_fringes': int(n),
                'lambda_local': float(lambda_local),
                'L_hauteur': float(L_hauteur_nm),
                'L_largeur': float(L_largeur_nm),
                'L_largeur_moy_emp': float(L_largeur_moy_emp),
                'fringe_lengths': all_fringe_lengths[-n:] if n > 0 else [],
                'fringe_widths': widths_nm,
            })

        # Sauvegarde résultats
        self.S.results = results
        self.S.out_vis = out

        # === AFFICHAGE ===
        for ax in self.ax6:
            ax.clear()

        display_opts = dict(cmap="gray", interpolation="nearest")

        # (haut) ROI
        if self.S.roi is not None:
            self.ax6[0].imshow(self.S.roi, **display_opts)
            self.ax6[0].set_title("ROI validée", fontsize=11)
            self.ax6[0].axis("off")

        # (bas gauche) IFFT
        self.ax6[1].imshow(self.S.img_back, **display_opts)
        self.ax6[1].set_title("IFFT filtrée", fontsize=11)
        self.ax6[1].axis("off")

        # (bas droite) Franges détectées
        self.ax6[2].imshow(out[..., ::-1], interpolation="nearest")
        self.ax6[2].set_title("Franges détectées", fontsize=11)
        self.ax6[2].axis("off")

        
        self.fig6.tight_layout(pad=0.5)
        self.fig6.subplots_adjust(left=0, right=1, top=0.98, bottom=0, wspace=0.02, hspace=0.05)

        self.canvas6.setVisible(True)
        self.canvas6.draw_idle()

        # Console
        self._info("=== Résultats automatiques ===")
        for i, r in enumerate(results, 1):
            self._info(f"{i:02d}. Zone {r['zone']} : {r['n_fringes']} franges, "
                       f"Hauteur ≈ {r['L_hauteur']:.2f} nm, "
                       f"Largeur moy. emp. ≈ {r['L_largeur_moy_emp']:.2f} nm")

        lam = [r['lambda_local'] for r in results if r['lambda_local'] > 0]
        if lam:
            self._info(f"📏 Distance moyenne entre franges : {np.mean(lam):.2f} nm")

        expected_fringes = sum(r['n_fringes'] for r in results)
        self._info(f"🔢 Nombre total attendu (somme des n_fringes) : {expected_fringes}")
        self._info(f"🔢 Nombre total récupéré (longueurs individuelles) : {len(all_fringe_lengths)}")

        if all_fringe_lengths:
            mean_len = float(np.mean(all_fringe_lengths))
            std_len = float(np.std(all_fringe_lengths))
            min_len = float(np.min(all_fringe_lengths))
            max_len = float(np.max(all_fringe_lengths))
            self._info(f"📊 Moyenne = {mean_len:.2f} nm, Écart-type = {std_len:.2f} nm")
            self._info(f"📊 Min = {min_len:.2f} nm, Max = {max_len:.2f} nm")

    # ----------- Step 7: Plots & histograms -----------
    def build_step7_plots(self):
        page=QWidget(); v=QVBoxLayout(page)
        v.addWidget(QLabel("Étape 7 — Courbes & histogrammes"))
        self.fig7a=Figure(figsize=(12,4)); self.ax7a=[self.fig7a.add_subplot(1,2,1), self.fig7a.add_subplot(1,2,2)]
        self.canvas7a=FigureCanvas(self.fig7a); v.addWidget(self.canvas7a); self.canvas7a.setVisible(False)
        self.fig7b=Figure(figsize=(12,4)); self.ax7b=[self.fig7b.add_subplot(1,2,1), self.fig7b.add_subplot(1,2,2)]
        self.canvas7b=FigureCanvas(self.fig7b); v.addWidget(self.canvas7b); self.canvas7b.setVisible(False)
        row=QHBoxLayout()
        btn_refresh=QPushButton("Actualiser graphiques"); btn_refresh.clicked.connect(self.refresh_plots)
        btn_back=QPushButton("← Retour mesures"); btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page6))
        row.addWidget(btn_refresh); row.addWidget(btn_back); v.addLayout(row)
        return page

    def refresh_plots(self):
        if not self.S.results:
            self._warn("Aucun résultat — lance l'étape 6 d'abord."); 
            return

        heights=[r['L_hauteur'] for r in self.S.results if r['L_hauteur']>0]
        nfr=[r['n_fringes'] for r in self.S.results if r['L_hauteur']>0]
        means=[r['L_largeur_moy_emp'] for r in self.S.results if r['L_largeur_moy_emp']>0]
        nfr2=[r['n_fringes'] for r in self.S.results if r['L_largeur_moy_emp']>0]
        lambdas=[r['lambda_local'] for r in self.S.results if r['lambda_local']>0]
        all_fringe_lengths = [l for r in self.S.results for l in r.get("fringe_lengths", []) if l > 0]

        for ax in (*self.ax7a, *self.ax7b): ax.clear()

        # --- Scatter: Franges vs hauteur ---
        self.ax7a[0].scatter(heights, nfr, s=50, edgecolors='k')
        self.ax7a[0].set_xlabel("Hauteur empilement (nm)", fontsize=7)
        self.ax7a[0].set_ylabel("Nombre de franges", fontsize=7)
        self.ax7a[0].set_title("Nombre de feuillets par empilement vs $L_c$", fontsize=9)
        self.ax7a[0].grid(True)
        if heights:
            mean_height = np.mean(heights)
            self.ax7a[0].text(0.95, 0.95, f"$\hat{{L}}_c$ = {mean_height:.2f} nm",
                              ha="right", va="top", transform=self.ax7a[0].transAxes,
                              fontsize=7, bbox=dict(facecolor="white", alpha=0.6))

        # --- Scatter: Franges vs longueur moyenne ---
        self.ax7a[1].scatter(means, nfr2, s=50, edgecolors='k')
        self.ax7a[1].set_xlabel("Longueur $L_a$ (nm)", fontsize=7)
        self.ax7a[1].set_ylabel("Nombre de feuillets/ emp N", fontsize=7)
        self.ax7a[1].set_title("Nombre de feullets par empilement vs $L_a$", fontsize=9)
        self.ax7a[1].grid(True)
        if means:
            mean_means = np.mean(means)
            self.ax7a[1].text(0.95, 0.95, f"$\hat{{L}}_a$ ={mean_means:.2f} nm",
                              ha="right", va="top", transform=self.ax7a[1].transAxes,
                              fontsize=7, bbox=dict(facecolor="white", alpha=0.6))

        # --- Histogramme: λ locaux ---

        # --- Histogramme: λ locaux ---
        if lambdas:
            self.ax7b[0].hist(lambdas, bins=20, edgecolor='k')
            self.ax7b[0].set_xlabel("<d₀₀₂> (nm)", fontsize=7)  # on garde simple
            self.ax7b[0].set_ylabel("Nombre d'empilement", fontsize=7)
            self.ax7b[0].set_title("Distribution des $\hat{d₀₀₂}$ moyens par empilement", fontsize=9)
            mean_lambda = np.mean(lambdas)
            self.ax7b[0].text(
                0.95, 0.95, f"$\hat{{d}}_{{emp}}$ ={mean_lambda:.2f} nm",
                ha="right", va="top", transform=self.ax7b[0].transAxes,
                fontsize=7, bbox=dict(facecolor="white", alpha=0.6)
            )
        
         


        # --- Histogramme: longueurs individuelles ---
        if all_fringe_lengths:
            self.ax7b[1].hist(all_fringe_lengths, bins=20, edgecolor='k')
            self.ax7b[1].set_xlabel("Longueur frange (nm)", fontsize=7)
            self.ax7b[1].set_ylabel("Nombre", fontsize=7)
            self.ax7b[1].set_title("Distribution des longueurs individuelles de feuillet", fontsize=9)
            mean_len = np.mean(all_fringe_lengths)
            self.ax7b[1].text(0.95, 0.95, f"$\hat{{l}}$ ={mean_len:.2f} nm",
                              ha="right", va="top", transform=self.ax7b[1].transAxes,
                              fontsize=7, bbox=dict(facecolor="white", alpha=0.6))

        self.canvas7a.setVisible(True); self.canvas7b.setVisible(True)
        # Légende globale pour toutes les figures (sous la figure complète)
        self.canvas7a.draw_idle(); self.canvas7b.draw_idle()
        
      


    # ----------- Export CSV -----------
    def export_csv(self):
        if not self.S.results:
            self._warn("Rien à exporter — lance la détection (étape 6).")
            return
    
        base = os.path.splitext(os.path.basename(self.S.img_path or 'resultats'))[0]
        folder = os.path.join(os.path.dirname(self.S.img_path or '.'), f"{base}_exports")
        os.makedirs(folder, exist_ok=True)
    
        out_path = os.path.join(folder, f"{base}_fringes.csv")
    
        try:
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    "zone_x","zone_y","zone_w","zone_h",
                    "n_fringes","lambda_local_nm",
                    "L_hauteur_nm","L_largeur_nm","L_largeur_moy_emp_nm"
                ])
                for r in self.S.results:
                    x,y,wz,hz = r['zone']
                    w.writerow([
                        x, y, wz, hz,
                        r['n_fringes'],
                        f"{r['lambda_local']:.6f}",
                        f"{r['L_hauteur']:.6f}",
                        f"{r['L_largeur']:.6f}",
                        f"{r['L_largeur_moy_emp']:.6f}"
                    ])
            self._info(f"💾 CSV exporté: {out_path}")
        except Exception as e:
            self._warn(f"Échec export CSV: {e}")
            # Fallback : proposition de sauvegarde manuelle
            fname,_ = QFileDialog.getSaveFileName(self, "Enregistrer CSV", f"{base}_fringes.csv", "CSV (*.csv)")
            if fname:
                try:
                    with open(fname,'w',newline='',encoding='utf-8') as f:
                        w=csv.writer(f)
                        w.writerow(["zone_x","zone_y","zone_w","zone_h","n_fringes","lambda_local_nm","L_hauteur_nm","L_largeur_nm","L_largeur_moy_emp_nm"])
                        for r in self.S.results:
                            x,y,wz,hz=r['zone']
                            w.writerow([x,y,wz,hz,r['n_fringes'],f"{r['lambda_local']:.6f}",f"{r['L_hauteur']:.6f}",f"{r['L_largeur']:.6f}",f"{r['L_largeur_moy_emp']:.6f}"])
                    self._info(f"💾 CSV exporté: {fname}")
                except Exception as e2:
                    self._warn(f"Échec export CSV (fallback): {e2}")


    def export_images(self):
        """Exporte toutes les figures et images dans un sous-dossier _exports."""
        if self.S.img_path is None:
            self._warn("Charge une image d'abord.")
            return
    
        base = os.path.splitext(os.path.basename(self.S.img_path))[0]
        # 📂 Crée un sous-dossier "nomimage_exports" à côté de l'image source
        folder = os.path.join(os.path.dirname(self.S.img_path), f"{base}_exports")
        os.makedirs(folder, exist_ok=True)
    
        try:
            # Step 2 : FFT + masque + IFFT (preview)
            if hasattr(self, 'fig2'):
                self.fig2.savefig(
                    os.path.join(folder, f"{base}_step2_fft_mask_ifft.png"),
                    dpi=200, bbox_inches='tight'
                )
    
            # Step 4 : FFT + masque final + IFFT
            if hasattr(self, 'fig4') and self.S.img_back is not None:
                self.fig4.savefig(
                    os.path.join(folder, f"{base}_step4_fft_mask_ifft.png"),
                    dpi=200, bbox_inches='tight'
                )
    
            # Masque brut
            if self.S.mask is not None:
                m8 = (self.S.mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, f"{base}_mask.png"), m8)
    
            # IFFT seul
            if self.S.img_back is not None:
                img8 = cv2.normalize(self.S.img_back, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, f"{base}_ifft.png"), img8)
    
            # Step 6 : détection des franges
            if hasattr(self, 'fig6') and self.S.out_vis is not None:
                self.fig6.savefig(
                    os.path.join(folder, f"{base}_step6_ifft_detect.png"),
                    dpi=200, bbox_inches='tight'
                )
                cv2.imwrite(os.path.join(folder, f"{base}_fringes_detected.png"),
                            self.S.out_vis)
    
            # Step 7 : graphiques
            if hasattr(self, 'fig7a'):
                self.fig7a.savefig(
                    os.path.join(folder, f"{base}_step7_scatter.png"),
                    dpi=200, bbox_inches='tight'
                )
            if hasattr(self, 'fig7b'):
                self.fig7b.savefig(
                    os.path.join(folder, f"{base}_step7_hist.png"),
                    dpi=200, bbox_inches='tight'
                )
    
            self._info(f"💾 Images & graphiques exportés dans : {folder}")
    
        except Exception as e:
            self._warn(f"Échec export images : {e}")
    
    
    # ----------- Key bindings -----------
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.stack.currentWidget() is self.page0 and len(self.S.scale_points)==2:
                self.confirm_scale(); return
            self.next_step()
        elif e.key()==Qt.Key_R:
            if self.stack.currentWidget() is self.page2: self.reset_wedge()
            elif self.stack.currentWidget() is self.page3: self.reset_crops()
        elif e.key() == Qt.Key_S and (e.modifiers() & Qt.ControlModifier):
            self.export_images()
        elif e.key()==Qt.Key_Q: self.close()
        else: super().keyPressEvent(e)
            
        
    def go_to_skeleton_tab(self):
        if self.S.roi is None:
            self._warn("Définis une ROI avant la squelettisation.")
            return
        self.stack.setCurrentWidget(self.page_alt)
    
    def build_alt_skeleton(self):
        page = QWidget()
        v = QVBoxLayout(page)
    
        title = QLabel("Méthode alternative — Squelettisation (spatial, sans FFT)")
        title.setStyleSheet("font-size:16px; font-weight:700; color:#0f172a;")
        v.addWidget(title)
    
        self.fig_alt = Figure(figsize=(16, 5))
        self.ax_alt = [
            self.fig_alt.add_subplot(1, 3, 1),
            self.fig_alt.add_subplot(1, 3, 2),
            self.fig_alt.add_subplot(1, 3, 3),
        ]
        self.canvas_alt = FigureCanvas(self.fig_alt)
        self.canvas_alt.setVisible(False)
        v.addWidget(self.canvas_alt)
    
        btn_run = QPushButton("🧬 Lancer squelettisation")
        btn_run.clicked.connect(self.run_alt_skeleton)
        v.addWidget(btn_run)
        
        btn_export = QPushButton("💾 Exporter squelettisation")
        btn_export.clicked.connect(self.export_skeleton_images)
        v.addWidget(btn_export)

    
        btn_back = QPushButton("← Retour à l’étape 1 (ROI)")
        btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page1))
        v.addWidget(btn_back)
    
        return page
    
    def run_alt_skeleton(self):
        if self.S.roi is None:
            self._warn("Définis une ROI avant la squelettisation.")
            return
    
        th, skel = skeletonize_roi(self.S.roi)
        self.S.skel_roi = self.S.roi.copy()
        self.S.skel_bin = th.copy()
        self.S.skel_img = skel.copy()

    
        for ax in self.ax_alt:
            ax.clear()
    
        self.ax_alt[0].imshow(self.S.roi, cmap="gray")
        self.ax_alt[0].set_title("ROI")
        self.ax_alt[0].axis("off")
    
        self.ax_alt[1].imshow(th, cmap="gray")
        self.ax_alt[1].set_title("Binarisation")
        self.ax_alt[1].axis("off")
    
        self.ax_alt[2].imshow(skel, cmap="gray")
        self.ax_alt[2].set_title("Squelette")
        self.ax_alt[2].axis("off")
    
        self.canvas_alt.setVisible(True)
        self.canvas_alt.draw_idle()
    
        self._info("🧬 Squelettisation terminée.")
    
        
        
    def export_skeleton_images(self):
        if not hasattr(self.S, "skel_img") or self.S.skel_img is None:
            self._warn("Aucune squelettisation à exporter.")
            return
    
        if self.S.img_path is None:
            self._warn("Chemin d’image inconnu.")
            return
    
        base = os.path.splitext(os.path.basename(self.S.img_path))[0]
        folder = os.path.join(os.path.dirname(self.S.img_path), f"{base}_exports")
        os.makedirs(folder, exist_ok=True)
    
        try:
            cv2.imwrite(
                os.path.join(folder, f"{base}_skeleton_roi.png"),
                self.S.skel_roi
            )
            cv2.imwrite(
                os.path.join(folder, f"{base}_skeleton_binary.png"),
                self.S.skel_bin
            )
            cv2.imwrite(
                os.path.join(folder, f"{base}_skeleton.png"),
                self.S.skel_img
            )
    
            self._info(f"💾 Images de squelettisation exportées dans : {folder}")
    
        except Exception as e:
            self._warn(f"Erreur export squelettisation : {e}")
    
    
# ================================
# Main
# ================================
if __name__ == '__main__':
    print(">>> Lancement de MainWindow...")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show(); w.raise_(); w.activateWindow()
    print(">>> Fenêtre affichée, attente exec_()")
 
    sys.exit(app.exec_())
    
    