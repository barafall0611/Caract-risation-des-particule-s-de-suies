# -*- coding: utf-8 -*-
"""
🔬 ANALYSE MORPHOMÉTRIQUE COMPLÈTE AVEC ÉDITION INTERACTIVE ET SAUVEGARDE TOTALE
→ Suppression (rouge), réactivation (vert), ajout automatique (magenta)
→ Sauvegarde automatique des résultats (images, CSV, graphes) dans le dossier d’origine
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure, morphology
import os

# ============================================================
# 1️⃣ Chargement de l'image
# ============================================================
img_path = r"C:\Users\bara.fall\Desktop\image traiter\NMA-2025-010-3-50K.jpg\NMA-2025-010-3-50K.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"❌ Impossible de charger l'image : {img_path}")

h, w = img.shape
base_dir = os.path.dirname(img_path)
base_name = os.path.splitext(os.path.basename(img_path))[0]
print(f"📷 Image chargée : {w}x{h} px")
print(f"📁 Dossier de travail : {base_dir}")

# ============================================================
# 2️⃣ Calibration manuelle
# ============================================================
def select_scale_bar(image, max_display=1200):
    """Sélection manuelle de la barre d’échelle."""
    h, w = image.shape
    scale_display = min(1.0, max_display / max(w, h))
    disp = cv2.resize(image, (int(w*scale_display), int(h*scale_display)))
    print("🧭 Sélectionne la barre d’échelle puis ENTER.")
    r = cv2.selectROI("Barre d’échelle", disp, showCrosshair=True)
    cv2.destroyWindow("Barre d’échelle")
    x, y, w_sel, h_sel = r
    if w_sel == 0:
        raise ValueError("⚠️ Aucune sélection détectée.")
    return int(w_sel/scale_display), (
        int(x/scale_display), int(y/scale_display),
        int(w_sel/scale_display), int(h_sel/scale_display)
    )

bar_px, bar_coords = select_scale_bar(img)
scale_real_nm = float(input("👉 Entre la longueur réelle de la barre (en nm) : "))
px_size_nm = scale_real_nm / bar_px
print(f"✅ Calibration : {px_size_nm:.3f} nm/pixel ({bar_px}px pour {scale_real_nm} nm)")

# ============================================================
# 3️⃣ Suppression de la barre d’échelle
# ============================================================
x_bar, y_bar, w_bar, h_bar = bar_coords
img_no_scale = img.copy()
region = img[max(y_bar-20,0):min(y_bar+h_bar+20,h), max(x_bar-20,0):min(x_bar+w_bar+20,w)]
mean_bg = np.mean(region)
img_no_scale[y_bar:y_bar+h_bar, x_bar:x_bar+w_bar] = int(mean_bg)
img_no_scale = cv2.GaussianBlur(img_no_scale, (9,9), 0)

# ============================================================
# 4️⃣ Prétraitement
# ============================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_eq = clahe.apply(img_no_scale)
_, vesicles = cv2.threshold(img_eq, 250, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
vesicles = cv2.morphologyEx(vesicles, cv2.MORPH_CLOSE, kernel)
vesicles = morphology.remove_small_objects(vesicles.astype(bool), min_size=1)
vesicles = vesicles.astype(np.uint8) * 255

background_mask = cv2.bitwise_not(vesicles)
img_dark = cv2.bitwise_and(img_eq, img_eq, mask=background_mask)

# ============================================================
# 5️⃣ Détection initiale
# ============================================================
blur = cv2.GaussianBlur(img_dark, (3,3), 0)
img_inv = cv2.bitwise_not(blur)
_, binary = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = morphology.remove_small_objects(mask.astype(bool), min_size=50)
mask = morphology.remove_small_holes(mask, area_threshold=80)
mask = mask.astype(np.uint8) * 255
mask[y_bar:y_bar+h_bar, x_bar:x_bar+w_bar] = 0

# ============================================================
# 6️⃣ Filtrage taille
# ============================================================
label_img = measure.label(mask)
props = measure.regionprops(label_img)
mask_filtered = np.zeros_like(mask)
for r in props:
    if 200 < r.area < 10000:
        mask_filtered[label_img == r.label] = 255
mask = mask_filtered

# ============================================================
# 🖱️ 7️⃣ Édition interactive (suppression / ajout automatique)
# ============================================================
img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_display, contours, -1, (0, 255, 0), 1)

label_img = measure.label(mask)
props = measure.regionprops(label_img)
centroids = [(int(p.centroid[1]), int(p.centroid[0]), p.label) for p in props]

mask_editable = mask.copy()
removed_labels = set()

def find_nearest_label(x, y, centroids, radius=20):
    """Trouve le label d’un agrégat proche du clic."""
    closest_lbl, min_dist = None, float('inf')
    for cx, cy, lbl in centroids:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < radius and dist < min_dist:
            closest_lbl, min_dist = lbl, dist
    return closest_lbl

def detect_local_aggregate(image, x, y, window=80):
    """Détecte automatiquement un contour d’agrégat autour d’un clic droit."""
    h, w = image.shape
    x1, x2 = max(0, x-window//2), min(w, x+window//2)
    y1, y2 = max(0, y-window//2), min(h, y+window//2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    roi_eq = clahe.apply(roi)
    roi_blur = cv2.GaussianBlur(roi_eq, (3,3), 0)
    roi_inv = cv2.bitwise_not(roi_blur)
    _, roi_bin = cv2.threshold(roi_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi_bin = morphology.remove_small_objects(roi_bin.astype(bool), min_size=20)
    roi_bin = morphology.remove_small_holes(roi_bin, area_threshold=30)
    roi_bin = roi_bin.astype(np.uint8) * 255
    contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_cnt = min(contours, key=lambda c: cv2.pointPolygonTest(c, (x-x1, y-y1), True)**2)
    mask_local = np.zeros_like(image)
    cv2.drawContours(mask_local[y1:y2, x1:x2], [best_cnt], -1, 255, -1)
    return mask_local, best_cnt, (x1, y1)

def mouse_callback(event, x, y, flags, param):
    global mask_editable, removed_labels, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        lbl = find_nearest_label(x, y, centroids)
        if lbl is not None:
            mask_lbl = (label_img == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if lbl in removed_labels:
                mask_editable[label_img == lbl] = 255
                removed_labels.remove(lbl)
                cv2.drawContours(img_display, cnts, -1, (0, 255, 0), -1)
            else:
                mask_editable[label_img == lbl] = 0
                removed_labels.add(lbl)
                cv2.drawContours(img_display, cnts, -1, (0, 0, 255), -1)
            cv2.imshow("Édition interactive", img_display)
    elif event == cv2.EVENT_RBUTTONDOWN:
        detection = detect_local_aggregate(img, x, y, window=80)
        if detection is not None:
            mask_local, cnt, offset = detection
            mask_editable = cv2.bitwise_or(mask_editable, mask_local)
            cv2.drawContours(img_display, [cnt + np.array([[offset]])], -1, (255, 0, 255), 1)
            print(f"➕ Agrégat ajouté automatiquement ({x},{y})")
        else:
            print("⚠️ Aucun contour détecté autour du clic.")
        cv2.imshow("Édition interactive", img_display)

cv2.namedWindow("Édition interactive", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Édition interactive", 1200, 800)
cv2.setMouseCallback("Édition interactive", mouse_callback)
cv2.imshow("Édition interactive", img_display)

print("\n🖱️ Clic gauche = supprimer/réactiver (Rouge/Vert)")
print("🖱️ Clic droit = ajouter un agrégat (détection locale magenta)")
print("💾 's' = sauvegarder | 'q' = quitter sans sauvegarde")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("💾 Sauvegarde des modifications...")
        cv2.destroyAllWindows()
        break
    elif key == ord('q'):
        print("🚫 Fermeture sans sauvegarde.")
        mask_editable = mask.copy()
        cv2.destroyAllWindows()
        break

# ============================================================
# 8️⃣ Sauvegarde automatique de tous les résultats
# ============================================================
mask_path = os.path.join(base_dir, f"{base_name}_mask_corrige.png")
csv_path = os.path.join(base_dir, f"{base_name}_agregats_corriges.csv")
annot_path = os.path.join(base_dir, f"{base_name}_annot_final.png")
graph_path = os.path.join(base_dir, f"{base_name}_graphiques.png")
corr_path = os.path.join(base_dir, f"{base_name}_correlation.png")
cum_path = os.path.join(base_dir, f"{base_name}_distribution_cumulative.png")

# --- Masque et CSV ---
cv2.imwrite(mask_path, mask_editable)
props_final = measure.regionprops_table(
    measure.label(mask_editable),
    properties=("label", "area", "equivalent_diameter", "perimeter_crofton",
                "eccentricity", "solidity")
)
df_final = pd.DataFrame(props_final)
df_final.rename(columns={"perimeter_crofton": "perimeter"}, inplace=True)
df_final["equivalent_diameter_nm"] = df_final["equivalent_diameter"] * px_size_nm
df_final["area_nm2"] = df_final["area"] * (px_size_nm ** 2)
df_final["circularity"] = np.where(df_final["perimeter"] > 0,
                                   4 * np.pi * df_final["area"] / (df_final["perimeter"] ** 2),
                                   np.nan)
df_final = df_final[df_final["circularity"] <= 1.5]
df_final.to_csv(csv_path, index=False)

# --- Image annotée ---
overlay_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours_final, _ = cv2.findContours(mask_editable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay_final, contours_final, -1, (0,255,0), 1)
cv2.imwrite(annot_path, overlay_final)

# ============================================================
# 9️⃣ Graphiques automatiques
# ============================================================
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
plt.hist(df_final["equivalent_diameter_nm"], bins=25, color="orange", edgecolor="k")
plt.xlabel("Diamètre équivalent (nm)")
plt.ylabel("Nombre")
plt.title("Distribution des tailles")

plt.subplot(2,2,2)
plt.hist(df_final["circularity"], bins=25, color="skyblue", edgecolor="k")
plt.xlabel("Circularité")
plt.ylabel("Nombre")
plt.title("Distribution des formes")

plt.subplot(2,2,3)
plt.hist(df_final["eccentricity"], bins=25, color="lightgreen", edgecolor="k")
plt.xlabel("Excentricité")
plt.ylabel("Nombre")
plt.title("Distribution de l’allongement")

plt.subplot(2,2,4)
plt.hist(df_final["solidity"], bins=25, color="plum", edgecolor="k")
plt.xlabel("Solidité")
plt.ylabel("Nombre")
plt.title("Distribution de la compacité")
plt.tight_layout()
plt.savefig(graph_path, dpi=300)
plt.close()

corr = df_final[["equivalent_diameter_nm", "area_nm2", "circularity", "eccentricity", "solidity"]].corr()
plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation des paramètres morphométriques")
plt.tight_layout()
plt.savefig(corr_path, dpi=300)
plt.close()

sorted_diam = np.sort(df_final["equivalent_diameter_nm"])
cum_freq = np.arange(1, len(sorted_diam)+1)/len(sorted_diam)
plt.figure(figsize=(7,5))
plt.plot(sorted_diam, cum_freq, color="darkorange")
plt.xlabel("Diamètre équivalent (nm)")
plt.ylabel("Fréquence cumulée")
plt.title("Distribution granulométrique cumulative")
plt.grid(True)
plt.tight_layout()
plt.savefig(cum_path, dpi=300)
plt.close()

print(f"\n✅ Tous les résultats ont été sauvegardés dans : {base_dir}")
print(f"📄 {csv_path}")
print(f"🖼️ {mask_path}")
print(f"🖼️ {annot_path}")
print(f"📊 {graph_path}")
print(f"📈 {corr_path}")
print(f"📉 {cum_path}")
print("\n✨ Analyse complète terminée avec export des graphiques.")
