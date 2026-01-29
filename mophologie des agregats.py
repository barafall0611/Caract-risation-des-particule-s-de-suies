# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:44:18 2025

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
🔬 ANALYSE MORPHOMÉTRIQUE COMPLÈTE AVEC ÉDITION INTERACTIVE ET SUPPRESSION ZONALE
→ Suppression (rouge), réactivation (vert), ajout automatique (magenta)
→ Suppression de plusieurs agrégats via un rectangle (Ctrl + clic + glisser)
→ Numérotation rouge dès la première segmentation et sur le résultat final
→ Sauvegarde automatique des résultats dans le dossier d’origine
→ Longueur = Feret max, largeur perpendiculaire, rectangle Feret qui suit L et l
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

img_path = r"C:\Users\bara.fall\Desktop\224110 -25kX -0017.jpg"
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
def select_scale_bar(image, max_display=1000):
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

# Masque des vésicules claires (facultatif, pour exclure le fond clair)
_, vesicles = cv2.threshold(img_eq,220, 255, cv2.THRESH_BINARY)
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

# Suppression de la barre d'échelle dans le masque
mask[y_bar:y_bar+h_bar, x_bar:x_bar+w_bar] = 0

# ============================================================
# 6️⃣ Filtrage par taille
# ============================================================
label_img = measure.label(mask)
props = measure.regionprops(label_img)
mask_filtered = np.zeros_like(mask)
for r in props:
    if 9000 < r.area < 1000000:
        mask_filtered[label_img == r.label] = 255
mask = mask_filtered

# ============================================================
# 🧾 Numérotation initiale
# ============================================================
img_initial_annot = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours_init, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_initial_annot, contours_init, -1, (0,255,0), 1)

label_img_init = measure.label(mask)
props_init = measure.regionprops(label_img_init)
font = cv2.FONT_HERSHEY_SIMPLEX
for p in props_init:
    y_c, x_c = p.centroid
    cv2.putText(img_initial_annot, str(p.label), (int(x_c), int(y_c)),
                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# ============================================================
# 🖱️ 7️⃣ Édition interactive
# ============================================================

img_display = img_initial_annot.copy()
label_img = measure.label(mask)
props = measure.regionprops(label_img)
centroids = [(int(p.centroid[1]), int(p.centroid[0]), p.label) for p in props]

mask_editable = mask.copy()
removed_labels = set()
added_regions = []
next_added_id = 1
drawing_rect = False
rect_start = None

def find_nearest_label(x, y, centroids, radius=20):
    closest_lbl, min_dist = None, float('inf')
    for cx, cy, lbl in centroids:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < radius and dist < min_dist:
            closest_lbl, min_dist = lbl, dist
    return closest_lbl

def detect_local_aggregate(image, x, y, window=200):
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
    global mask_editable, removed_labels, img_display, added_regions, next_added_id
    global drawing_rect, rect_start

    # --- Rectangle de suppression (Ctrl + clic gauche)
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        drawing_rect = True
        rect_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing_rect:
        temp_img = img_display.copy()
        cv2.rectangle(temp_img, rect_start, (x, y), (0, 0, 255), 1)
        cv2.imshow("Édition interactive", temp_img)

    elif event == cv2.EVENT_LBUTTONUP and drawing_rect:
        drawing_rect = False
        x1, y1 = rect_start
        x2, y2 = x, y
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        mask_zone = np.zeros_like(mask_editable)
        mask_zone[y_min:y_max, x_min:x_max] = 255
        mask_editable = cv2.bitwise_and(mask_editable, cv2.bitwise_not(mask_zone))
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        print(f"🧹 Zone supprimée : {x_min},{y_min} → {x_max},{y_max}")
        cv2.imshow("Édition interactive", img_display)
        return

    # --- 🔹 Clic gauche simple : suppression ou réactivation
    if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):
        # 1️⃣ D'abord : vérifier si on clique sur un agrégat ajouté (magenta)
        found_idx = None
        for i, reg in enumerate(added_regions):
            cnt = reg["cnt"]  # contour déjà en coordonnées globales
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                found_idx = i
                break

        if found_idx is not None:
            cnt = added_regions[found_idx]["cnt"]
            mask_local = np.zeros_like(mask_editable)
            cv2.drawContours(mask_local, [cnt], -1, 255, -1)
            mask_editable = cv2.bitwise_and(mask_editable, cv2.bitwise_not(mask_local))
            cv2.drawContours(img_display, [cnt], -1, (0, 0, 255), -1)
            del added_regions[found_idx]
            print("❌ Agrégat ajouté supprimé")
            cv2.imshow("Édition interactive", img_display)
            return

        # 2️⃣ Sinon : gestion classique des labels existants
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


    # --- 🔹 Clic droit : ajout automatique
    elif event == cv2.EVENT_RBUTTONDOWN:
        detection = detect_local_aggregate(img, x, y, window=80)
        if detection is not None:
            mask_local, cnt, offset = detection
            mask_editable = cv2.bitwise_or(mask_editable, mask_local)

            # Conversion du contour en coordonnées globales
            cnt_global = cnt + np.array([[offset]])

            # Tracé en magenta
            cv2.drawContours(img_display, [cnt_global], -1, (255, 0, 255), 1)

            # Sauvegarde pour suppression future
            added_regions.append({"id": next_added_id, "cnt": cnt_global})
            next_added_id += 1

            print(f"➕ Agrégat ajouté ({x},{y}) — clic gauche pour le supprimer")
        else:
            print("⚠️ Aucun contour détecté autour du clic.")
        cv2.imshow("Édition interactive", img_display)


cv2.namedWindow("Édition interactive", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Édition interactive", 1200, 800)
cv2.setMouseCallback("Édition interactive", mouse_callback)
cv2.imshow("Édition interactive", img_display)

print("\n🖱️ Clic gauche = supprimer/réactiver")
print("🖱️ Clic droit = ajouter un agrégat")
print("🖱️ Ctrl + clic + glisser = suppression zonale")
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
# 8️⃣ Calcul Feret + sauvegardes
# ============================================================

mask_path = os.path.join(base_dir, f"{base_name}_mask_corrige.png")
csv_path = os.path.join(base_dir, f"{base_name}_agregats_corriges.csv")
annot_numbered_path = os.path.join(base_dir, f"{base_name}_annot_num.png")
fig_path = os.path.join(base_dir, f"{base_name}_fig.png")

cv2.imwrite(mask_path, mask_editable)

label_img_final = measure.label(mask_editable)

# ------------------------------------------
# ⚡ Fonction Feret MAX (Convex Hull)
# ------------------------------------------
def feret_max(cnt):
    """
    Feret max (diamètre maximal réel) + 2 points extrêmes.
    """
    pts = cnt.reshape(-1, 2)
    hull = cv2.convexHull(pts, returnPoints=True).reshape(-1, 2)
    n = len(hull)

    if n < 2:
        return 0.0, hull[0], hull[0]

    max_d2 = -1.0
    p1_best, p2_best = hull[0], hull[1]

    for i in range(n):
        for j in range(i+1, n):
            d2 = np.sum((hull[i] - hull[j])**2)
            if d2 > max_d2:
                max_d2 = d2
                p1_best, p2_best = hull[i], hull[j]

    return np.sqrt(max_d2), p1_best, p2_best

# ------------------------------------------
# ⚡ Largeur perpendiculaire au Feret max
# ------------------------------------------
def feret_min_perp(cnt, p1_max, p2_max):
    """
    Largeur = distance minimale mesurée dans la direction
    perpendiculaire au Feret max.

    Retourne :
      - largeur (pixels)
      - point A, point B (extrêmes en largeur)
    """
    pts = cnt.reshape(-1, 2).astype(float)

    # Direction du Feret max
    v = (p2_max - p1_max).astype(float)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return 0.0, p1_max, p2_max
    v /= nrm

    # Vecteur perpendiculaire
    n = np.array([-v[1], v[0]])

    # Projection de tous les points du contour sur n
    projections = np.dot(pts, n)

    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)

    pA = pts[min_idx]
    pB = pts[max_idx]

    width_px = projections[max_idx] - projections[min_idx]

    return abs(width_px), pA.astype(int), pB.astype(int)

# ------------------------------------------
# 🔵 Analyse finale
# ------------------------------------------
overlay_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
results = []

max_label = label_img_final.max()
for lbl in range(1, max_label + 1):

    region_mask = (label_img_final == lbl).astype(np.uint8)
    cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue

    cnt = cnts[0]

    # 1️⃣ Feret max (longueur réelle)
    L_px, p1, p2 = feret_max(cnt)
    L_nm = L_px * px_size_nm

    # 2️⃣ Largeur perpendiculaire au Feret max
    l_px, wp1, wp2 = feret_min_perp(cnt, p1, p2)
    l_nm = l_px * px_size_nm if l_px > 0 else np.nan

    ratio = L_nm / l_nm if l_nm and l_nm > 0 else np.nan

    # 3️⃣ Aire et circularité
    area_px = np.sum(region_mask)
    area_nm2 = area_px * (px_size_nm ** 2)
    per = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area_px / (per * per) if per > 0 else np.nan

    # Enregistrement
    results.append({
        "label": lbl,
        "feret_max_nm": L_nm,
        "feret_min_nm": l_nm,
        "rapport_L_sur_l": ratio,
        "area_nm2": area_nm2,
        "circularity": circularity
    })

    # -------------------- TRACÉ SUR IMAGE FINALE --------------------

    # Vecteurs unité longueur / largeur
    v = (p2 - p1).astype(float)
    v /= np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1.0
    n = np.array([-v[1], v[0]])

    # Centre du contour
    center = np.mean(cnt.reshape(-1,2), axis=0)

    half_L = L_px / 2.0
    half_l = l_px / 2.0

    # Coins du rectangle Feret (suivant L et l)
    pA = center + v*half_L + n*half_l
    pB = center + v*half_L - n*half_l
    pC = center - v*half_L - n*half_l
    pD = center - v*half_L + n*half_l

    boxF = np.int32([pA, pB, pC, pD])
    # Rectangle Feret en orange
    cv2.drawContours(overlay_final, [boxF], -1, (0, 165, 255), 2)

    # ✨ TRAIT FÉRET MAX EN JAUNE ✨
    p1_draw = tuple(map(int, p1))
    p2_draw = tuple(map(int, p2))
    cv2.line(overlay_final, p1_draw, p2_draw, (0, 255, 255), 2)

    # ✨ TRAIT LARGEUR (perpendiculaire) EN CYAN ✨
    wp1_draw = tuple(map(int, wp1))
    wp2_draw = tuple(map(int, wp2))
    cv2.line(overlay_final, wp1_draw, wp2_draw, (255, 255, 0), 2)

    # Numéro en rouge au centre
    cx, cy = center
    cv2.putText(overlay_final, str(lbl), (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

# Sauvegarde CSV
df_final = pd.DataFrame(results)
df_final.to_csv(csv_path, index=False)

# Sauvegarde image annotée finale
cv2.imwrite(annot_numbered_path, overlay_final)

print(f"📄 Données enregistrées : {csv_path}")
print(f"🖼️ Masque final :        {mask_path}")
print(f"🖼️ Annotée finale :      {annot_numbered_path}")

# ============================================================
# 🔟 Figure combinée
# ============================================================

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Analyse morphométrique — {base_name}", fontsize=14, fontweight='bold')

# 1. Feret max
mean_L = df["feret_max_nm"].mean()
sns.histplot(df["feret_max_nm"], bins=20, ax=axes[0,0], color="skyblue")
axes[0,0].axvline(mean_L, color="red", linestyle="--")
axes[0,0].set_title("Longueur (Feret max, nm)")
axes[0,0].set_xlabel("Feret max (nm)")
axes[0,0].text(0.95, 0.9, f"µ = {mean_L:.1f} nm",
               transform=axes[0,0].transAxes, ha="right")

# 2. Largeur perpendiculaire
mean_l = df["feret_min_nm"].mean()
sns.histplot(df["feret_min_nm"], bins=20, ax=axes[0,1], color="lightgreen")
axes[0,1].axvline(mean_l, color="red", linestyle="--")
axes[0,1].set_title("Largeur (perp. à L, nm)")
axes[0,1].set_xlabel("Largeur (nm)")
axes[0,1].text(0.95, 0.9, f"µ = {mean_l:.1f} nm",
               transform=axes[0,1].transAxes, ha="right")

# 3. Rapport L/l
mean_ratio = df["rapport_L_sur_l"].mean()
sns.histplot(df["rapport_L_sur_l"], bins=20, ax=axes[1,0], color="orange")
axes[1,0].axvline(mean_ratio, color="red", linestyle="--")
axes[1,0].set_title("Rapport L / l")
axes[1,0].set_xlabel("L / l")
axes[1,0].text(0.95, 0.9, f"µ = {mean_ratio:.2f}",
               transform=axes[1,0].transAxes, ha="right")

# 4. Circularité
mean_circ = df["circularity"].mean()
sns.histplot(df["circularity"], bins=20, ax=axes[1,1], color="salmon")
axes[1,1].axvline(mean_circ, color="red", linestyle="--")
axes[1,1].set_title("Circularité")
axes[1,1].set_xlabel("Circularité")
axes[1,1].text(0.95, 0.9, f"µ = {mean_circ:.3f}",
               transform=axes[1,1].transAxes, ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"📊 Figure combinée enregistrée : {fig_path}")
print("\n🎉 Calculs terminés avec rectangle Feret aligné sur longueur et largeur !")




