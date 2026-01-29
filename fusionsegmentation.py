# -*- coding: utf-8 -*-
"""
🔬 ANALYSE MORPHOMÉTRIQUE COMPLÈTE AVEC ÉDITION INTERACTIVE ET FUSION FINALE
→ Suppression (rouge), réactivation (vert), ajout automatique (magenta)
→ Suppression multiple par rectangle (Ctrl + clic + glisser)
→ Numérotation rouge dès la première segmentation et sur le résultat final
→ Fusion morphologique finale pour particules nettes et continues
→ Sauvegarde automatique des résultats dans le dossier d’origine
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

img_path =  r"C:\Users\bara.fall\Desktop\Manip\MET-CM20\M911 process\M911--115K-4.jpg"
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
_, vesicles = cv2.threshold(img_eq, 230, 255, cv2.THRESH_BINARY)
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
    if 1 < r.area < 5:
        mask_filtered[label_img == r.label] = 255
mask = mask_filtered

# ============================================================
# 🧾 6bis — Numérotation initiale
# ============================================================
img_initial_annot = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours_init, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_initial_annot, contours_init, -1, (0,255,0), 1)

label_img_init = measure.label(mask)
props_init = measure.regionprops(label_img_init)
font = cv2.FONT_HERSHEY_SIMPLEX
for p in props_init:
    y, x = p.centroid
    cv2.putText(img_initial_annot, str(p.label), (int(x), int(y)),
                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# ============================================================
# 🖱️ 7️⃣ Édition interactive complète
# ============================================================
img_display = img_initial_annot.copy()
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

label_img = measure.label(mask)
props = measure.regionprops(label_img)
centroids = [(int(p.centroid[1]), int(p.centroid[0]), p.label) for p in props]

mask_editable = mask.copy()
removed_labels = set()
added_regions = []
next_added_id = 1
drawing_rect = False
rect_start = None
rect_end = None

def find_nearest_label(x, y, centroids, radius=20):
    closest_lbl, min_dist = None, float('inf')
    for cx, cy, lbl in centroids:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < radius and dist < min_dist:
            closest_lbl, min_dist = lbl, dist
    return closest_lbl

def detect_local_aggregate(image, x, y, window=50):
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
    best_cnt = max(contours, key=cv2.contourArea)
    mask_local = np.zeros_like(image)
    cv2.drawContours(mask_local[y1:y2, x1:x2], [best_cnt], -1, 255, -1)
    return mask_local, best_cnt, (x1, y1)

def mouse_callback(event, x, y, flags, param):
    global mask_editable, removed_labels, img_display, added_regions, next_added_id
    global drawing_rect, rect_start, rect_end

    # --- 🔹 Mode suppression rectangulaire (Ctrl + clic gauche + glisser)
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        drawing_rect = True
        rect_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing_rect:
        temp_img = img_display.copy()
        cv2.rectangle(temp_img, rect_start, (x, y), (0, 0, 255), 1)
        cv2.imshow("Édition interactive", temp_img)

    elif event == cv2.EVENT_LBUTTONUP and drawing_rect:
        drawing_rect = False
        rect_end = (x, y)
        x1, y1 = rect_start
        x2, y2 = rect_end
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        mask_zone = np.zeros_like(mask_editable)
        mask_zone[y_min:y_max, x_min:x_max] = 255
        mask_editable = cv2.bitwise_and(mask_editable, cv2.bitwise_not(mask_zone))
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        print(f"🧹 Zone supprimée : {x_min},{y_min} → {x_max},{y_max}")
        cv2.imshow("Édition interactive", img_display)
        return

    # --- 🔹 Clic gauche simple : suppression ou réactivation individuelle
    if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):
        found_idx = None
        # 🔸 Vérifie d'abord si le clic est sur un agrégat ajouté (magenta)
        for i, reg in enumerate(added_regions):
            cnt, offset = reg["cnt"], reg["offset"]
            if cv2.pointPolygonTest(cnt + np.array([[offset]]), (x, y), False) >= 0:
                found_idx = i
                break

        # 🔸 Si on clique sur un agrégat ajouté → on le supprime
        if found_idx is not None:
            cnt, offset = added_regions[found_idx]["cnt"], added_regions[found_idx]["offset"]
            mask_local = np.zeros_like(mask_editable)
            cv2.drawContours(mask_local, [cnt + np.array([[offset]])], -1, 255, -1)
            mask_editable = cv2.bitwise_and(mask_editable, cv2.bitwise_not(mask_local))
            cv2.drawContours(img_display, [cnt + np.array([[offset]])], -1, (0, 0, 255), -1)
            del added_regions[found_idx]
            print("❌ Agrégat ajouté supprimé")
            cv2.imshow("Édition interactive", img_display)
            return

        # 🔸 Sinon, on clique sur un agrégat détecté → on le supprime ou réactive
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

    # --- 🔹 Clic droit : ajout automatique d’un nouvel agrégat
    elif event == cv2.EVENT_RBUTTONDOWN:
        detection = detect_local_aggregate(img, x, y, window=80)
        if detection is not None:
            mask_local, cnt, offset = detection
            mask_editable = cv2.bitwise_or(mask_editable, mask_local)
            cv2.drawContours(img_display, [cnt + np.array([[offset]])], -1, (255, 0, 255), 1)
            added_regions.append({"id": next_added_id, "cnt": cnt, "offset": offset})
            next_added_id += 1
            print(f"➕ Agrégat ajouté ({x},{y}) — clic gauche pour le supprimer")
        else:
            print("⚠️ Aucun contour détecté autour du clic.")
        cv2.imshow("Édition interactive", img_display)


cv2.namedWindow("Édition interactive", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Édition interactive", 1200, 800)
cv2.setMouseCallback("Édition interactive", mouse_callback)
cv2.imshow("Édition interactive", img_display)

print("\n🖱️ Clic gauche = supprimer/réactiver (Rouge/Vert)")
print("🖱️ Clic droit = ajouter un agrégat (Magenta, supprimable)")
print("🖱️ Ctrl + clic + glisser = supprimer tous les agrégats dans une zone 🔴")
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
# 🧮 Fusion et nettoyage morphologique final
# ============================================================
print("🧩 Fusion des agrégats ajoutés pour créer des particules nettes...")

kernel_fusion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_fused = cv2.morphologyEx(mask_editable, cv2.MORPH_CLOSE, kernel_fusion)

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_fused = cv2.morphologyEx(mask_fused, cv2.MORPH_OPEN, kernel_clean)

mask_fused = morphology.remove_small_objects(mask_fused.astype(bool), min_size=500)
mask_fused = morphology.remove_small_holes(mask_fused, area_threshold=500)
mask_editable = mask_fused.astype(np.uint8) * 255

print("✅ Fusion et nettoyage terminés — contours lissés et particules unifiées.")

# ============================================================
# 8️⃣ Sauvegarde automatique et numérotation finale
# ============================================================
mask_path = os.path.join(base_dir, f"{base_name}_mask_corrige.png")
csv_path = os.path.join(base_dir, f"{base_name}_agregats_corriges.csv")
annot_path = os.path.join(base_dir, f"{base_name}_annot_final.png")

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

overlay_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours_final, _ = cv2.findContours(mask_editable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay_final, contours_final, -1, (0,255,0), 1)

label_img_final = measure.label(mask_editable)
props_numbered = measure.regionprops(label_img_final)
for p in props_numbered:
    y, x = p.centroid
    cv2.putText(overlay_final, str(p.label), (int(x), int(y)),
                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

annot_numbered_path = os.path.join(base_dir, f"{base_name}_annot_num.png")
cv2.imwrite(annot_numbered_path, overlay_final)

print(f"\n✅ Analyse terminée — agrégats fusionnés et nettoyés.")
print(f"📄 Données : {csv_path}")
print(f"🖼️ Masque : {mask_path}")
print(f"🖼️ Annotée finale : {annot_numbered_path}")

# ============================================================
# 9️⃣ Figures combinées avec moyennes
# ============================================================
df = pd.read_csv(csv_path)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Analyse morphométrique — {base_name}", fontsize=14, fontweight='bold')

def plot_hist(ax, data, color, title, xlabel):
    mean_val = np.nanmean(data)
    sns.histplot(data, bins=20, color=color, ax=ax)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)
    ax.text(0.95, 0.90, f"Moyenne = {mean_val:.3f}",
            transform=ax.transAxes, color="red", ha="right", va="top", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fréquence")

plot_hist(axes[0,0], df["equivalent_diameter_nm"], "skyblue", "Taille (nm)", "Diamètre équivalent (nm)")
plot_hist(axes[0,1], df["circularity"], "lightgreen", "Circularité", "Circularité")
plot_hist(axes[1,0], df["eccentricity"], "orange", "Excentricité", "Excentricité")
plot_hist(axes[1,1], df["solidity"], "salmon", "Solidité", "Solidité")

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = os.path.join(base_dir, f"{base_name}-fig.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"📊 Figure combinée enregistrée : {output_path}")
print(f"📈 Moyennes :")
print(f"   • Diamètre équivalent : {df['equivalent_diameter_nm'].mean():.2f} nm")
print(f"   • Circularité : {df['circularity'].mean():.3f}")
print(f"   • Excentricité : {df['eccentricity'].mean():.3f}")
print(f"   • Solidité : {df['solidity'].mean():.3f}")
