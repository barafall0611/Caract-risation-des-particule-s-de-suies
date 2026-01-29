# ============================================================
# 🔬 ANALYSE MORPHOMÉTRIQUE COMPLÈTE D’AGRÉGATS MET (corrigée)
# ============================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure, morphology

# ============================================================
# 1️⃣ CHARGEMENT DE L’IMAGE
# ============================================================
#img_path = r"C:\Users\bara.fall\Desktop\image traiter\m911-7,8-8,8K.jpg"
img_path = r"C:\Users\bara.fall\Desktop\traiter 04-12-2025\224110\224110 -25kX -0017.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"❌ Impossible de charger l'image : {img_path}")

h, w = img.shape
print(f"📷 Image chargée : {w}x{h} px")

# ============================================================
# 2️⃣ CALIBRATION MANUELLE AVEC IMAGE RÉDUITE (ET EXCLUSION DE LA BARRE)
# ============================================================

def select_scale_bar(image, max_display=1200):
    """
    Affiche une version réduite de l'image pour sélectionner la barre d’échelle,
    puis reconvertit la sélection à la taille réelle.
    Retourne la taille réelle + les coordonnées exactes dans l'image originale.
    """
    h, w = image.shape
    scale_display = min(1.0, max_display / max(w, h))

    # Redimensionner pour affichage
    display_img = cv2.resize(image, (int(w * scale_display), int(h * scale_display)))

    print("🧭 Sélectionne la barre d’échelle sur l'image réduite puis appuie sur ENTER.")
    r = cv2.selectROI("Sélection de la barre d’échelle (image réduite)", display_img, showCrosshair=True)
    cv2.destroyWindow("Sélection de la barre d’échelle (image réduite)")

    x_sel, y_sel, w_sel, h_sel = r
    if w_sel == 0 or h_sel == 0:
        raise ValueError("⚠️ Aucune région sélectionnée pour la barre d’échelle.")

    # Recalcul coordonnées dans l'image originale
    x_real = int(x_sel / scale_display)
    y_real = int(y_sel / scale_display)
    w_real = int(w_sel / scale_display)
    h_real = int(h_sel / scale_display)

    bar_length_px = w_real
    return bar_length_px, (x_real, y_real, w_real, h_real)

# --- Sélection manuelle de la barre ---
bar_length_px, bar_coords = select_scale_bar(img)
print(f"📏 Barre d’échelle mesurée : {bar_length_px:.1f} px")

# --- Saisie de la longueur réelle ---
scale_nm = float(input("👉 Entrez la longueur réelle de la barre (en nm) : "))
px_size_nm = scale_nm / bar_length_px
print(f"✅ Calibration : {px_size_nm:.4f} nm/pixel (barre = {bar_length_px:.1f} px pour {scale_nm} nm)")

# ============================================================
# 3️⃣ PRÉTRAITEMENT DE L’IMAGE
# ============================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)
blur = cv2.GaussianBlur(img_clahe, (51, 51), 0)
highpass = cv2.subtract(img_clahe, blur)

# ============================================================
# 4️⃣ SEGMENTATION DES AGRÉGATS
# ============================================================
_, binary_otsu = cv2.threshold(highpass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = morphology.remove_small_objects(binary_otsu.astype(bool), min_size=400)
binary = morphology.remove_small_holes(binary, area_threshold=50)
mask = (binary * 255).astype(np.uint8)

# Fermeture morphologique pour lisser les contours
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Suppression de la barre d’échelle selon la sélection réelle
x_bar, y_bar, w_bar, h_bar = bar_coords
mask[y_bar:y_bar+h_bar, x_bar:x_bar+w_bar] = 0
print("🚫 Barre d’échelle exclue du masque d'analyse.")

# ============================================================
# 5️⃣ MESURES MORPHOMÉTRIQUES
# ============================================================
props = measure.regionprops_table(
    measure.label(mask),
    properties=("label", "area", "equivalent_diameter",
                "perimeter_crofton", "eccentricity", "solidity", "centroid")
)
df = pd.DataFrame(props)
df.rename(columns={"perimeter_crofton": "perimeter"}, inplace=True)

# Conversion en unités réelles
df["equivalent_diameter_nm"] = df["equivalent_diameter"] * px_size_nm
df["area_nm2"] = df["area"] * (px_size_nm ** 2)

# Circularité (corrigée)
df["circularity"] = np.where(
    df["perimeter"] > 0,
    4 * np.pi * df["area"] / (df["perimeter"] ** 2),
    np.nan
)

# Nettoyage : suppression des valeurs aberrantes
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["circularity"])
df = df[df["circularity"] <= 1.5]  # suppression des outliers >1.5

print(f"✅ {len(df)} agrégats détectés et mesurés après nettoyage")

# ============================================================
# 6️⃣ VISUALISATION SUR IMAGE + COMPARAISON
# ============================================================
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

font_scale = w / 1000.0
thickness_bg = max(2, w // 500)
thickness_fg = max(1, w // 1000)

for region in measure.regionprops(measure.label(mask)):
    y, x = region.centroid
    label = str(region.label)
    cv2.putText(overlay, label, (int(x)+10, int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_bg)
    cv2.putText(overlay, label, (int(x)+10, int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness_fg)

# --- 🧩 Comparaison visuelle triple ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img, cmap="gray")
axs[0].set_title("Image TEM originale")
axs[0].axis("off")

axs[1].imshow(mask, cmap="gray")
axs[1].set_title("Masque sans barre d’échelle")
axs[1].axis("off")

axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axs[2].set_title("Agrégats annotés")
axs[2].axis("off")

plt.tight_layout()
plt.savefig("comparaison_images.png", dpi=300)
plt.show()

# --- Enregistrement des résultats ---
cv2.imwrite("overlay_agregats.png", overlay)
df.to_csv("agregats_mesures.csv", index=False)
print("💾 Résultats enregistrés : overlay_agregats.png / agregats_mesures.csv / comparaison_images.png")

# ============================================================
# 7️⃣ ANALYSE STATISTIQUE ET VISUALISATION
# ============================================================
df = df[df["equivalent_diameter_nm"] > 0]
# --- Statistiques globales ---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)



# --- Statistiques globales ---
print("\n📊 STATISTIQUES GLOBALES")
print(df[["equivalent_diameter_nm", "area_nm2", "circularity", "eccentricity", "solidity"]].describe())

# --- Histogrammes ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.hist(df["equivalent_diameter_nm"], bins=20, color="orange", edgecolor="k")
plt.xlabel("Diamètre équivalent (nm)"); plt.ylabel("Nombre")
plt.title("Distribution des tailles")

plt.subplot(2, 2, 2)
plt.hist(df["circularity"], bins=20, color="skyblue", edgecolor="k")
plt.xlabel("Circularité"); plt.ylabel("Nombre")
plt.title("Distribution des formes")

plt.subplot(2, 2, 3)
plt.hist(df["eccentricity"], bins=20, color="lightgreen", edgecolor="k")
plt.xlabel("Excentricité"); plt.ylabel("Nombre")
plt.title("Distribution de l’allongement")

plt.subplot(2, 2, 4)
plt.hist(df["solidity"], bins=20, color="plum", edgecolor="k")
plt.xlabel("Solidité"); plt.ylabel("Nombre")
plt.title("Distribution de la densité interne")

plt.tight_layout()
plt.show()

# --- Corrélations ---
plt.figure(figsize=(6,5))
plt.scatter(df["equivalent_diameter_nm"], df["circularity"], alpha=0.7, color="teal")
plt.xlabel("Diamètre équivalent (nm)")
plt.ylabel("Circularité")
plt.title("Relation Taille – Forme")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(df["eccentricity"], df["circularity"], alpha=0.7, color="coral")
plt.xlabel("Excentricité")
plt.ylabel("Circularité")
plt.title("Relation Forme – Allongement")
plt.grid(True)
plt.show()

# --- Heatmap de corrélation ---
plt.figure(figsize=(8,6))
corr = df[["equivalent_diameter_nm", "area_nm2", "circularity", "eccentricity", "solidity"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation des paramètres morphométriques")
plt.show()

# --- Boxplots ---
plt.figure(figsize=(10,6))
sns.boxplot(data=df[["equivalent_diameter_nm", "circularity", "eccentricity", "solidity"]], palette="Set2")
plt.title("Résumé statistique des paramètres morphométriques")
plt.xticks(rotation=15)
plt.show()

# --- Distribution granulométrique cumulative ---
sorted_diam = np.sort(df["equivalent_diameter_nm"])
cum_freq = np.arange(1, len(sorted_diam)+1) / len(sorted_diam)

plt.figure(figsize=(7,5))
plt.plot(sorted_diam, cum_freq, color="darkorange")
plt.xlabel("Diamètre équivalent (nm)")
plt.ylabel("Fréquence cumulée")
plt.title("Distribution granulométrique cumulative")
plt.grid(True)
plt.show()

# D10, D50, D90
D10, D50, D90 = np.percentile(sorted_diam, [10, 50, 90])
print(f"📈 D10 = {D10:.2f} nm | D50 = {D50:.2f} nm | D90 = {D90:.2f} nm")

print("\n✅ Analyse morphométrique complète terminée (circularité corrigée).")
