import cv2
import numpy as np
import pandas as pd
import os


# ============================================================
# 1️⃣ Chargement de l'image
# ============================================================
img_path =  r"C:\Users\bara.fall\Desktop\Manips MACLE - 251219 - ICMN NM\NMA 2025 011 TEM (4)\NMA-2025-011 -80kV -150kX -0011.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"❌ Impossible de charger l'image : {img_path}")

H, W = img.shape
base_dir = os.path.dirname(img_path)
base_name = os.path.splitext(os.path.basename(img_path))[0]

print(f"📷 Image chargée : {W}x{H}px")
print(f"📁 Dossier : {base_dir}")


# ============================================================
# 2️⃣ Calibration manuelle (barre d'échelle)
# ============================================================

def select_scale_bar(image, max_display=900):
    h, w = image.shape
    scale = min(1.0, max_display / max(h, w))
    disp = cv2.resize(image, (int(w*scale), int(h*scale)))

    print("🧭 Sélectionne la barre d’échelle puis ENTER.")
    r = cv2.selectROI("Barre d’échelle", disp, showCrosshair=True)
    cv2.destroyWindow("Barre d’échelle")

    x, y, w_sel, h_sel = r
    if w_sel == 0:
        raise ValueError("⚠ Aucune sélection détectée.")

    return int(w_sel/scale), (int(x/scale), int(y/scale),
                              int(w_sel/scale), int(h_sel/scale))


bar_px, bar_coords = select_scale_bar(img)
scale_real_nm = float(input("👉 Longueur réelle de la barre (en nm) : "))
px_size_nm = scale_real_nm / bar_px
print(f"✅ Calibration : {px_size_nm:.4f} nm/pixel")


# ============================================================
# 3️⃣ Suppression de la barre d’échelle
# ============================================================

x_bar, y_bar, w_bar, h_bar = bar_coords
img_no_scale = img.copy()
img_work = img_no_scale.copy()


region = img[max(y_bar-20,0):min(y_bar+h_bar+20,H),
             max(x_bar-20,0):min(x_bar+w_bar+20,W)]
mean_bg = np.mean(region)

img_no_scale[y_bar:y_bar+h_bar, x_bar:x_bar+w_bar] = int(mean_bg)
img_no_scale = cv2.GaussianBlur(img_no_scale, (9,9), 0)


# ============================================================
# 4️⃣ Fonction de réduction pour affichage
# ============================================================

def reduce(img, max_width=900):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    s = max_width / w
    return cv2.resize(img, (int(w*s), int(h*s))), s


# ============================================================
# 5️⃣ Sélection de 3 points
# ============================================================

def select_three_points(image, max_display=900):
    disp, scale = reduce(image, max_display)
    pts = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            print("Point :", (x, y))

    cv2.namedWindow("Clique 3 points")
    cv2.setMouseCallback("Clique 3 points", click)

    print("\n👉 Clique 3 points sur la particule puis 'v'. ESC = quitter.")

    while True:
        temp = cv2.cvtColor(disp.copy(), cv2.COLOR_GRAY2BGR)
        for p in pts:
            cv2.circle(temp, p, 5, (0,0,255), -1)
        cv2.imshow("Clique 3 points", temp)

        key = cv2.waitKey(1)
    

        if key == 27:
            cv2.destroyWindow("Clique 3 points")
            return None, True

        if key == ord('v') and len(pts) >= 3:
            cv2.destroyWindow("Clique 3 points")
            pts_real = [(int(px/scale), int(py/scale)) for px,py in pts[:3]]
            return np.array(pts_real), False


# ============================================================
# 6️⃣ Cercle défini par 3 points
# ============================================================

def circle_from_3_points(p1, p2, p3):
    (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
    temp = x2*x2 + y2*y2
    bc = (x1*x1+y1*y1-temp)/2
    cd = (temp-x3*x3-y3*y3)/2
    det = (x1-x2)*(y2-y3)-(x2-x3)*(y1-y2)
    if abs(det) < 1e-6:
        raise ValueError("⚠ Points alignés → pas de cercle.")
    cx = (bc*(y2-y3) - cd*(y1-y2)) / det
    cy = ((x1-x2)*cd - (x2-x3)*bc) / det
    r  = np.sqrt((cx-x1)**2+(cy-y1)**2)
    return int(cx), int(cy), int(r)


# ============================================================
# 7️⃣ Ajustement manuel du cercle (déplacement)
# ============================================================

drag = False
offset = (0,0)

def adjust_circle(img, cx_init, cy_init, r_init, max_width=900):
    global drag, offset

    img_small, s = reduce(img, max_width)

    cx = int(cx_init * s)
    cy = int(cy_init * s)
    r  = int(r_init * s)

    drag = False

    def mouse(event, x, y, flags, param):
        nonlocal cx, cy
        global drag, offset

        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - cx) < 20 and abs(y - cy) < 20:
                drag = True
                offset = (x-cx, y-cy)

        elif event == cv2.EVENT_MOUSEMOVE and drag:
            cx = x - offset[0]
            cy = y - offset[1]

        elif event == cv2.EVENT_LBUTTONUP:
            drag = False

    cv2.namedWindow("Ajuster cercle")
    cv2.setMouseCallback("Ajuster cercle", mouse)

    print("🟢 Ajuste le cercle → ENTER = valider | r = refaire | ESC = quitter.")

    while True:
        temp = cv2.cvtColor(img_small.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(temp, (cx,cy), r, (0,255,0), 2)
        cv2.circle(temp, (cx,cy), 8, (0,0,255), -1)
        cv2.imshow("Ajuster cercle", temp)

        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyWindow("Ajuster cercle")
            return None

        if key == ord('r'):
            cv2.destroyWindow("Ajuster cercle")
            return None

        if key in (10,13):  # ENTER accepté
            cv2.destroyWindow("Ajuster cercle")
            return int(cx/s), int(cy/s), r_init


# ============================================================
# 8️⃣ Mesures & boucle principale
# ============================================================

results = []

while True:
    pts, stop = select_three_points(img_work)

    if stop:
        break

    cx, cy, r = circle_from_3_points(pts[0], pts[1], pts[2])
    adj = adjust_circle(img_work, cx, cy, r)

    # ⛔ si l'utilisateur annule l'ajustement
    if adj is None:
        continue

    # ✅ PRENDRE LA POSITION AJUSTÉE
    cx, cy, r = adj

    # 🎯 DESSIN FINAL (BONNE POSITION)
    cv2.circle(img_work, (cx, cy), r, 255, 2)
    cv2.circle(img_work, (cx, cy), 4, 255, -1)

    # 📏 MESURES BASÉES SUR LA POSITION AJUSTÉE
    radius_nm   = r * px_size_nm
    diameter_nm = 2 * radius_nm
    area_nm2    = np.pi * radius_nm**2

    results.append([cx, cy, r, radius_nm, diameter_nm, area_nm2])


# ============================================================
# 9️⃣ Table pandas + moyennes + export Excel
# ============================================================

df = pd.DataFrame(results, columns=["cx_px","cy_px","r_px","r_nm","diam_nm","area_nm2"])

print("\n📊 Résultats :")
print(df)

print("\n📈 Moyennes :")
print(df.mean())

excel_path = os.path.join(base_dir, base_name + "_mesures.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n💾 Fichier Excel exporté : {excel_path}")


# ============================================================
# 🔟 Affichage final + sauvegarde image
# ============================================================

display, s = reduce(cv2.cvtColor(img_no_scale, cv2.COLOR_GRAY2BGR), 900)

for row in results:
    cx, cy, r = row[0], row[1], row[2]

    cx_s = int(cx * s)
    cy_s = int(cy * s)
    r_s  = int(r * s)

    cv2.circle(display, (cx_s, cy_s), r_s, (0,255,0), 2)
    cv2.circle(display, (cx_s, cy_s), 6, (0,0,255), -1)

# 💾 Sauvegarde finale
save_path = os.path.join(base_dir, base_name + "_cercles.png")
cv2.imwrite(save_path, display)
print(f"📸 Image finale enregistrée : {save_path}")

cv2.imshow("Cercles finaux validés", display)
cv2.waitKey(0)
cv2.destroyAllWindows()


import matplotlib.pyplot as plt
import seaborn as sns



# ============================================================
# 📊 11️⃣ Distribution des diamètres (Histogramme avec moyenne, sans KDE)
# ============================================================

diameters = df["diam_nm"]
mean_diam = diameters.mean()

plt.figure(figsize=(8,5))

# 🔵 Histogramme seul (sans courbe KDE)
sns.histplot(diameters, bins=10, color="blue")

plt.title("Distribution des diamètres des particules")
plt.xlabel("Diamètre (nm)")
plt.ylabel("Fréquence")

# 🔴 Moyenne affichée en haut à droite
plt.text(
    0.98, 0.92,
    f"Moyenne = {mean_diam:.2f} nm",
    ha='right', va='center',
    transform=plt.gca().transAxes,
    fontsize=12, color="red", fontweight="bold"
)

# 💾 Sauvegarde du graphique
plot_path = os.path.join(base_dir, base_name + "_distribution_diametres.png")
plt.savefig(plot_path, dpi=200)
print(f"📈 Distribution enregistrée : {plot_path}")

plt.show()

