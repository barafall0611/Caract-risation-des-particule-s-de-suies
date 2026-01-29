# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:58:17 2025

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
📊 FUSION ET 3 TYPES DE MOYENNES :
→ Moyenne par image
→ Moyenne pondérée (vraie moyenne de toutes les particules)
→ Moyenne non pondérée (chaque image compte autant)
"""

import pandas as pd
import os

fichiers = [

    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-11.5K-17_agregats_corriges.csv",
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-11.5K-20_agregats_corriges.csv",
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-15K-10_agregats_corriges.csv",
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-20K-9_agregats_corriges.csv",
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-27.5K-14_agregats_corriges.csv",
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911-38K-13_agregats_corriges.csv",   
    r"C:\Users\bara.fall\Desktop\Rendu\ech911\M911--115K-4_agregats_corriges.csv",
   
   
]


data_list = []
for fichier in fichiers:
    df = pd.read_csv(fichier)
    df["source"] = os.path.basename(fichier).replace("_agregats_corriges.csv", "")
    data_list.append(df)
    print(f"📄 Fichier chargé : {os.path.basename(fichier)}")

df_global = pd.concat(data_list, ignore_index=True)

# 🔹 Moyenne par image
df_moy = df_global.groupby("source")[[
    "equivalent_diameter_nm", "circularity", "eccentricity", "solidity","area_nm2"
]].mean().reset_index()

# 🔹 Moyenne pondérée (toutes particules)
moyenne_ponderee = {
    "source": "Moyenne_globale_pondérée (particule)",
    "equivalent_diameter_nm": df_global["equivalent_diameter_nm"].mean(),
    "circularity": df_global["circularity"].mean(),
    "eccentricity": df_global["eccentricity"].mean(),
    "solidity": df_global["solidity"].mean(),
    "area_nm2":df_global["area_nm2"].mean()
}

# 🔹 Moyenne non pondérée (chaque image compte autant)
moyenne_non_ponderee = {
    "source": "Moyenne_globale_non_pondérée (moy. des images)",
    "equivalent_diameter_nm": df_moy["equivalent_diameter_nm"].mean(),
    "circularity": df_moy["circularity"].mean(),
    "eccentricity": df_moy["eccentricity"].mean(),
    "solidity": df_moy["solidity"].mean(),
    "area_nm2":df_moy["area_nm2"].mean()
}

# Ajouter au tableau final
df_moy = pd.concat(
    [df_moy,
     pd.DataFrame([moyenne_ponderee]),
     pd.DataFrame([moyenne_non_ponderee])],
    ignore_index=True
)

# 🔹 Arrondir
df_moy = df_moy.round({
    "equivalent_diameter_nm": 3,
    "circularity": 3,
    "eccentricity": 3,
    "solidity": 3
})

# 💾 Sauvegarde
output_path = os.path.join(os.path.dirname(fichiers[0]), "résumé_global.csv")
df_moy.to_csv(output_path, index=False, sep=";", decimal=",")

print(f"\n✅ Résumé enregistré : {output_path}")
print(df_moy)
