# 🏥 Health-InsurTech — Application Streamlit RGPD

**Auteur** : Hady COULIBALY — Data Analyst & Data Scientist
**Projet** : Prédire les frais de santé de manière éthique

---

## 🗂️ Structure du projet

```
Health_InsurTech/
├── app.py                        ← Application Streamlit principale
├── insurance_notebook.ipynb      ← Notebook analyse + modélisation + biais
├── note_rgpd.docx                ← Note conformité RGPD (1 page)
├── insurance_data.csv            ← Dataset (1 338 lignes × 22 colonnes)
├── requirements.txt              ← Dépendances
├── README.md                     ← Ce fichier
├── .gitignore                    ← Exclusions Git
└── .streamlit/
    ├── config.toml               ← Thème + sécurité
    └── secrets.toml              ← 🔒 NE PAS COMMITER
```

---

## 🚀 Installation locale

```bash
python -m venv venv
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Déploiement Streamlit Cloud

```bash
git add app.py requirements.txt README.md .gitignore .streamlit/config.toml
git commit -m "🚀 Health-InsurTech deploy"
git push -u origin main
```

**Secrets à configurer** (Settings → Secrets) :
```toml
[auth]
username = "admin"
password = "votre_mot_de_passe"
```

---

## 🛡️ Conformité RGPD

| Mesure | Implémentation |
|--------|---------------|
| Consentement | Bannière obligatoire à l'entrée |
| Minimisation | 7 variables identifiantes exclues |
| Sécurité | HTTPS + Auth + Logs |
| Portabilité | Export JSON des simulations |
| Transparence | Modèle interprétable + feature importance |

---

## 📄 Pages de l'application

- **Page 1** — Upload CSV + Dashboard (IMC, âge, frais)
- **Page 2** — Modélisation interprétable + Analyse des biais
- **Page 3** — Formulaire de simulation avec recommandation
- **Page 4** — Conformité RGPD + Logs d'activité

---

## 🤖 Modèles disponibles

- Régression Linéaire (coefficients clairs)
- Arbre de Décision (règles si-alors)
- Random Forest (robuste, feature importance)

**Accès :** défini dans `.streamlit/secrets.toml` (voir « Secrets à configurer »). Aucun identifiant par défaut n'est fourni.
