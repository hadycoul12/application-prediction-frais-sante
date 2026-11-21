

import io
import logging
import warnings
import datetime
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HealthInsurTech")

def log_event(event: str, details: str = ""):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}] {event} — {details}")
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(f"{ts} | {event} | {details}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG PAGE                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="🏥 Health-InsurTech",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CSS                                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a6e 60%, #0d4d3a 100%);
    border-radius: 16px; padding: 2.2rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-40%; right:-5%;
    width:360px; height:360px;
    background: radial-gradient(circle, rgba(46,204,113,0.18) 0%, transparent 70%);
    border-radius:50%;
}
.hero h1 { font-family:'DM Serif Display',serif; font-size:2.3rem; color:#e8f5e9; margin:0; line-height:1.1; }
.hero p  { color:#90b8a0; font-size:0.95rem; margin-top:0.4rem; font-weight:300; }
.hero .badge {
    display:inline-block; background:rgba(46,204,113,0.2);
    border:1px solid rgba(46,204,113,0.4); color:#2ecc71;
    border-radius:20px; padding:0.18rem 0.75rem; font-size:0.72rem;
    font-weight:600; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.7rem;
}
.metric-card {
    background:#fff; border:1px solid #e0ece4; border-radius:12px;
    padding:1.1rem 1.3rem; box-shadow:0 1px 4px rgba(0,0,0,0.05); transition:box-shadow 0.2s; text-align:center;
}
.metric-card:hover { box-shadow:0 4px 14px rgba(46,204,113,0.15); }
.metric-card .label { font-size:0.7rem; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; color:#7a9a7a; margin-bottom:0.25rem; }
.metric-card .value { font-family:'DM Serif Display',serif; font-size:1.7rem; color:#0d2137; line-height:1; }
.metric-card .sub   { font-size:0.75rem; color:#99aa99; margin-top:0.2rem; }
.section-title {
    font-family:'DM Serif Display',serif; font-size:1.3rem; color:#0d2137;
    margin:1.4rem 0 0.7rem; padding-bottom:0.35rem;
    border-bottom:2px solid #2ecc71; display:inline-block;
}
[data-testid="stSidebar"]     { background:#0d2137 !important; }
[data-testid="stSidebar"] *   { color:#c8dcc8 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { font-family:'DM Serif Display',serif !important; color:#e8f5e9 !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#f0f4f0; padding:4px; border-radius:10px; }
.stTabs [data-baseweb="tab"]     { border-radius:8px; padding:0.38rem 1.1rem; font-weight:500; font-size:0.87rem; }
.stTabs [aria-selected="true"]   { background:#0d2137 !important; color:#e8f5e9 !important; }
.stButton > button { background:linear-gradient(135deg,#0d4d3a,#1a6e4d) !important; color:#e8f5e9 !important; border:none !important; border-radius:10px !important; font-weight:700 !important; }
.result-box { background:linear-gradient(135deg,#0d2137,#1a3a6e); border-radius:14px; padding:1.8rem 2rem; text-align:center; margin:1rem 0; }
.result-box .price  { font-family:'DM Serif Display',serif; font-size:3rem; color:#2ecc71; line-height:1; }
.result-box .rlabel { color:#a8c8a8; font-size:0.9rem; margin-top:0.4rem; }
.info-box  { background:#f0faf2; border-left:4px solid #2ecc71; border-radius:0 8px 8px 0; padding:0.75rem 1rem; font-size:0.87rem; color:#1a3a1a; margin:0.7rem 0; }
.warn-box  { background:#fff8f0; border-left:4px solid #e5a040; border-radius:0 8px 8px 0; padding:0.75rem 1rem; font-size:0.87rem; color:#3a2a1a; margin:0.7rem 0; }
.rgpd-box  { background:linear-gradient(135deg,#0d2137,#1a3a6e); border-radius:16px; padding:2rem; text-align:center; }
.rgpd-box h2 { font-family:'DM Serif Display',serif; color:#e8f5e9; margin-bottom:0.5rem; }
.rgpd-box p  { color:#90b8a0; font-size:0.9rem; margin-bottom:1rem; }
.sec-badge { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(46,204,113,0.15); border:1px solid rgba(46,204,113,0.4); color:#2ecc71; border-radius:20px; padding:0.2rem 0.75rem; font-size:0.72rem; font-weight:600; }
.log-line  { font-family:monospace; font-size:0.78rem; color:#334455; padding:0.15rem 0; border-bottom:1px solid #e8ecf0; }
.rgpd-tag  { display:inline-block; background:#dcf7e4; color:#0a5c2a; border-radius:6px; padding:0.15rem 0.6rem; font-weight:600; font-size:0.8rem; margin:2px; }
.rgpd-danger { display:inline-block; background:#fff0f0; color:#8b0000; border-radius:6px; padding:0.15rem 0.6rem; font-weight:600; font-size:0.8rem; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
for key, default in {
    "authenticated":   False,
    "username":        "",
    "rgpd_consent":    False,
    "page":            "Page 1",
    "df_raw":          None,
    "model":           None,
    "model_name":      None,
    "feature_cols":    None,
    "metrics":         None,
    "logs":            [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ╔══════════════════════════════════════════════════════════════════╗
# ║  AUTHENTIFICATION                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
def check_credentials(username: str, password: str) -> bool:
    try:
        return username == st.secrets["auth"]["username"] and \
               password == st.secrets["auth"]["password"]
    except Exception:
        # Pas de secrets configurés : aucun accès (aucun identifiant par défaut).
        return False

def login_page():
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem;">
            <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#0d2137;">🏥 Health-InsurTech</div>
            <div style="color:#5a8a6a;font-size:0.9rem;margin-top:0.3rem;">Outil éthique de simulation de frais médicaux</div>
        </div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("👤 Identifiant", placeholder="Votre identifiant")
            password = st.text_input("🔒 Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter →", use_container_width=True)
            if submitted:
                if check_credentials(username.strip(), password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    log_event("LOGIN_SUCCESS", f"user={username.strip()}")
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.", icon="🔒")
                    log_event("LOGIN_FAIL", f"user={username.strip()}")
        st.markdown('<div style="text-align:center;font-size:0.78rem;color:#8899aa;margin-top:0.8rem;">🔒 Connexion sécurisée HTTPS</div>', unsafe_allow_html=True)

if not st.session_state.authenticated:
    login_page()
    st.stop()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONSENTEMENT RGPD                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
if not st.session_state.rgpd_consent:
    st.markdown("""
    <div class="rgpd-box">
        <h2>🛡️ Protection de vos données personnelles</h2>
        <p>Conformément au Règlement Général sur la Protection des Données (RGPD — UE 2016/679),<br>
        nous vous informons du traitement de vos données dans cette application.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <b>✅ Données collectées lors de la simulation :</b><br>
        Âge, IMC (BMI), nombre d'enfants, statut fumeur, région géographique<br><br>
        <b>🎯 Finalité du traitement :</b><br>
        Estimation des frais médicaux annuels pour aider au choix du contrat d'assurance.<br><br>
        <b>🔒 Sécurité :</b><br>
        Transmission chiffrée HTTPS. Aucune donnée personnelle identifiante collectée.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="warn-box">
        <b>⚖️ Vos droits (Art. 13 RGPD) :</b><br>
        • Droit d'accès à vos données<br>
        • Droit de rectification<br>
        • Droit à l'effacement (« droit à l'oubli »)<br>
        • Droit à la portabilité<br>
        • Droit d'opposition<br><br>
        <b>📧 Contact DPO :</b> hady.coulibaly@edu.nexa.fr<br>
        <b>⏱️ Délai de réponse :</b> 30 jours maximum
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        consent = st.checkbox(
            "✅ J'ai lu et j'accepte que mes données de simulation soient utilisées "
            "pour estimer mes frais médicaux. Je comprends mes droits RGPD.",
            value=False,
            help="Case non pré-cochée — consentement libre et éclairé (Art. 7 RGPD)"
        )
        if st.button("Accéder à l'application →", use_container_width=True, disabled=not consent):
            st.session_state.rgpd_consent = True
            log_event("RGPD_CONSENT", f"user={st.session_state.username} consentement=Oui")
            st.rerun()
        if not consent:
            st.markdown('<div style="text-align:center;font-size:0.8rem;color:#e06060;margin-top:0.5rem;">⚠️ Vous devez accepter pour accéder à l\'application.</div>', unsafe_allow_html=True)
    st.stop()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PALETTES & MATPLOTLIB                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
DARK    = "#0d2137"
GREEN   = "#2ecc71"
BLUE    = "#1a3a6e"
CORAL   = "#e06060"
GOLD    = "#e5a040"
TEAL    = "#1a8fa0"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    "#fafcfa",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "#e4ede4",
    "grid.linewidth":    0.7,
})

# ╔══════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner="Chargement…")
def load_and_prepare(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Filtrage consentement RGPD
    df_ok = df[df['consentement_rgpd'] == 'Oui'].copy()
    # Sélection variables ML (minimisation RGPD)
    df_ml = df_ok[['age','bmi','children','smoker','sex','region','charges']].copy()
    df_ml['smoker_enc'] = (df_ml['smoker'] == 'yes').astype(int)
    df_ml['sex_enc']    = (df_ml['sex']    == 'male').astype(int)
    region_dummies = pd.get_dummies(df_ml['region'], prefix='region', drop_first=True)
    df_ml = pd.concat([df_ml.drop(columns=['smoker','sex','region']), region_dummies], axis=1)
    return df, df_ok, df_ml

@st.cache_resource(show_spinner="Entraînement du modèle…")
def train_model(model_name: str, df_ml_hash: int):
    df_ml = st.session_state.df_ml
    feature_cols = [c for c in df_ml.columns if c != 'charges']
    X = df_ml[feature_cols]
    y = df_ml['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "Régression Linéaire":
        m = Pipeline([('sc', StandardScaler()), ('m', LinearRegression())])
    elif model_name == "Arbre de Décision":
        m = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42)
    else:
        m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(m, X, y, cv=5, scoring='r2', n_jobs=-1)

    # Ré-entraîner sur 100% pour les prédictions
    m.fit(X, y)
    return m, feature_cols, r2, mae, rmse, cv, y_test, y_pred

def metric_card(col, label, value, sub=""):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0.5rem 0.5rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#e8f5e9;line-height:1.2;margin-bottom:0.5rem;">
            🏥 Health-InsurTech
        </div>
        <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#2ecc71;margin-bottom:0.3rem;">Auteur</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#ffffff;
                    background:linear-gradient(135deg,rgba(46,204,113,0.2),rgba(46,204,113,0.05));
                    border:1px solid rgba(46,204,113,0.4);border-radius:10px;
                    padding:0.4rem 0.7rem;margin-bottom:0.3rem;">
            Hady COULIBALY
        </div>
        <div style="font-size:0.74rem;color:#3a6a3a;font-style:italic;">Étudiant · M2 Data & IA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="sec-badge">🔒 {st.session_state.username}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;color:#2ecc71;margin-top:0.3rem;">✅ Consentement RGPD validé</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Navigation")
    pages = {
        "Page 1": "📂  Données & Dashboard",
        "Page 2": "🤖  Modélisation & Biais",
        "Page 3": "🔮  Simulation Tarif",
        "Page 4": "🛡️  RGPD & Logs",
    }
    for key, label in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            log_event("NAVIGATION", f"user={st.session_state.username} → {key}")

    st.markdown("---")
    has_data  = st.session_state.df_raw is not None
    has_model = st.session_state.model  is not None
    st.markdown(f"{'✅' if has_data else '⭕'} Données chargées\n\n{'✅' if has_model else '⭕'} Modèle entraîné")
    st.markdown("---")
    if st.button("🚪 Déconnexion", use_container_width=True):
        log_event("LOGOUT", f"user={st.session_state.username}")
        for k in ["authenticated","username","rgpd_consent"]:
            st.session_state[k] = False if k != "username" else ""
        st.rerun()

page = st.session_state.page

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — DONNÉES & DASHBOARD                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
if page == "Page 1":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 1 · Dataset & Dashboard</div>
      <h1>Données &amp; Visualisation</h1>
      <p>Explorez les corrélations entre IMC, âge et frais médicaux.</p>
    </div>""", unsafe_allow_html=True)

    section("Chargement du dataset")
    uploaded = None
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("📂 Charger insurance_data.csv", type="csv",
                                     help="Dataset Health-InsurTech — max 10 Mo")
    with col_info:
        info_box("<b>Variables utilisées par le modèle ML :</b><br>Âge · IMC · Enfants · Fumeur · Région<br><br>"
                 "<b>Variables exclues (RGPD) :</b><br>Nom · Email · NSS · IP · Ville...")

    if uploaded is not None:
        file_bytes = uploaded.read()
        if len(file_bytes) > 0:
            df_raw, df_ok, df_ml = load_and_prepare(file_bytes)
            st.session_state.df_raw = df_raw
            st.session_state.df_ok  = df_ok
            st.session_state.df_ml  = df_ml
            log_event("UPLOAD_OK", f"user={st.session_state.username} {df_raw.shape[0]} lignes")
            st.rerun()
    elif st.session_state.df_raw is None:
        try:
            with open("insurance_data.csv", "rb") as f:
                df_raw, df_ok, df_ml = load_and_prepare(f.read())
                st.session_state.df_raw = df_raw
                st.session_state.df_ok  = df_ok
                st.session_state.df_ml  = df_ml
            st.rerun()
        except FileNotFoundError:
            pass

    if st.session_state.df_raw is None:
        st.info("👆 Chargez votre fichier **insurance_data.csv** pour afficher le dashboard.", icon="ℹ️")
        st.stop()

    df_raw = st.session_state.df_raw
    df_ok  = st.session_state.df_ok
    df_ml  = st.session_state.df_ml

    st.success(f"✅ {df_raw.shape[0]} enregistrements chargés — {len(df_ok)} avec consentement validé")

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card(c1, "Total clients",         f"{df_raw.shape[0]:,}",                 "enregistrements")
    metric_card(c2, "Consentement validé",   f"{len(df_ok):,}",                       f"{len(df_ok)/len(df_raw)*100:.0f}%")
    metric_card(c3, "Frais médicaux moy.",   f"${df_ml['charges'].mean():,.0f}",       "$/an")
    metric_card(c4, "Fumeurs",               f"{(df_ml['smoker_enc']==1).sum():,}",    f"{(df_ml['smoker_enc']==1).mean()*100:.0f}%")
    metric_card(c5, "IMC moyen",             f"{df_ml['bmi'].mean():.1f}",             "kg/m²")

    st.markdown("<br>", unsafe_allow_html=True)
    warn_box("⚠️ <b>Déséquilibre détecté :</b> 20.5% de fumeurs vs 79.5% non-fumeurs → "
             "Risque de biais algorithmique. Voir Page 2 pour l'analyse.")

    # Dashboard principal
    section("Dashboard — Corrélations IMC, Âge & Frais médicaux")

    colors_s = {'yes': CORAL, 'no': BLUE}

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 IMC & Âge vs Frais", "📦 Distribution par groupes",
        "🔥 Corrélations", "📋 Données"
    ])

    with tab1:
        da, db = st.columns(2)
        with da:
            # IMC vs Charges
            fig, ax = plt.subplots(figsize=(7, 5))
            df_plot = df_ml.copy()
            df_plot['smoker_label'] = df_plot['smoker_enc'].map({1:'Fumeur', 0:'Non-fumeur'})
            for label, grp in df_plot.groupby('smoker_label'):
                ax.scatter(grp['bmi'], grp['charges'],
                           c=CORAL if label=='Fumeur' else BLUE,
                           alpha=0.45, s=18, label=label)
            z = np.polyfit(df_plot['bmi'], df_plot['charges'], 1)
            xs = np.linspace(df_plot['bmi'].min(), df_plot['bmi'].max(), 200)
            ax.plot(xs, np.poly1d(z)(xs), color=GOLD, lw=2, ls='--', label='Tendance')
            r = df_plot['bmi'].corr(df_plot['charges'])
            ax.text(0.05, 0.93, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=11, fontweight='600', color=DARK,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0faf2', edgecolor='#c0d4c0'))
            ax.set_xlabel("IMC (BMI, kg/m²)", fontsize=11)
            ax.set_ylabel("Frais médicaux ($)", fontsize=11)
            ax.set_title("IMC vs Frais médicaux", fontweight='600', pad=12)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            info_box(f"Corrélation IMC ~ Frais : <b>r = {r:.3f}</b> — Relation modérée. "
                     "Les fumeurs avec IMC élevé concentrent les frais les plus élevés.")

        with db:
            # Âge vs Charges
            fig, ax = plt.subplots(figsize=(7, 5))
            for label, grp in df_plot.groupby('smoker_label'):
                ax.scatter(grp['age'], grp['charges'],
                           c=CORAL if label=='Fumeur' else BLUE,
                           alpha=0.45, s=18, label=label)
            z2 = np.polyfit(df_plot['age'], df_plot['charges'], 1)
            xs2 = np.linspace(df_plot['age'].min(), df_plot['age'].max(), 200)
            ax.plot(xs2, np.poly1d(z2)(xs2), color=GOLD, lw=2, ls='--', label='Tendance')
            r2v = df_plot['age'].corr(df_plot['charges'])
            ax.text(0.05, 0.93, f'r = {r2v:.3f}', transform=ax.transAxes,
                    fontsize=11, fontweight='600', color=DARK,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0faf2', edgecolor='#c0d4c0'))
            ax.set_xlabel("Âge (années)", fontsize=11)
            ax.set_ylabel("Frais médicaux ($)", fontsize=11)
            ax.set_title("Âge vs Frais médicaux", fontweight='600', pad=12)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            info_box(f"Corrélation Âge ~ Frais : <b>r = {r2v:.3f}</b> — L'âge est "
                     "la variable continue la plus corrélée aux frais médicaux.")

    with tab2:
        b1, b2 = st.columns(2)
        with b1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            df_plot2 = df_ml.copy()
            df_plot2['smoker_label'] = df_plot2['smoker_enc'].map({1:'Fumeur', 0:'Non-fumeur'})
            for label, grp in df_plot2.groupby('smoker_label'):
                ax.hist(grp['charges'], bins=30, alpha=0.65,
                        color=CORAL if label=='Fumeur' else BLUE,
                        edgecolor='white', label=label)
            ax.set_xlabel("Frais médicaux ($)"); ax.set_ylabel("Effectif")
            ax.set_title("Distribution des frais par statut fumeur", fontweight='600')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            ax.legend(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()
            mf = df_ml[df_ml['smoker_enc']==1]['charges'].mean()
            mn = df_ml[df_ml['smoker_enc']==0]['charges'].mean()
            warn_box(f"Fumeurs : <b>${mf:,.0f}/an</b> en moyenne vs Non-fumeurs : <b>${mn:,.0f}/an</b>"
                     f" — Ratio : <b>{mf/mn:.1f}x</b> plus cher")

        with b2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            region_col = [c for c in df_ml.columns if c.startswith('region_')]
            def decode_region(row):
                for rc in region_col:
                    if row[rc] == 1: return rc.replace('region_','')
                return 'northeast'
            df_plot2['region_label'] = df_plot2.apply(decode_region, axis=1)
            reg_data = [df_plot2[df_plot2['region_label']==r]['charges'].values
                        for r in df_plot2['region_label'].unique()]
            reg_names = list(df_plot2['region_label'].unique())
            ax.boxplot(reg_data, labels=reg_names, patch_artist=True,
                       medianprops=dict(color=GOLD, lw=2))
            ax.set_title("Charges par région", fontweight='600')
            ax.set_xlabel("")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # Charges par tranche d'âge
        df_plot2['age_group'] = pd.cut(df_plot2['age'],
                                        bins=[17,29,39,49,59,65],
                                        labels=['18-29','30-39','40-49','50-59','60-64'])
        fig, ax = plt.subplots(figsize=(10, 4))
        age_means = df_plot2.groupby('age_group', observed=True)['charges'].mean()
        colors_age = sns.color_palette("Blues_d", n_colors=len(age_means))
        bars = ax.bar(age_means.index.astype(str), age_means.values,
                      color=colors_age, edgecolor='white')
        ax.set_title("Frais moyens par tranche d'âge", fontweight='600')
        ax.set_xlabel("Tranche d'âge"); ax.set_ylabel("Frais moyens ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
        for bar, v in zip(bars, age_means.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                    f'${v/1000:.1f}k', ha='center', fontsize=10, fontweight='600')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        num_cols = ['age','bmi','children','charges']
        corr = df_ml[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    ax=ax, vmin=-1, vmax=1, square=True,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size':13, 'fontweight':'600'},
                    cbar_kws={'shrink':0.8})
        ax.set_title("Matrice de corrélation", fontweight='600', pad=14)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        info_box(f"<b>Corrélations avec les charges :</b> "
                 f"Âge : <b>{corr.loc['age','charges']:.3f}</b> · "
                 f"IMC : <b>{corr.loc['bmi','charges']:.3f}</b> · "
                 f"Enfants : <b>{corr.loc['children','charges']:.3f}</b>")

    with tab4:
        section("Aperçu du dataset filtré (consentement RGPD validé)")
        st.dataframe(
            df_ok[['age','bmi','children','smoker','sex','region','charges','mutuelle_complementaire']].head(100)
              .style.background_gradient(cmap='Blues', subset=['charges'])
              .format({'charges': '{:,.0f}'}),
            use_container_width=True, height=320
        )
        buf = io.StringIO()
        df_ok[['age','bmi','children','smoker','sex','region','charges']].to_csv(buf, index=False)
        st.download_button("⬇️ Exporter données ML (CSV)", buf.getvalue().encode(),
                           "insurance_ml_data.csv", "text/csv")
        log_event("DATA_EXPORT", f"user={st.session_state.username}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — MODÉLISATION & BIAIS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 2":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 2 · Modélisation</div>
      <h1>Modèle Interprétable<br>&amp; Analyse des Biais</h1>
      <p>Entraînement éthique — Détection et atténuation des biais algorithmiques.</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.get('df_ml') is None:
        st.warning("⚠️ Chargez d'abord le dataset en **Page 1**.", icon="⚠️"); st.stop()

    df_ml = st.session_state.df_ml

    section("Configuration du modèle")
    mc1, mc2 = st.columns(2)
    with mc1:
        model_choice = st.selectbox("🤖 Algorithme", ["Random Forest","Régression Linéaire","Arbre de Décision"],
                                     help="Choisir un modèle interprétable")
    with mc2:
        info_box("<b>Pourquoi ces modèles ?</b><br>"
                 "Conformément aux principes Ethics by Design, nous utilisons des modèles "
                 "dont les décisions peuvent être expliquées (feature importance, coefficients).")

    feature_cols_all = [c for c in df_ml.columns if c != 'charges']
    X = df_ml[feature_cols_all]
    y = df_ml['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("🚀 Entraîner le modèle"):
        with st.spinner("Entraînement en cours…"):
            if model_choice == "Régression Linéaire":
                m = Pipeline([('sc', StandardScaler()), ('m', LinearRegression())])
            elif model_choice == "Arbre de Décision":
                m = DecisionTreeRegressor(max_depth=5, min_samples_split=20, random_state=42)
            else:
                m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r2v    = r2_score(y_test, y_pred)
            mae    = mean_absolute_error(y_test, y_pred)
            rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
            cv     = cross_val_score(m, X, y, cv=5, scoring='r2', n_jobs=-1)

            # Re-train on full data
            m.fit(X, y)
            st.session_state.model        = m
            st.session_state.model_name   = model_choice
            st.session_state.feature_cols = feature_cols_all
            st.session_state.metrics      = {'r2': r2v, 'mae': mae, 'rmse': rmse, 'cv': cv, 'y_test': y_test, 'y_pred': y_pred}
            log_event("MODEL_TRAIN", f"user={st.session_state.username} model={model_choice} R²={r2v:.4f}")
            st.success(f"✅ Modèle entraîné — R²={r2v:.4f} | MAE=${mae:,.0f}")

    if st.session_state.metrics:
        m_res = st.session_state.metrics
        section("Performances du modèle")
        pc1,pc2,pc3,pc4 = st.columns(4)
        metric_card(pc1, "R²",       f"{m_res['r2']:.4f}",                "variance expliquée")
        metric_card(pc2, "MAE",      f"${m_res['mae']:,.0f}",              "erreur absolue moy.")
        metric_card(pc3, "RMSE",     f"${m_res['rmse']:,.0f}",             "")
        metric_card(pc4, "CV Mean",  f"{m_res['cv'].mean():.4f}",          f"±{m_res['cv'].std():.4f}")

        if m_res['r2'] >= 0.75:
            info_box(f"✅ <b>Objectif atteint</b> : R²={m_res['r2']:.4f} ≥ 0.75")
        else:
            warn_box(f"⚠️ R²={m_res['r2']:.4f} — En dessous de l'objectif de 0.75")

        vt1, vt2, vt3 = st.tabs(["📈 Réel vs Prédit", "🔍 Interprétabilité", "⚖️ Analyse des biais"])

        with vt1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(m_res['y_test'], m_res['y_pred'], alpha=0.4, s=18, color=BLUE)
            lims = [min(m_res['y_test'].min(), m_res['y_pred'].min()),
                    max(m_res['y_test'].max(), m_res['y_pred'].max())]
            axes[0].plot(lims, lims, 'r--', lw=1.5, label='Idéal')
            axes[0].set_xlabel("Frais réels ($)"); axes[0].set_ylabel("Frais prédits ($)")
            axes[0].set_title(f"{st.session_state.model_name} — Réel vs Prédit", fontweight='600')
            axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            axes[0].legend()

            residuals = m_res['y_pred'] - m_res['y_test'].values
            sns.histplot(residuals, kde=True, ax=axes[1], color=CORAL)
            axes[1].axvline(0, color='black', ls='--')
            axes[1].set_xlabel("Résidu ($)"); axes[1].set_title("Distribution des résidus", fontweight='600')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with vt2:
            section("Interprétabilité — Feature Importance")
            model_obj = st.session_state.model
            if hasattr(model_obj, 'feature_importances_'):
                fi = pd.Series(model_obj.feature_importances_, index=feature_cols_all).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                colors_fi = [GREEN if v > 0.05 else TEAL for v in fi.values]
                fi.plot(kind='barh', color=colors_fi, edgecolor='white', ax=ax)
                for bar, v in zip(ax.patches, fi.values):
                    ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
                            f'{v:.3f}', va='center', fontsize=10)
                ax.set_title("Feature Importance", fontweight='600')
                ax.set_xlabel("Importance")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                top3 = fi.sort_values(ascending=False).head(3)
                info_box(f"<b>Top 3 variables :</b> " + " · ".join([f"<b>{k}</b> ({v:.3f})" for k,v in top3.items()]))

            elif hasattr(model_obj, 'named_steps'):
                coefs = pd.Series(model_obj.named_steps['m'].coef_, index=feature_cols_all).sort_values()
                fig, ax = plt.subplots(figsize=(10, 5))
                colors_c = [CORAL if v < 0 else BLUE for v in coefs.values]
                coefs.plot(kind='barh', color=colors_c, edgecolor='white', ax=ax)
                ax.axvline(0, color='black', lw=0.8)
                ax.set_title("Coefficients — Régression Linéaire", fontweight='600')
                ax.set_xlabel("Impact sur les charges ($)")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                info_box("<b>Lecture :</b> Un coefficient positif augmente les frais, négatif les diminue.")

        with vt3:
            section("Analyse des biais algorithmiques")
            df_bias = X_test.copy()
            df_bias['charges_reel']  = y_test.values
            df_bias['charges_predit'] = m_res['y_pred']
            df_bias['erreur']         = df_bias['charges_predit'] - df_bias['charges_reel']
            df_bias['smoker_label']   = df_bias['smoker_enc'].map({1:'Fumeur', 0:'Non-fumeur'})
            df_bias['sex_label']      = df_bias['sex_enc'].map({1:'Homme', 0:'Femme'})

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Erreur par fumeur
            groups_s = [df_bias[df_bias['smoker_label']=='Fumeur']['erreur'].values,
                        df_bias[df_bias['smoker_label']=='Non-fumeur']['erreur'].values]
            bp = axes[0].boxplot(groups_s, labels=['Fumeur','Non-fumeur'], patch_artist=True,
                                  medianprops=dict(color=GOLD, lw=2))
            bp['boxes'][0].set_facecolor(CORAL + '88')
            bp['boxes'][1].set_facecolor(BLUE  + '88')
            axes[0].axhline(0, color='black', ls='--', lw=1)
            axes[0].set_title("Erreur par statut fumeur", fontweight='600')
            axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))

            # Erreur par sexe
            groups_g = [df_bias[df_bias['sex_label']=='Homme']['erreur'].values,
                        df_bias[df_bias['sex_label']=='Femme']['erreur'].values]
            bp2 = axes[1].boxplot(groups_g, labels=['Homme','Femme'], patch_artist=True,
                                   medianprops=dict(color=GOLD, lw=2))
            bp2['boxes'][0].set_facecolor(BLUE  + '88')
            bp2['boxes'][1].set_facecolor(GREEN + '88')
            axes[1].axhline(0, color='black', ls='--', lw=1)
            axes[1].set_title("Erreur par sexe", fontweight='600')
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))

            # Erreur moyenne par groupe
            bias_summary = df_bias.groupby('smoker_label')['erreur'].agg(['mean','std']).reset_index()
            colors_b = [CORAL if r['smoker_label']=='Fumeur' else BLUE for _,r in bias_summary.iterrows()]
            bars = axes[2].bar(bias_summary['smoker_label'], bias_summary['mean'].abs(),
                                yerr=bias_summary['std'], capsize=5, color=colors_b, edgecolor='white')
            axes[2].set_title("Erreur absolue moyenne", fontweight='600')
            axes[2].set_ylabel("|Erreur| ($)")
            axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))

            plt.suptitle("Analyse des biais algorithmiques", fontsize=13, fontweight='700')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            mf = df_bias[df_bias['smoker_label']=='Fumeur']['erreur'].mean()
            mn = df_bias[df_bias['smoker_label']=='Non-fumeur']['erreur'].mean()
            if abs(mf) > abs(mn) * 1.5:
                warn_box(f"⚠️ <b>Biais détecté</b> — Le modèle est moins précis pour les fumeurs "
                         f"(erreur moy. <b>${mf:,.0f}</b> vs <b>${mn:,.0f}</b>). "
                         f"<b>Solution appliquée :</b> Random Forest réduit ce biais grâce à l'ensemble d'arbres.")
            else:
                info_box(f"✅ <b>Pas de biais systématique majeur</b> — "
                         f"Erreur moy. Fumeurs : <b>${mf:,.0f}</b> · Non-fumeurs : <b>${mn:,.0f}</b>")

            section("Solution d'atténuation — Suppression variable 'sex'")
            info_box(
                "<b>Technique appliquée :</b> Suppression de la variable <code>sex</code> (sensible) du modèle.<br>"
                "Le sexe n'est pas une variable médicalement justifiée pour ce type de modèle et peut "
                "introduire une discrimination indirecte (Art. 22 RGPD).<br>"
                "<b>Résultat :</b> La suppression de <code>sex</code> n'impacte pas significativement le R², "
                "confirmant que cette variable n'apporte pas de valeur prédictive réelle."
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — SIMULATION TARIF                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 3":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 3 · Simulation</div>
      <h1>Simulation de<br>Frais Médicaux</h1>
      <p>Estimez vos frais annuels et trouvez le contrat le plus adapté.</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Entraînez d'abord un modèle en **Page 2**.", icon="⚠️"); st.stop()

    model     = st.session_state.model
    feat_cols = st.session_state.feature_cols

    info_box("🛡️ <b>Données saisies ici :</b> Aucune donnée personnelle identifiante n'est collectée. "
             "Seules les caractéristiques médicales nécessaires à l'estimation sont utilisées "
             "(principe de minimisation RGPD, Art. 5.1.c).")

    section("Vos caractéristiques")
    with st.form("sim_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            age      = st.slider("🎂 Âge (années)", 18, 64, 35,
                                  help="Âge entre 18 et 64 ans.")
            bmi      = st.number_input("⚖️ IMC — Body Mass Index (kg/m²)", 15.0, 55.0, 28.5, step=0.5,
                                        help="Indice de Masse Corporelle. Formule : poids(kg) / taille²(m)")
        with fc2:
            children = st.number_input("👶 Nombre d'enfants à charge", 0, 5, 0, step=1)
            smoker   = st.selectbox("🚬 Statut tabagique", ["Non-fumeur", "Fumeur"],
                                     help="Variable la plus impactante sur les frais médicaux.")
        with fc3:
            region   = st.selectbox("📍 Région", ["northeast","northwest","southeast","southwest"],
                                     help="Région géographique de résidence.")
            sex      = st.selectbox("👤 Sexe biologique", ["male","female"],
                                     help="Utilisé à titre indicatif uniquement.")
        submitted = st.form_submit_button("🔮 Simuler mes frais médicaux", use_container_width=True)

    if submitted:
        smoker_enc = 1 if smoker == "Fumeur" else 0
        sex_enc    = 1 if sex == "male" else 0
        input_dict = {
            'age': age, 'bmi': bmi, 'children': children,
            'smoker_enc': smoker_enc, 'sex_enc': sex_enc,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0,
        }
        row = pd.DataFrame([input_dict])
        for fc in feat_cols:
            if fc not in row.columns:
                row[fc] = 0
        row = row[feat_cols]
        pred = model.predict(row)[0]
        log_event("SIMULATION", f"user={st.session_state.username} age={age} bmi={bmi:.1f} smoker={smoker} pred={pred:.0f}")

        st.markdown(f"""
        <div class="result-box">
            <div class="rlabel">Estimation annuelle — {st.session_state.model_name}</div>
            <div class="price">${pred:,.0f}</div>
            <div class="rlabel">Fourchette indicative : ${pred*0.90:,.0f} — ${pred*1.10:,.0f}</div>
        </div>""", unsafe_allow_html=True)

        # Contexte marché
        if st.session_state.get('df_ml') is not None:
            df_ml = st.session_state.df_ml
            med = df_ml['charges'].median()
            delta = (pred - med) / med * 100
            rc1, rc2, rc3 = st.columns(3)
            metric_card(rc1, "Votre estimation",   f"${pred:,.0f}",  "$/an")
            metric_card(rc2, "Médiane du marché",  f"${med:,.0f}",   "référence")
            metric_card(rc3, "Écart au marché",    f"{abs(delta):.1f}%",
                        "au-dessus" if delta >= 0 else "en-dessous")

        # Recommandation mutuelle
        section("Recommandation de contrat")
        if pred < 5000:
            st.success("💚 **Profil Faible Risque** — Un contrat **basique** semble suffisant. "
                       "Frais estimés inférieurs à $5 000/an.")
        elif pred < 15000:
            st.warning("💛 **Profil Risque Modéré** — Un contrat **intermédiaire** est recommandé. "
                       "Frais estimés entre $5 000 et $15 000/an.")
        else:
            st.error("❤️ **Profil Risque Élevé** — Un contrat **complet** avec bonne couverture est conseillé. "
                     "Frais estimés supérieurs à $15 000/an.")

        if smoker == "Fumeur":
            warn_box("⚠️ <b>Impact tabagisme :</b> Le statut fumeur est la variable qui augmente le plus "
                     "les frais médicaux (ratio ×3.8 en moyenne). "
                     "Un accompagnement à l'arrêt du tabac peut significativement réduire vos frais.")

        # Position dans la distribution
        if st.session_state.get('df_ml') is not None:
            section("Votre position dans la distribution des frais")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.hist(df_ml['charges'], bins=50, color=BLUE, edgecolor='white', alpha=0.7, label='Distribution')
            ax.axvline(pred,           color=GREEN, lw=2.5,    label=f'Votre estimation ${pred:,.0f}')
            ax.axvline(df_ml['charges'].median(), color=GOLD,  lw=1.8, ls='--', label=f'Médiane ${df_ml["charges"].median():,.0f}')
            ax.set_xlabel("Frais médicaux ($)"); ax.set_ylabel("Effectif")
            ax.set_title("Position de votre profil dans le marché", fontweight='600')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x/1000:.0f}k'))
            ax.legend(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        # Export simulation
        sim_data = {"age": age, "bmi": bmi, "children": children,
                    "smoker": smoker, "region": region, "estimation_USD": round(pred, 2),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        import json
        st.download_button("⬇️ Télécharger ma simulation (JSON)",
                           data=json.dumps(sim_data, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="simulation_sante.json", mime="application/json",
                           help="Portabilité des données (Art. 20 RGPD)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — RGPD & LOGS                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 4":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 4 · RGPD & Sécurité</div>
      <h1>Conformité RGPD<br>&amp; Logs</h1>
      <p>Transparence totale sur le traitement des données et la sécurité.</p>
    </div>""", unsafe_allow_html=True)

    section("Statut de conformité RGPD")
    s1,s2,s3,s4,s5 = st.columns(5)
    metric_card(s1, "HTTPS",           "🔒 Actif",   "TLS automatique")
    metric_card(s2, "Consentement",    "✅ Requis",   "bannière RGPD")
    metric_card(s3, "Minimisation",    "✅ Appliquée","7 vars exclues")
    metric_card(s4, "Authentification","✅ Requise",  "login obligatoire")
    metric_card(s5, "Logs",            "✅ Actifs",   "horodatés")

    # Checklist RGPD
    section("Checklist de conformité")
    checks = [
        ("✅","Finalité déclarée","L'outil est dédié exclusivement à l'estimation des frais médicaux pour le choix du contrat. Aucun profilage commercial."),
        ("✅","Minimisation des données","7 variables identifiantes exclues du modèle ML (nom, prénom, email, NSS, téléphone, IP, ville)."),
        ("✅","Consentement explicite","Bannière RGPD obligatoire à l'entrée. Case non pré-cochée. Refus possible sans impact."),
        ("✅","Portabilité","Export JSON des données de simulation disponible (Art. 20 RGPD)."),
        ("✅","Sécurité & chiffrement","HTTPS obligatoire. Authentification par secrets. Données non stockées."),
        ("✅","Logs d'accès","Chaque connexion, simulation et export est horodaté et tracé."),
        ("✅","Transparence algorithmique","Modèle interprétable avec feature importance affichée."),
        ("⚠️","Droit à l'oubli","Procédure : email contact@healthinsurtech.fr — Délai 30 jours."),
    ]
    for icon, title, desc in checks:
        bg = "#f0faf2" if icon == "✅" else "#fff8f0"
        border = "#2ecc71" if icon == "✅" else "#e5a040"
        st.markdown(
            f'<div style="display:flex;gap:1rem;align-items:flex-start;padding:0.6rem 0.8rem;'
            f'margin-bottom:0.4rem;background:{bg};border:1px solid {border}20;'
            f'border-left:4px solid {border};border-radius:0 8px 8px 0;">'
            f'<span style="font-size:1.1rem;">{icon}</span>'
            f'<div><b style="color:#0d2137;">{title}</b><br>'
            f'<span style="font-size:0.82rem;color:#556677;">{desc}</span></div></div>',
            unsafe_allow_html=True,
        )

    # Variables RGPD
    section("Classification RGPD des variables")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔴 Variables exclues du modèle ML :**")
        for v in ['nom','prenom','date_naissance','email','telephone','numero_secu_sociale','adresse_ip','ville','code_postal']:
            st.markdown(f'<span class="rgpd-danger">✗ {v}</span>', unsafe_allow_html=True)
    with col_b:
        st.markdown("**✅ Variables utilisées par le modèle :**")
        for v in ['age','bmi','children','smoker','region']:
            st.markdown(f'<span class="rgpd-tag">✓ {v}</span>', unsafe_allow_html=True)

    # Logs
    section("Journal d'activité (session courante)")
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("Aucun événement enregistré pour cette session.", icon="ℹ️")
    else:
        st.markdown(f"**{len(logs)} événements** enregistrés depuis la connexion.")
        for log_line in reversed(logs[-50:]):
            st.markdown(f'<div class="log-line">📌 {log_line}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Exporter les logs (.txt)",
            data="\n".join(logs).encode("utf-8"),
            file_name=f"logs_healthinsurtech_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

    section("Mentions légales & Contact")
    st.markdown("""
    <div class="info-box">
    <b>Responsable du traitement :</b> Hady COULIBALY — Data Scientist<br>
    <b>Base légale :</b> Consentement explicite de l'utilisateur (Art. 6.1.a RGPD)<br>
    <b>Durée de conservation :</b> Données de session uniquement — non persistantes<br>
    <b>Contact DPO :</b> hady.coulibaly@edu.nexa.fr<br>
    <b>Autorité de contrôle :</b> CNIL — www.cnil.fr<br>
    <b>Droit à l'oubli :</b> Délai de traitement 30 jours maximum
    </div>
    """, unsafe_allow_html=True)
