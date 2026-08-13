"""Auto Dashboard — version multi-formats et analyse exploratoire.

L'application accepte des fichiers tabulaires courants, profile automatiquement
les données, propose un nettoyage non destructif, génère des visualisations,
détecte les anomalies et exporte les résultats.

Lancer : streamlit run streamlit_app_ameliore.py
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NAVY = "#0B2447"
ACCENT = "#1D4ED8"
ACCENT_DARK = "#1E3A8A"
ACCENT_LIGHT = "#DBEAFE"
INK = "#0F172A"
MUTED = "#5B6472"
GRID = "#DCE3EC"
BG_CARD = "#FFFFFF"
BG_PAGE = "#FFFFFF"
ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|code|identifiant|identifier|uuid|guid|^ref$|_ref$|^key$|_key$|^n[uo]m[ée]ro|^number$)",
    re.IGNORECASE,
)
# Bilingue français/anglais : la détection de hiérarchie géographique ne doit pas
# dépendre de la langue des noms de colonnes.
GEO_HIERARCHY_PATTERN = re.compile(
    r"(r[ée]gion|region|d[ée]partement|department|arrondissement|commune|com_arrt|"
    r"\bville\b|\bcity\b|\btown(?:ship)?\b|quartier|neighbou?rhood|village|hameau|hamlet|"
    r"province|district|\bzone\b|localit[ée]|\blocality\b|borough|municipalit[éy]|\bward\b|"
    r"\bcounty\b|\bstate\b|\bcountry\b|\bpays\b|suburb|parish|\bzip\b|postal[_ ]?code|\bcp\b)",
    re.IGNORECASE,
)
DEMO_COUNT_PATTERN = re.compile(
    r"(^population$|^pop$|hommes?$|femmes?$|m[ée]nages?$|concessions?$|habitants?$|effectif|"
    r"\bmales?$|\bfemales?$|\bmen$|\bwomen$|households?$|families?$|inhabitants?$|residents?$)",
    re.IGNORECASE,
)
# Colonnes de coordonnées géographiques (latitude/longitude), très fréquentes dans les
# données de mobilité, géolocalisation, capteurs, etc. — indépendant de la langue.
LAT_PATTERN = re.compile(r"^(?P<prefix>.*?)[_\s]*lat(?:itude)?$", re.IGNORECASE)
LON_PATTERN = re.compile(r"^(?P<prefix>.*?)[_\s]*(?:lon(?:gitude)?|lng)$", re.IGNORECASE)
MAX_ROWS_COMFORTABLE = 300_000
MAX_PREVIEW_ROWS = 200
SUPPORTED_EXTENSIONS = [
    "csv", "tsv", "txt", "xlsx", "xls", "json", "parquet", "dta", "sav", "sas7bdat"
]

CMAP_CORR = LinearSegmentedColormap.from_list("blue_corr", ["#FFFFFF", ACCENT_DARK])
PLOTLY_CORR_SCALE = [[0, "#FFFFFF"], [1, ACCENT_DARK]]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": False, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

st.set_page_config(page_title="Auto Dashboard", page_icon="▪", layout="wide")

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{ font-family: Inter, -apple-system, sans-serif; }}
.stApp {{ background: {BG_PAGE}; }}
.app-header {{ background: {BG_CARD}; border-bottom: 3px solid {ACCENT_DARK}; padding: 22px 4px 18px; margin-bottom: 24px; }}
.app-header .eyebrow {{ font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; }}
.app-header h1 {{ margin: 4px 0; font-size: 2rem; font-weight: 700; color: {NAVY}; font-family: 'Source Serif 4', Georgia, serif; }}
.app-header p {{ margin: 0; color: {MUTED}; font-size: .92rem; }}
.section-title {{ font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: {ACCENT_DARK}; margin: 10px 0 14px; padding-bottom: 8px; border-bottom: 1px solid {GRID}; }}
.insight-item {{ background: {BG_CARD}; border-left: 3px solid {ACCENT_DARK}; border-top: 1px solid {GRID}; border-right: 1px solid {GRID}; border-bottom: 1px solid {GRID}; border-radius: 2px; padding: 10px 14px; margin-bottom: 8px; font-size: .88rem; color: {INK}; }}
.insight-item b {{ color: {NAVY}; }}
.stButton>button, .stDownloadButton>button {{ border-radius: 3px; font-weight: 600; border-color: {ACCENT_DARK}; color: {ACCENT_DARK}; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def _decode_candidates() -> list[str]:
    return ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def _read_delimited(data: bytes, sep: str | None) -> pd.DataFrame:
    last_error = None
    for enc in _decode_candidates():
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, engine="python" if sep is None else None, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Impossible de lire le fichier texte : {last_error}")


def _read_json(data: bytes) -> pd.DataFrame:
    try:
        obj = json.loads(data.decode("utf-8-sig"))
        if isinstance(obj, dict):
            # Orientations classiques : records, columns, split, ou dictionnaire de listes.
            for orient in ("records", "columns", "split"):
                try:
                    df = pd.DataFrame.from_dict(obj, orient=orient) if orient != "records" else pd.DataFrame(obj)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df
                except Exception:
                    continue
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        return pd.read_json(io.BytesIO(data))
    except Exception as exc:
        raise RuntimeError(f"JSON non lisible ou structure non tabulaire : {exc}") from exc


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str, sep: str | None = None, sheet_name: str | int | None = 0) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            return _read_delimited(file_bytes, sep)
        if ext in {".tsv", ".txt"}:
            return _read_delimited(file_bytes, "\t" if ext == ".tsv" and sep is None else sep)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        if ext == ".json":
            return _read_json(file_bytes)
        if ext == ".parquet":
            return pd.read_parquet(io.BytesIO(file_bytes))
        if ext == ".dta":
            return pd.read_stata(io.BytesIO(file_bytes), convert_categoricals=False)
        if ext == ".sav":
            return pd.read_spss(io.BytesIO(file_bytes))
        if ext == ".sas7bdat":
            return pd.read_sas(io.BytesIO(file_bytes), format="sas7bdat", encoding="utf-8")
    except ImportError as exc:
        raise RuntimeError(
            f"Le moteur nécessaire au format {ext} n'est pas installé. "
            "Ajoute les dépendances indiquées dans requirements.txt."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Lecture impossible de {filename} : {exc}") from exc
    raise RuntimeError(f"Format non pris en charge : {ext or 'sans extension'}")


def get_excel_sheets(file_bytes: bytes, filename: str) -> list[str]:
    if Path(filename).suffix.lower() not in {".xlsx", ".xls"}:
        return []
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Profilage / détection
# -----------------------------------------------------------------------------

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing_tokens = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
    for col in out.select_dtypes(include=["object", "string"]).columns:
        s = out[col].astype("string")
        mask = s.str.strip().str.lower().isin(missing_tokens)
        out.loc[mask.fillna(False), col] = pd.NA
    return out


def detect_column_types(df: pd.DataFrame, max_categorical: int = 20) -> dict[str, list[str]]:
    numeric, dates, categorical, text, identifiers, binary = [], [], [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            binary.append(col); categorical.append(col); continue
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            uniqueness = valid.nunique() / len(valid) if len(valid) else 0
            if uniqueness >= .9 and ID_NAME_PATTERN.search(str(col)):
                identifiers.append(col)
            else:
                numeric.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            dates.append(col); continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(100)
            if len(sample):
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= .8:
                    df[col] = pd.to_datetime(s, errors="coerce", format="mixed")
                    dates.append(col); continue
        n_unique = s.nunique(dropna=True)
        if n_unique <= max_categorical:
            categorical.append(col)
        else:
            text.append(col)
    return {
        "numeric": numeric, "datetime": dates, "categorical": categorical,
        "text": text, "identifier": identifiers, "binary": binary,
    }


def detect_geo_demo(df: pd.DataFrame, types: dict) -> dict:
    """Repère une éventuelle structure de type recensement : hiérarchie géographique
    (région > département > commune > quartier…) et compteurs démographiques
    (population, hommes, femmes, ménages, concessions…)."""
    geo_cols = [c for c in df.columns if GEO_HIERARCHY_PATTERN.search(str(c)) and c in types["categorical"] + types["identifier"] + types["text"]]
    # Ordonne du plus large au plus fin selon le nombre de modalités distinctes.
    geo_cols = sorted(set(geo_cols), key=lambda c: df[c].nunique(dropna=True))
    demo_cols = [c for c in types["numeric"] if DEMO_COUNT_PATTERN.search(str(c))]
    male_col = next((c for c in demo_cols if re.search(r"hommes?$|\bmales?$|\bmen$", str(c), re.IGNORECASE)), None)
    female_col = next((c for c in demo_cols if re.search(r"femmes?$|\bfemales?$|\bwomen$", str(c), re.IGNORECASE)), None)
    pop_col = next((c for c in demo_cols if re.search(r"^population$|^pop$|habitants?$|inhabitants?$|residents?$", str(c), re.IGNORECASE)), None)
    household_col = next((c for c in demo_cols if re.search(r"m[ée]nages?$|households?$|families?$", str(c), re.IGNORECASE)), None)
    return {
        "is_census_like": len(geo_cols) >= 1 and len(demo_cols) >= 1,
        "geo_cols": geo_cols,
        "demo_cols": demo_cols,
        "male_col": male_col,
        "female_col": female_col,
        "pop_col": pop_col,
        "household_col": household_col,
    }


def detect_coordinates(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    """Repère les paires latitude/longitude (ex : pickup_latitude / pickup_longitude),
    fréquentes dans les données de mobilité, livraison, capteurs, etc. Ne dépend pas de
    noms de colonnes en français : reconnaît lat/latitude/lon/longitude/lng dans n'importe
    quelle langue, avec ou sans préfixe (pickup_, dropoff_, store_…)."""
    lats, lons = {}, {}
    for c in numeric_cols:
        valid = pd.to_numeric(df[c], errors="coerce").dropna()
        if valid.empty:
            continue
        m = LAT_PATTERN.match(str(c).strip())
        if m and valid.between(-90, 90).mean() > .98:
            lats[m.group("prefix").strip("_ ").lower()] = c
            continue
        m = LON_PATTERN.match(str(c).strip())
        if m and valid.between(-180, 180).mean() > .98:
            lons[m.group("prefix").strip("_ ").lower()] = c
    pairs = []
    for prefix, lat_col in lats.items():
        lon_col = lons.get(prefix)
        if lon_col:
            label = prefix.replace("_", " ").strip().title() or "Position"
            pairs.append({"label": label, "lat": lat_col, "lon": lon_col})
    return pairs


def column_stats_rows(df: pd.DataFrame, types: dict) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        valid = s.dropna()
        unique = s.nunique(dropna=True)
        missing = round(s.isna().mean() * 100, 1)
        if col in types["identifier"]:
            kind, detail = "Identifiant", f"{unique} valeurs quasi uniques · exclu des statistiques"
        elif col in types["numeric"]:
            mean = valid.mean() if len(valid) else np.nan
            med = valid.median() if len(valid) else np.nan
            kind = "Numérique"
            detail = f"min {valid.min():.2f} · moy {mean:.2f} · médiane {med:.2f} · max {valid.max():.2f}" if len(valid) else "Aucune valeur exploitable"
        elif col in types["datetime"]:
            kind = "Temporelle"
            detail = f"{valid.min().date()} → {valid.max().date()}" if len(valid) else "Aucune date exploitable"
        elif col in types["categorical"]:
            kind = "Binaire" if col in types.get("binary", []) else "Catégorielle"
            vc = s.value_counts(dropna=True).head(1)
            detail = f"{unique} valeurs · top : {vc.index[0]} ({vc.iloc[0]})" if len(vc) else "—"
        else:
            kind, detail = "Texte libre", f"{unique} valeurs distinctes"
        rows.append({"Colonne": col, "Type": kind, "Uniques": int(unique), "Manquant (%)": missing, "Détail": detail})
    return rows


def quality_report(df: pd.DataFrame) -> dict:
    total_cells = max(df.size, 1)
    missing_cells = int(df.isna().sum().sum())
    duplicated = int(df.duplicated().sum())
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    rows_empty = int(df.isna().all(axis=1).sum()) if len(df) else 0
    return {
        "missing_pct": missing_cells / total_cells * 100,
        "duplicated": duplicated,
        "duplicate_pct": duplicated / max(len(df), 1) * 100,
        "constant": constant,
        "empty_columns": empty_cols,
        "empty_rows": rows_empty,
        "columns": len(df.columns),
        "rows": len(df),
    }


def numeric_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    records = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        records.append({
            "Variable": col,
            "N": int(s.size),
            "Moyenne": s.mean(),
            "Médiane": s.median(),
            "Écart-type": s.std(),
            "Min": s.min(),
            "Q1": s.quantile(.25),
            "Q3": s.quantile(.75),
            "Max": s.max(),
            "Asymétrie": s.skew(),
            "Kurtosis": s.kurt(),
            "Manquant (%)": df[col].isna().mean() * 100,
        })
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Nettoyage
# -----------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, drop_duplicates: bool, strip_strings: bool,
                   convert_numeric: bool, normalize_missing: bool,
                   missing_numeric: str, missing_categorical: str) -> pd.DataFrame:
    out = df.copy()
    if normalize_missing:
        out = normalize_missing_values(out)
    if strip_strings:
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].astype("string").str.strip()
    if convert_numeric:
        for col in out.columns:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                converted = pd.to_numeric(out[col].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
                non_missing = out[col].notna().sum()
                if non_missing and converted.notna().sum() / non_missing >= .95:
                    out[col] = converted
    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    numeric_cols = out.select_dtypes(include=np.number).columns
    if missing_numeric != "Ne rien faire":
        for col in numeric_cols:
            if missing_numeric == "Médiane":
                out[col] = out[col].fillna(out[col].median())
            elif missing_numeric == "Moyenne":
                out[col] = out[col].fillna(out[col].mean())
    if missing_categorical != "Ne rien faire":
        cat_cols = out.select_dtypes(include=["object", "string", "category"]).columns
        for col in cat_cols:
            if missing_categorical == "Mode":
                mode = out[col].mode(dropna=True)
                if len(mode):
                    out[col] = out[col].fillna(mode.iloc[0])
            elif missing_categorical == "Inconnue":
                out[col] = out[col].fillna("Inconnue")
    return out

# -----------------------------------------------------------------------------
# Anomalies
# -----------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, numeric_cols: list[str], method: str, threshold: float):
    summary, combined, bounds = [], pd.Series(False, index=df.index), {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 4:
            continue
        if method == "iqr":
            q1, q3 = valid.quantile(.25), valid.quantile(.75)
            spread = q3 - q1
            lower, upper = (valid.min(), valid.max()) if spread == 0 else (q1 - threshold * spread, q3 + threshold * spread)
        elif method == "zscore":
            mean, std = valid.mean(), valid.std()
            if pd.isna(std) or std == 0:
                continue
            lower, upper = mean - threshold * std, mean + threshold * std
        else:  # MAD
            med = valid.median()
            mad = np.median(np.abs(valid - med))
            if mad == 0:
                continue
            robust_sigma = 1.4826 * mad
            lower, upper = med - threshold * robust_sigma, med + threshold * robust_sigma
        if valid.min() >= 0 and lower < 0:
            lower = 0.0
        mask = ((s < lower) | (s > upper)).fillna(False)
        combined |= mask
        bounds[col] = (float(lower), float(upper))
        n = int(mask.sum())
        summary.append({
            "Colonne": col,
            "Bornes acceptées": f"[{lower:.2f} ; {upper:.2f}]",
            "Valeurs aberrantes": n,
            "% de la colonne": round(n / max(len(valid), 1) * 100, 1),
        })
    return summary, combined, bounds

# -----------------------------------------------------------------------------
# Visualisations
# -----------------------------------------------------------------------------

def plotly_hist(series: pd.Series, title: str) -> go.Figure:
    fig = px.histogram(x=series.dropna(), nbins=min(30, max(8, int(series.nunique() ** .5) + 5)))
    fig.update_traces(marker_color=ACCENT)
    return style_plotly(fig, title, "Valeur", "Fréquence")


def plotly_bar(series: pd.Series, title: str) -> go.Figure:
    vc = series.value_counts(dropna=True).head(15).sort_values()
    fig = go.Figure(go.Bar(x=vc.values, y=vc.index.astype(str), orientation="h", marker_color=ACCENT))
    return style_plotly(fig, title, "Occurrences", "")


def plotly_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color if color else None, opacity=.75)
    fig.update_traces(marker_line_width=0)
    return style_plotly(fig, f"{y} selon {x}", x, y)


def plotly_box(df: pd.DataFrame, y: str, x: str | None = None) -> go.Figure:
    fig = px.box(df, y=y, x=x)
    fig.update_traces(marker_color=ACCENT, line_color=ACCENT)
    return style_plotly(fig, f"Distribution de {y}", x or "", y)


def plotly_corr(df: pd.DataFrame, numeric_cols: list[str], method: str = "pearson") -> go.Figure:
    corr = df[numeric_cols].corr(method=method, numeric_only=True)
    mag = corr.abs()
    fig = go.Figure(go.Heatmap(z=mag.values, x=numeric_cols, y=numeric_cols,
        colorscale=PLOTLY_CORR_SCALE, zmin=0, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}<br>r = %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style_plotly(fig, f"Corrélations — {method.title()}", "", "", height=max(380, 55 * len(numeric_cols)))


def plotly_timeseries(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str) -> go.Figure | None:
    cols = [date_col] if value_col is None else [date_col, value_col]
    valid = df[cols].dropna(subset=[date_col]).copy()
    if valid.empty:
        return None
    valid[date_col] = pd.to_datetime(valid[date_col], errors="coerce")
    valid = valid.dropna(subset=[date_col]).sort_values(date_col)
    grouped = valid.set_index(date_col)[value_col].resample(freq).mean() if value_col else valid.set_index(date_col).resample(freq).size()
    temp = grouped.reset_index(name=value_col if value_col else "Nombre de lignes")
    fig = px.line(temp, x=date_col, y=temp.columns[-1])
    fig.update_traces(line_color=ACCENT, line_width=2.4)
    return style_plotly(fig, f"{value_col or 'Volume'} dans le temps", date_col, value_col or "Nombre de lignes")


def plotly_geo_bar(df: pd.DataFrame, level_col: str, value_col: str, top_n: int = 20) -> go.Figure:
    """Total d'une variable démographique agrégée par niveau géographique, trié décroissant."""
    grouped = df.groupby(level_col, dropna=True)[value_col].sum().sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(x=grouped.values, y=grouped.index.astype(str), orientation="h", marker_color=ACCENT))
    fig.update_yaxes(autorange="reversed")
    return style_plotly(fig, f"{value_col} par {level_col} (top {min(top_n, len(grouped))})", value_col, level_col,
                         height=max(340, 24 * len(grouped)))


def plotly_sex_ratio(df: pd.DataFrame, level_col: str, male_col: str, female_col: str, top_n: int = 15) -> go.Figure:
    """Pyramide comparant hommes / femmes pour les niveaux géographiques les plus peuplés."""
    grouped = df.groupby(level_col, dropna=True)[[male_col, female_col]].sum()
    grouped["total"] = grouped[male_col] + grouped[female_col]
    grouped = grouped.sort_values("total", ascending=False).head(top_n).sort_values("total")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=-grouped[male_col].values, y=grouped.index.astype(str), orientation="h",
                          name="Hommes", marker_color=ACCENT_DARK))
    fig.add_trace(go.Bar(x=grouped[female_col].values, y=grouped.index.astype(str), orientation="h",
                          name="Femmes", marker_color=ACCENT_LIGHT, marker_line=dict(color=ACCENT_DARK, width=1)))
    fig.update_layout(barmode="overlay", bargap=0.15)
    fig = style_plotly(fig, f"Hommes / Femmes par {level_col}", "Effectif", level_col, height=max(340, 26 * len(grouped)))
    fig.update_layout(showlegend=True)
    fig.update_xaxes(tickvals=None)
    return fig


def plotly_geo_treemap(df: pd.DataFrame, hierarchy_cols: list[str], value_col: str) -> go.Figure:
    """Vue hiérarchique (région > département > commune…) pondérée par une variable démographique."""
    fig = px.treemap(df, path=[px.Constant("Total")] + hierarchy_cols, values=value_col,
                      color_discrete_sequence=[ACCENT_LIGHT, ACCENT, ACCENT_DARK, NAVY])
    fig.update_traces(marker=dict(line=dict(color="white", width=1)), textfont=dict(color=INK))
    return style_plotly(fig, f"Répartition de {value_col} — {' → '.join(hierarchy_cols)}", "", "", height=520)


def plotly_points_map(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: str | None,
                       label: str, max_points: int = 8000) -> go.Figure:
    """Carte de points pour des coordonnées lat/lon (mobilité, géolocalisation, capteurs…),
    utilisable même sans aucune colonne géographique nommée (région, ville…)."""
    valid = df[[lat_col, lon_col] + ([color_col] if color_col else [])].copy()
    valid[lat_col] = pd.to_numeric(valid[lat_col], errors="coerce")
    valid[lon_col] = pd.to_numeric(valid[lon_col], errors="coerce")
    valid = valid.dropna(subset=[lat_col, lon_col])
    valid = valid[valid[lat_col].between(-90, 90) & valid[lon_col].between(-180, 180)]
    if len(valid) > max_points:
        valid = valid.sample(max_points, random_state=0)
    fig = px.scatter_map(
        valid, lat=lat_col, lon=lon_col, color=color_col,
        color_continuous_scale=PLOTLY_CORR_SCALE if color_col and pd.api.types.is_numeric_dtype(valid[color_col]) else None,
        opacity=.65, zoom=9, height=520, map_style="open-street-map",
    )
    fig.update_traces(marker=dict(size=6) if not color_col else {})
    fig.update_layout(margin=dict(l=0, r=0, t=42, b=0),
                       title=dict(text=f"Carte des points — {label}", x=0, xanchor="left", font=dict(size=13, color=NAVY)))
    return fig


def style_plotly(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_white", font=dict(family="Inter, sans-serif", color=INK, size=12),
                      title=dict(text=title, x=0, xanchor="left", font=dict(size=13, color=NAVY)),
                      margin=dict(l=10, r=10, t=46, b=10), height=height,
                      plot_bgcolor="white", paper_bgcolor="white", showlegend=bool(fig.layout.showlegend))
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=GRID)
    fig.update_yaxes(title_text=y_title, showgrid=False, linecolor=GRID)
    return fig


def matplotlib_from_plotly(fig: go.Figure) -> plt.Figure:
    # Export PDF/PNG robuste : création graphique équivalent avec Kaleido n'est
    # pas requise. Ici on utilise un visuel simplifié pour les exports lorsque
    # nécessaire.
    raise RuntimeError("Conversion Plotly → Matplotlib directe non disponible ; utilisez une figure dédiée.")


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_matplotlib_hist(series: pd.Series, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    data = series.dropna()
    ax.hist(data, bins=min(30, max(8, int(data.nunique() ** .5) + 5)), color=ACCENT, alpha=.88, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Fréquence")
    fig.tight_layout()
    return fig


def make_matplotlib_bar(series: pd.Series, title: str) -> plt.Figure:
    vc = series.value_counts(dropna=True).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(vc.index.astype(str), vc.values, color=ACCENT)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Occurrences")
    fig.tight_layout()
    return fig


def make_matplotlib_corr(df: pd.DataFrame, cols: list[str], method: str) -> plt.Figure:
    corr = df[cols].corr(method=method, numeric_only=True)
    mag = corr.abs()
    n = len(cols)
    cell = .7 if n <= 12 else .48
    fig, ax = plt.subplots(figsize=(max(5, 2.4 + cell*n), max(4, 1.8 + cell*n)))
    im = ax.imshow(mag, cmap=CMAP_CORR, vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(cols, fontsize=7)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if mag.iloc[i,j] > .55 else INK)
    fig.colorbar(im, ax=ax, shrink=.8, label="|r|")
    ax.set_title(f"Corrélations — {method.title()}", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Insights et IA
# -----------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame, types: dict, outlier_rows: int = 0, geo: dict | None = None,
                       coord_pairs: list[dict] | None = None) -> list[str]:
    out = []
    q = quality_report(df)
    if coord_pairs:
        p = coord_pairs[0]
        lat_valid = pd.to_numeric(df[p["lat"]], errors="coerce").dropna()
        if len(lat_valid):
            out.append(f"**Géolocalisation —** {len(coord_pairs)} jeu(x) de coordonnées détecté(s) ({', '.join(c['label'] for c in coord_pairs)}), consultez l'onglet Carte.")
    if geo and geo["is_census_like"]:
        finest = geo["geo_cols"][-1] if geo["geo_cols"] else None
        broadest = geo["geo_cols"][0] if geo["geo_cols"] else None
        if geo["pop_col"]:
            total_pop = df[geo["pop_col"]].sum()
            out.append(f"**Population totale —** {total_pop:,.0f} habitants recensés.".replace(",", " "))
            if broadest:
                top_zone = df.groupby(broadest)[geo["pop_col"]].sum().idxmax()
                top_val = df.groupby(broadest)[geo["pop_col"]].sum().max()
                out.append(f"**Zone la plus peuplée —** « {top_zone} » avec {top_val:,.0f} habitants ({broadest}).".replace(",", " "))
        if geo["male_col"] and geo["female_col"]:
            m, f = df[geo["male_col"]].sum(), df[geo["female_col"]].sum()
            if f > 0:
                out.append(f"**Sex-ratio —** {m/f*100:.1f} hommes pour 100 femmes sur l'ensemble du périmètre.")
        if geo["household_col"] and geo["pop_col"]:
            hh = df[geo["household_col"]].sum()
            if hh > 0:
                out.append(f"**Taille moyenne des ménages —** {df[geo['pop_col']].sum()/hh:.1f} personnes par ménage.")
        if finest:
            out.append(f"**Granularité —** {df[finest].nunique(dropna=True)} unités distinctes au niveau « {finest} ».")
    if q["duplicated"]:
        out.append(f"**Doublons —** {q['duplicated']} lignes dupliquées ({q['duplicate_pct']:.1f}%).")
    if q["missing_pct"] == 0:
        out.append("**Complétude —** aucune valeur manquante détectée.")
    else:
        top = df.isna().mean().sort_values(ascending=False).iloc[0]
        col = df.isna().mean().sort_values(ascending=False).index[0]
        out.append(f"**Complétude —** « {col} » est la variable la plus incomplète ({top*100:.1f}%).")
    if q["constant"]:
        out.append(f"**Colonnes constantes —** {len(q['constant'])} variable(s) ne présentent aucune variation utile.")
    if len(types["numeric"]) >= 2:
        corr = df[types["numeric"]].corr(numeric_only=True).abs()
        if not corr.empty:
            corr_arr = corr.values.copy()
            np.fill_diagonal(corr_arr, 0)
            i, j = np.unravel_index(np.nanargmax(corr_arr), corr_arr.shape)
            a, b = corr.index[i], corr.columns[j]
            real = df[[a,b]].corr().iloc[0,1]
            if np.isfinite(real):
                out.append(f"**Association —** « {a} » et « {b} » présentent la plus forte corrélation de Pearson (r = {real:.2f}).")
    if types["datetime"]:
        c = types["datetime"][0]
        v = df[c].dropna()
        if len(v): out.append(f"**Période —** « {c} » couvre du {v.min().date()} au {v.max().date()}.")
    if outlier_rows:
        out.append(f"**Anomalies —** {outlier_rows} ligne(s) contiennent au moins une valeur aberrante selon les paramètres choisis.")
    return out


def summary_for_ai(df: pd.DataFrame, types: dict, stats: pd.DataFrame, q: dict, outlier_summary: list[dict]) -> str:
    parts = [f"Dataset : {q['rows']} lignes, {q['columns']} colonnes, {q['missing_pct']:.1f}% de cellules manquantes, {q['duplicated']} doublons."]
    parts.append(f"Types : {len(types['numeric'])} numériques, {len(types['categorical'])} catégorielles, {len(types['datetime'])} temporelles, {len(types['text'])} texte, {len(types['identifier'])} identifiants.")
    if not stats.empty:
        top = stats.sort_values("Asymétrie", key=lambda s: s.abs(), ascending=False).head(3)
        parts.append("Variables numériques remarquables : " + "; ".join(f"{r['Variable']} (moy={r['Moyenne']:.2f}, médiane={r['Médiane']:.2f}, skew={r['Asymétrie']:.2f})" for _, r in top.iterrows()))
    flagged = [x for x in outlier_summary if x["Valeurs aberrantes"] > 0]
    if flagged:
        parts.append("Anomalies : " + ", ".join(f"{x['Colonne']}={x['Valeurs aberrantes']}" for x in flagged[:5]))
    return "\n".join(parts)


def call_gemini(api_key: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text or ""

# -----------------------------------------------------------------------------
# Export Excel / PDF
# -----------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    return "_".join(keep.split()).lower()[:80] or "dashboard"


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Excel (openpyxl) ne supporte pas les datetimes avec fuseau horaire : on les rend naïfs."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]) and getattr(out[col].dt, "tz", None) is not None:
            out[col] = out[col].dt.tz_localize(None)
    return out


def to_excel_bytes(df: pd.DataFrame, stats: pd.DataFrame | None = None, outliers: pd.DataFrame | None = None, cleaned: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _strip_tz(df).to_excel(writer, index=False, sheet_name="Données")
        if cleaned is not None:
            _strip_tz(cleaned).to_excel(writer, index=False, sheet_name="Nettoyées")
        if stats is not None and not stats.empty:
            stats.to_excel(writer, index=False, sheet_name="Statistiques")
        if outliers is not None and not outliers.empty:
            _strip_tz(outliers).to_excel(writer, index=False, sheet_name="Anomalies")
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(title: str, df: pd.DataFrame, stats_rows: list[dict], insights: list[str], figures: list[tuple[str, plt.Figure]], q: dict, outlier_summary: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm, leftMargin=1.6*cm, rightMargin=1.6*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("T", parent=styles["Title"], fontSize=24, textColor=colors.HexColor(ACCENT_DARK), spaceAfter=5)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15, textColor=colors.HexColor(ACCENT_DARK), spaceBefore=12, spaceAfter=7)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.5, leading=13)
    story = [Paragraph(title, h1), Paragraph(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", body), Spacer(1, 10)]
    kpi = Table([["Lignes", "Colonnes", "Manquants", "Doublons"], [str(q["rows"]), str(q["columns"]), f"{q['missing_pct']:.1f}%", str(q["duplicated"])]], colWidths=[4*cm]*4)
    kpi.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(ACCENT)), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("GRID", (0,0), (-1,-1), .4, colors.HexColor(GRID))]))
    story.append(kpi)
    if insights:
        story.append(Paragraph("Observations", h2))
        for s in insights[:10]:
            story.append(Paragraph(re.sub(r"\*\*", "", s), body))
            story.append(Spacer(1, 3))
    story.append(Paragraph("Détail des colonnes", h2))
    rows = [["Colonne", "Type", "Uniques", "Manquant (%)", "Détail"]]
    for r in stats_rows:
        rows.append([str(r["Colonne"]), r["Type"], str(r["Uniques"]), f"{r['Manquant (%)']}%", str(r["Détail"])[:100]])
    table = Table(rows, colWidths=[3*cm, 2.3*cm, 1.7*cm, 2.2*cm, 7.2*cm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(ACCENT_DARK)), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTSIZE", (0,0), (-1,-1), 7), ("GRID", (0,0), (-1,-1), .3, colors.HexColor(GRID)), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(table)
    if outlier_summary:
        story.append(Paragraph("Détection d'anomalies", h2))
        od = [["Colonne", "Bornes", "Nb", "%"]] + [[x["Colonne"], x["Bornes acceptées"], str(x["Valeurs aberrantes"]), f"{x['% de la colonne']}%"] for x in outlier_summary]
        ot = Table(od, colWidths=[4*cm, 5*cm, 3*cm, 3*cm], repeatRows=1)
        ot.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(ACCENT_DARK)), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTSIZE", (0,0), (-1,-1), 8), ("GRID", (0,0), (-1,-1), .3, colors.HexColor(GRID))]))
        story.append(ot)
    if figures:
        story.append(PageBreak()); story.append(Paragraph("Visualisations", h2))
        for title2, fig in figures:
            story.append(Paragraph(title2, body)); story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig, 160)), width=16.5*cm, height=8.6*cm)); story.append(Spacer(1, 6))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
  <div class="eyebrow">Analyse exploratoire automatisée</div>
  <h1>Auto Dashboard</h1>
  <p>CSV, Excel, JSON, Parquet, Stata, SPSS et SAS · profilage · nettoyage · anomalies · rapports</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Paramètres")
    max_categorical = st.slider("Seuil catégoriel", 5, 100, 20)
    sep_choice = st.selectbox("Séparateur CSV/TXT", ["Auto-détecté", ",", ";", "Tabulation (\\t)", "|"])
    sep_map = {"Auto-détecté": None, ",": ",", ";": ";", "Tabulation (\\t)": "\t", "|": "|"}
    sep = sep_map[sep_choice]
    outlier_label = st.selectbox("Méthode anomalies", ["IQR", "Z-score", "MAD"])
    outlier_method = outlier_label.lower().replace("-score", "score")
    if outlier_method == "iqr":
        threshold = st.slider("Seuil IQR", 1.0, 3.0, 1.5, .1)
    elif outlier_method == "zscore":
        threshold = st.slider("Seuil Z-score", 2.0, 4.0, 3.0, .1)
    else:
        threshold = st.slider("Seuil MAD", 2.0, 6.0, 3.5, .1)
    corr_method = st.selectbox("Corrélation", ["pearson", "spearman", "kendall"])
    ai_enabled = st.checkbox("Commentaires IA")
    ai_key = st.text_input("Clé Gemini", type="password") if ai_enabled else ""

uploaded = st.file_uploader("Dépose ton fichier de données", type=SUPPORTED_EXTENSIONS, accept_multiple_files=False)
if uploaded is None:
    st.info("Dépose un fichier CSV, Excel, JSON, Parquet, Stata, SPSS ou SAS pour commencer.")
    st.stop()

file_bytes = uploaded.getvalue()
ext = Path(uploaded.name).suffix.lower()

# Excel : choix de feuille avant lecture.
sheets = get_excel_sheets(file_bytes, uploaded.name)
sheet = sheets[0] if sheets else 0
if sheets:
    sheet = st.sidebar.selectbox("Feuille Excel", sheets)

try:
    df = load_data(file_bytes, uploaded.name, sep=sep, sheet_name=sheet)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if not isinstance(df, pd.DataFrame):
    st.error("Le fichier n'a pas produit un tableau de données exploitable.")
    st.stop()

# Noms de colonnes nettoyés sans toucher aux valeurs.
df = df.copy()
df.columns = [str(c).strip() if str(c).strip() else f"Colonne_{i+1}" for i, c in enumerate(df.columns)]

if df.empty:
    st.warning("Le fichier est vide.")
    st.stop()

if len(df) > MAX_ROWS_COMFORTABLE:
    st.warning(f"{len(df):,} lignes détectées. Le dashboard peut ralentir au-delà de {MAX_ROWS_COMFORTABLE:,} lignes.".replace(",", " "))

types = detect_column_types(df, max_categorical)
geo = detect_geo_demo(df, types)
coord_pairs = detect_coordinates(df, types["numeric"])
q = quality_report(df)
stats_rows = column_stats_rows(df, types)
stats_num = numeric_profile(df, types["numeric"])

# -----------------------------------------------------------------------------
# Vue d'ensemble
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Vue d’ensemble</div>', unsafe_allow_html=True)
c = st.columns(6)
c[0].metric("Lignes", f"{len(df):,}".replace(",", " "))
c[1].metric("Colonnes", len(df.columns))
c[2].metric("Manquants", f"{q['missing_pct']:.1f}%")
c[3].metric("Doublons", q["duplicated"])
c[4].metric("Numériques", len(types["numeric"]))
c[5].metric("Catégorielles", len(types["categorical"]))

# Score qualité simple et transparent.
quality_score = max(0, min(100, 100 - q["missing_pct"] * .7 - q["duplicate_pct"] * .3 - 5 * min(len(q["constant"]), 5)))
st.progress(quality_score / 100, text=f"Score indicatif de qualité : {quality_score:.0f}/100")

# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
outlier_summary, outlier_mask, outlier_bounds = detect_outliers(df, types["numeric"], outlier_method, threshold)
insights = generate_insights(df, types, int(outlier_mask.sum()), geo, coord_pairs)
st.markdown('<div class="section-title">Observations automatiques</div>', unsafe_allow_html=True)
for line in insights:
    line_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
    st.markdown(f'<div class="insight-item">{line_html}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
tab_labels = ["Vue d’ensemble", "Données", "Nettoyage", "Visualisations"]
if geo["is_census_like"]:
    tab_labels.append("Géo / Démo")
if coord_pairs:
    tab_labels.append("Carte")
tab_labels += ["Statistiques", "Anomalies", "IA", "Exports"]
tabs = st.tabs(tab_labels)
tab_overview, tab_data, tab_clean, tab_viz = tabs[0], tabs[1], tabs[2], tabs[3]
next_idx = 4
if geo["is_census_like"]:
    tab_geo = tabs[next_idx]; next_idx += 1
else:
    tab_geo = None
if coord_pairs:
    tab_map = tabs[next_idx]; next_idx += 1
else:
    tab_map = None
tab_stats, tab_anom, tab_ai, tab_export = tabs[next_idx], tabs[next_idx+1], tabs[next_idx+2], tabs[next_idx+3]

with tab_overview:
    st.subheader("Profil du dataset")
    type_df = pd.DataFrame([
        ["Numériques", len(types["numeric"])], ["Catégorielles", len(types["categorical"])],
        ["Temporelles", len(types["datetime"])], ["Texte libre", len(types["text"])],
        ["Identifiants", len(types["identifier"])],
    ], columns=["Type", "Nombre"])
    st.dataframe(type_df, use_container_width=True, hide_index=True)
    if q["empty_columns"]:
        st.warning("Colonnes entièrement vides : " + ", ".join(q["empty_columns"]))
    if q["constant"]:
        st.info("Colonnes constantes : " + ", ".join(q["constant"]))

with tab_data:
    st.subheader("Détail des colonnes")
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    st.subheader("Aperçu")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)
    if len(df) > MAX_PREVIEW_ROWS:
        st.caption(f"Affichage limité aux {MAX_PREVIEW_ROWS} premières lignes.")

with tab_clean:
    st.subheader("Nettoyage non destructif")
    st.caption("Les transformations s'appliquent à une copie et ne modifient jamais le fichier original.")
    drop_dupes = st.checkbox("Supprimer les doublons", value=False)
    strip_strings = st.checkbox("Supprimer les espaces inutiles dans les textes", value=True)
    normalize_missing = st.checkbox("Reconnaître NA / N/A / null / ? / - comme manquants", value=True)
    convert_numeric = st.checkbox("Convertir automatiquement les colonnes quasi-numériques", value=True)
    c1, c2 = st.columns(2)
    miss_num = c1.selectbox("Valeurs manquantes numériques", ["Ne rien faire", "Médiane", "Moyenne"])
    miss_cat = c2.selectbox("Valeurs manquantes catégorielles", ["Ne rien faire", "Mode", "Inconnue"])
    cleaned_df = clean_dataset(df, drop_dupes, strip_strings, convert_numeric, normalize_missing, miss_num, miss_cat)
    cq = quality_report(cleaned_df)
    st.metric("Lignes après nettoyage", len(cleaned_df), delta=len(cleaned_df)-len(df))
    st.write(f"Valeurs manquantes après nettoyage : **{cq['missing_pct']:.1f}%** · doublons : **{cq['duplicated']}**")
    st.dataframe(cleaned_df.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)
    st.download_button("Télécharger le jeu nettoyé", cleaned_df.to_csv(index=False).encode("utf-8-sig"), f"{safe_filename(uploaded.name)}_nettoye.csv", "text/csv")

with tab_viz:
    st.subheader("Visualisations automatiques")
    if types["numeric"]:
        st.markdown("**Variables numériques**")
        cols = st.multiselect("Variables à afficher", types["numeric"], default=types["numeric"][:min(4, len(types["numeric"]))])
        grid = st.columns(2)
        for i, col in enumerate(cols):
            with grid[i % 2]:
                st.plotly_chart(plotly_hist(df[col], f"Distribution — {col}"), use_container_width=True, key=f"hist_{col}")
                st.download_button("PNG", fig_to_png_bytes(make_matplotlib_hist(df[col], f"Distribution — {col}")), f"{safe_filename(col)}_hist.png", "image/png", key=f"pngh_{i}")
    cat = [c for c in types["categorical"] if df[c].nunique(dropna=True) > 1]
    if cat:
        st.markdown("**Variables catégorielles**")
        cat_choice = st.multiselect("Catégories à afficher", cat, default=cat[:min(4, len(cat))])
        grid = st.columns(2)
        for i, col in enumerate(cat_choice):
            with grid[i % 2]:
                st.plotly_chart(plotly_bar(df[col], f"Répartition — {col}"), use_container_width=True, key=f"bar_{col}")
    if len(types["numeric"]) >= 2:
        st.plotly_chart(plotly_corr(df, types["numeric"], corr_method), use_container_width=True, key="corr_main")
    if types["datetime"]:
        dc = st.selectbox("Variable temporelle", types["datetime"], key="date_viz")
        vc = st.selectbox("Variable à suivre", [None] + types["numeric"], key="value_viz", format_func=lambda x: "Nombre de lignes" if x is None else x)
        freq_label = st.selectbox("Fréquence", ["Journalière", "Hebdomadaire", "Mensuelle", "Trimestrielle", "Annuelle"])
        freq = {"Journalière":"D", "Hebdomadaire":"W", "Mensuelle":"MS", "Trimestrielle":"QS", "Annuelle":"YS"}[freq_label]
        ts = plotly_timeseries(df, dc, vc, freq)
        if ts:
            st.plotly_chart(ts, use_container_width=True, key="timeseries")
    if len(types["numeric"]) >= 2:
        st.markdown("**Graphique personnalisé**")
        x = st.selectbox("X", types["numeric"], key="custom_x")
        y = st.selectbox("Y", [c for c in types["numeric"] if c != x], key="custom_y")
        color_options = ["Aucune"] + [c for c in types["categorical"] if df[c].nunique(dropna=True) <= 20]
        color_choice = st.selectbox("Couleur", color_options, key="custom_color")
        st.plotly_chart(plotly_scatter(df, x, y, None if color_choice == "Aucune" else color_choice), use_container_width=True, key="custom_scatter")

if tab_geo is not None:
    with tab_geo:
        st.subheader("Analyse géographique et démographique")
        st.caption("Détecté automatiquement : hiérarchie géographique et compteurs de population dans ce fichier.")

        level = st.selectbox("Niveau d'agrégation", geo["geo_cols"], index=len(geo["geo_cols"]) - 1, key="geo_level")
        value_options = geo["demo_cols"] or types["numeric"]
        default_value = geo["pop_col"] if geo["pop_col"] in value_options else (value_options[0] if value_options else None)
        value_col = st.selectbox("Variable à agréger", value_options,
                                  index=value_options.index(default_value) if default_value in value_options else 0,
                                  key="geo_value")

        agg = df.groupby(level, dropna=True)[value_col].sum().sort_values(ascending=False)
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Total {value_col}", f"{agg.sum():,.0f}".replace(",", " "))
        k2.metric("Unités géographiques", agg.shape[0])
        k3.metric(f"Zone la plus forte", str(agg.index[0]))

        st.plotly_chart(plotly_geo_bar(df, level, value_col), use_container_width=True, key="geo_bar")

        if geo["male_col"] and geo["female_col"]:
            st.markdown("**Répartition Hommes / Femmes**")
            st.plotly_chart(plotly_sex_ratio(df, level, geo["male_col"], geo["female_col"]), use_container_width=True, key="geo_sexratio")

        if len(geo["geo_cols"]) >= 2:
            st.markdown("**Vue hiérarchique**")
            hierarchy_choice = st.multiselect("Niveaux à inclure (du plus large au plus fin)", geo["geo_cols"],
                                               default=geo["geo_cols"][:min(3, len(geo["geo_cols"]))], key="geo_hierarchy")
            if len(hierarchy_choice) >= 1:
                st.plotly_chart(plotly_geo_treemap(df, hierarchy_choice, value_col), use_container_width=True, key="geo_treemap")

        st.markdown("**Tableau agrégé**")
        agg_table = df.groupby(level, dropna=True)[geo["demo_cols"]].sum().sort_values(value_col, ascending=False) if geo["demo_cols"] else agg.to_frame()
        agg_table = agg_table.reset_index()
        st.dataframe(agg_table, use_container_width=True, hide_index=True)
        st.download_button("Exporter l'agrégation", agg_table.to_csv(index=False).encode("utf-8-sig"),
                            f"agregation_{safe_filename(level)}.csv", "text/csv", key="geo_export")

if tab_map is not None:
    with tab_map:
        st.subheader("Carte des points géolocalisés")
        st.caption("Détecté automatiquement : colonnes de latitude/longitude (mobilité, livraison, capteurs…), sans besoin de colonnes région/ville nommées.")
        pair_labels = [p["label"] for p in coord_pairs]
        pair_choice = st.selectbox("Jeu de coordonnées", pair_labels, key="map_pair") if len(pair_labels) > 1 else pair_labels[0]
        pair = next(p for p in coord_pairs if p["label"] == pair_choice)

        color_options = ["Aucune"] + types["numeric"] + [c for c in types["categorical"] if df[c].nunique(dropna=True) <= 20]
        color_choice = st.selectbox("Colorer par", color_options, key="map_color")
        color_col = None if color_choice == "Aucune" else color_choice

        lat_valid = pd.to_numeric(df[pair["lat"]], errors="coerce").dropna()
        lon_valid = pd.to_numeric(df[pair["lon"]], errors="coerce").dropna()
        m1, m2, m3 = st.columns(3)
        m1.metric("Points valides", f"{lat_valid.shape[0]:,}".replace(",", " "))
        m2.metric("Étendue latitude", f"{lat_valid.min():.3f} → {lat_valid.max():.3f}")
        m3.metric("Étendue longitude", f"{lon_valid.min():.3f} → {lon_valid.max():.3f}")

        st.plotly_chart(plotly_points_map(df, pair["lat"], pair["lon"], color_col, pair["label"]), use_container_width=True, key="points_map")

        if len(coord_pairs) >= 2:
            st.caption(f"{len(coord_pairs)} jeux de coordonnées détectés : {', '.join(pair_labels)}. Change de sélection ci-dessus pour comparer (ex. prise en charge vs dépose).")

with tab_stats:
    st.subheader("Statistiques descriptives")
    if stats_num.empty:
        st.info("Aucune variable numérique exploitable.")
    else:
        st.dataframe(stats_num.style.format({c: "{:.3f}" for c in stats_num.columns if c != "Variable"}), use_container_width=True, hide_index=True)
        st.download_button("Exporter les statistiques", stats_num.to_csv(index=False).encode("utf-8-sig"), "statistiques.csv", "text/csv")
    if len(types["numeric"]) >= 2:
        st.markdown(f"**Matrice de corrélation ({corr_method})**")
        st.dataframe(df[types["numeric"]].corr(method=corr_method).round(3), use_container_width=True)

with tab_anom:
    st.subheader("Détection d'anomalies")
    if not types["numeric"]:
        st.info("Aucune colonne numérique.")
    else:
        a, b = st.columns(2)
        a.metric("Lignes concernées", int(outlier_mask.sum()))
        b.metric("Part du dataset", f"{outlier_mask.mean()*100:.1f}%")
        if outlier_summary:
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
            flagged = [x["Colonne"] for x in outlier_summary if x["Valeurs aberrantes"] > 0]
            for i, col in enumerate(flagged[:10]):
                st.plotly_chart(plotly_box(df, col), use_container_width=True, key=f"outbox_{col}_{i}")
            st.dataframe(df[outlier_mask].head(500), use_container_width=True, hide_index=True)
        else:
            st.success("Aucune anomalie avec les paramètres actuels.")

with tab_ai:
    st.subheader("Assistant d'analyse")
    if not ai_enabled:
        st.info("Active les commentaires IA dans la barre latérale pour utiliser cette section.")
    elif not ai_key:
        st.warning("Renseigne une clé Gemini dans la barre latérale.")
    else:
        dataset_summary = summary_for_ai(df, types, stats_num, q, outlier_summary)
        prompt = (
            "Tu es analyste de données. À partir uniquement du résumé statistique ci-dessous, "
            "produis un court diagnostic factuel en français : qualité des données, variables remarquables, "
            "anomalies, relations et 3 analyses utiles. Ne fabrique aucun chiffre et rappelle qu'une corrélation "
            "n'est pas une causalité.\n\n" + dataset_summary
        )
        if st.button("Analyser avec l'IA"):
            try:
                with st.spinner("Analyse en cours…"):
                    st.write(call_gemini(ai_key, prompt))
            except Exception as exc:
                st.error(f"IA indisponible : {exc}")

with tab_export:
    st.subheader("Exports")
    cleaned_export = clean_dataset(df, True, True, True, True, "Ne rien faire", "Ne rien faire")
    anomaly_df = df[outlier_mask] if outlier_mask.any() else pd.DataFrame(columns=df.columns)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Données originales · CSV", df.to_csv(index=False).encode("utf-8-sig"), f"{safe_filename(uploaded.name)}.csv", "text/csv")
        st.download_button("Données nettoyées · CSV", cleaned_export.to_csv(index=False).encode("utf-8-sig"), f"{safe_filename(uploaded.name)}_nettoye.csv", "text/csv")
    with col2:
        st.download_button("Classeur Excel complet", to_excel_bytes(df, stats_num, anomaly_df, cleaned_export), f"{safe_filename(uploaded.name)}_analyse.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    figures_pdf: list[tuple[str, plt.Figure]] = []
    for col in types["numeric"][:6]:
        figures_pdf.append((f"Distribution — {col}", make_matplotlib_hist(df[col], f"Distribution — {col}")))
    if len(types["numeric"]) >= 2:
        figures_pdf.append(("Corrélations", make_matplotlib_corr(df, types["numeric"], corr_method)))
    for col in [c for c in types["categorical"] if df[c].nunique(dropna=True) > 1][:4]:
        figures_pdf.append((f"Répartition — {col}", make_matplotlib_bar(df[col], f"Répartition — {col}")))
    pdf = generate_pdf(f"Dashboard — {Path(uploaded.name).stem}", df, stats_rows, insights, figures_pdf, q, outlier_summary)
    st.download_button("Rapport PDF complet", pdf, f"{safe_filename(uploaded.name)}_rapport.pdf", "application/pdf")

st.markdown(f'<div class="app-footer">Auto Dashboard · {uploaded.name} · généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}</div>', unsafe_allow_html=True)
