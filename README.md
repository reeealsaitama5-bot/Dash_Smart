# auto-dashboard

Génère automatiquement un dashboard HTML à partir de n'importe quel fichier CSV, en une seule commande — sans serveur, sans connexion internet, sans configuration.

![Aperçu du dashboard](examples/apercu_dashboard.png)

## Pourquoi

Un CSV à explorer rapidement ne mérite pas toujours d'ouvrir Excel ou Jupyter. `auto_dashboard.py` détecte automatiquement le type de chaque colonne (numérique, catégorielle, date, texte) et génère les statistiques et graphiques pertinents dans un unique fichier `dashboard.html`, ouvrable dans n'importe quel navigateur.

## Fonctionnalités

- Détection automatique du séparateur et de l'encodage du CSV
- Détection automatique du type de chaque colonne
- Statistiques globales : nombre de lignes/colonnes, % de valeurs manquantes, doublons
- Statistiques par colonne : min/moyenne/médiane/max, valeurs uniques, top valeurs
- Graphiques générés selon les données disponibles :
  - Série temporelle si une colonne date est détectée
  - Matrice de corrélation si ≥ 2 colonnes numériques
  - Histogrammes pour les colonnes numériques
  - Diagrammes en barres pour les colonnes catégorielles
- Sortie en **un seul fichier HTML autonome** (images encodées en base64) — partageable par email, sans dépendance externe

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python3 auto_dashboard.py data.csv
```

Options :

| Option | Description | Défaut |
|---|---|---|
| `-o`, `--output` | Fichier HTML de sortie | `dashboard.html` |
| `-t`, `--title` | Titre du dashboard | `Dashboard — <nom_du_fichier>` |
| `--sep` | Séparateur CSV (`,`, `;`, `\t`...) | Auto-détecté |
| `--max-categorical` | Nombre max de valeurs uniques pour traiter une colonne texte comme catégorielle | `20` |

### Exemple

```bash
python3 auto_dashboard.py examples/exemple_simple.csv -o rapport.html -t "Ventes 2024"
```

Un fichier CSV d'exemple est fourni dans [`examples/exemple_simple.csv`](examples/exemple_simple.csv) pour tester rapidement.

## Prérequis

- Python 3.9+
- pandas, numpy, matplotlib (voir `requirements.txt`)

## Version web (upload CSV dans le navigateur)

En plus du script en ligne de commande, `streamlit_app.py` fournit une interface web : l'utilisateur dépose un CSV et le dashboard s'affiche instantanément, sans rien installer côté client.

La version web ajoute :

- **Un design "pro"** style rapport corporate : fond clair, bleu marine/bleu vif, titre en serif, sections en petites capitales, et une section **observations automatiques** qui résume en langage naturel les points clés du fichier (colonne la plus incomplète, corrélation la plus forte, doublons, période couverte, anomalies, etc.). Tous les graphiques utilisent exclusivement des nuances de bleu.
- **Graphiques interactifs** (zoom, survol, export png natif) via Plotly, avec en plus un bouton "Télécharger (PNG)" sous chaque visualisation pour une image nette et prête à l'emploi.
- **Détection d'anomalies** : repère les valeurs aberrantes de chaque colonne numérique (méthode IQR ou Z-score, seuil réglable dans la barre latérale), affiche un résumé par colonne et des box plots interactifs, et permet de consulter les lignes concernées.
- **Export des données** en CSV et Excel (`.xlsx`) — au choix : toutes les lignes, les données nettoyées (sans les anomalies), ou uniquement les lignes aberrantes.
- **Export du dashboard complet en PDF** : un bouton "Rapport PDF" en haut de page génère un rapport prêt à partager (KPI, observations, détail des colonnes, résumé des anomalies, tous les graphiques), mis en page avec `reportlab`.

**Tester en local :**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Ça ouvre automatiquement `http://localhost:8501` dans le navigateur.

**Déployer en ligne (gratuit) — Streamlit Community Cloud :**

1. Pousse ce repo sur GitHub (voir plus haut).
2. Va sur [share.streamlit.io](https://share.streamlit.io) et connecte-toi avec ton compte GitHub.
3. Clique sur "New app", sélectionne ce repo, la branche `main`, et le fichier `streamlit_app.py`.
4. Clique sur "Deploy". En 1-2 minutes, l'app est en ligne à une adresse du type `https://ton-app.streamlit.app`, publique et partageable.

Chaque `git push` sur `main` redéploie automatiquement l'app.

**Alternative — Hugging Face Spaces :** même principe (gratuit, GitHub-friendly), utile si Streamlit Cloud est indisponible ou si tu veux regrouper plusieurs projets sur un seul profil.

## Limites connues

- Pensé pour des CSV de taille raisonnable (jusqu'à quelques centaines de milliers de lignes) ; pas optimisé pour du big data.
- Une seule colonne temporelle est utilisée pour la série temporelle (la première détectée).
- Les colonnes texte à forte cardinalité (ex : identifiants, emails) sont listées mais pas graphées.

## Licence

MIT — libre d'utilisation, modification et redistribution.
