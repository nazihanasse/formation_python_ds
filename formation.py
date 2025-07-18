# ==============================================
# 1. IMPORTATION DES LIBRAIRIES
# ==============================================

# pandas (pd) : Pour la manipulation et l'analyse des données
import pandas as pd

# numpy (np) : Pour les calculs numériques
import numpy as np

# matplotlib.pyplot (plt) : Pour la visualisation graphique
import matplotlib.pyplot as plt

# seaborn (sns) : Pour des visualisations statistiques avancées
import seaborn as sns

# scipy.stats : Pour les tests statistiques
from scipy import stats

# sklearn.linear_model.LinearRegression : Pour la régression linéaire
from sklearn.linear_model import LinearRegression

# sklearn.model_selection.train_test_split : Pour diviser les données
from sklearn.model_selection import train_test_split


# ==============================================
# 2. CONFIGURATION GLOBALE
# ==============================================

# sns.set_theme() : Configure le thème visuel de seaborn (style="whitegrid" pour un fond avec grille)
sns.set_theme(style="whitegrid")

# plt.rcParams : Paramètres globaux de matplotlib (taille des figures par défaut)
plt.rcParams["figure.figsize"] = (10, 5)


# ==============================================
# 3. CHARGEMENT DES DONNÉES
# ==============================================

def charger_donnees(chemin_fichier):
    """
    Charge les données à partir d'un fichier CSV.
    Si le fichier n'existe pas, crée un jeu de données exemple.

    Args:
        chemin_fichier (str): Chemin vers le fichier CSV

    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    try:
        # pd.read_csv() : Lit un fichier CSV dans un DataFrame pandas
        df = pd.read_csv(chemin_fichier)
    except FileNotFoundError:
        print("Attention: Fichier introuvable. Création d'un exemple de données.")
        data = {
            'jour': [7, 20, 29, 15, 11],
            'mois': [9, 6, 11, 11, 1],
            'saison': ['automne', 'ete', 'automne', 'automne', 'hiver'],
            'temp_moyenne': [18.05, 16.65, 19.77, 18.35, 18.24],
            'nb_evenements': [0, 1, 0, 5, 2],
            'ensoleillement': [7.94, 6.46, 6.57, 6.37, 6.01],
            'distance_aeroport': [23.03, 17.04, 18.17, 26.55, 15.05],
            'jours_sans_pluie': [17.99, 16.53, 17.61, 26.71, 17.18],
            'nuitées': [46.38, 57.42, 66.46, 113.45, 78.69]
        }
        # pd.DataFrame() : Crée un DataFrame à partir d'un dictionnaire
        df = pd.DataFrame(data)
    return df

# Chargement des données
df = charger_donnees("jeu_de_donnees_touristiques_1.csv")


# ==============================================
# 4. EXPLORATION DES DONNÉES
# ==============================================

def explorer_donnees(df):
    """
    Affiche des informations de base sur le DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à explorer
    """
    # df.head() : Affiche les 5 premières lignes
    print("\n=== Aperçu des données ===")
    print(df.head())

    # df.info() : Affiche les types de données et valeurs manquantes
    print("\n=== Structure des données ===")
    print(df.info())

    # df.describe() : Statistiques descriptives pour les colonnes numériques
    print("\n=== Statistiques descriptives ===")
    print(df.describe())

explorer_donnees(df)


# ==============================================
# 5. NETTOYAGE DES DONNÉES
# ==============================================

def nettoyer_donnees(df):
    """
    Nettoie le DataFrame en gérant les valeurs manquantes et aberrantes.

    Args:
        df (pd.DataFrame): DataFrame à nettoyer

    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    # df.isnull().sum() : Compte les valeurs manquantes par colonne
    print("\n=== Valeurs manquantes ===")
    print(df.isnull().sum())

    # df.dropna() : Supprime les lignes avec valeurs manquantes
    df_clean = df.dropna()

    # Correction des valeurs aberrantes avec lambda et max()
    df_clean["jours_sans_pluie"] = df_clean["jours_sans_pluie"].apply(
        lambda x: max(0, x))

    print(f"\nLignes avant/après nettoyage: {len(df)}/{len(df_clean)}")
    return df_clean

df_clean = nettoyer_donnees(df)


# ==============================================
# 6. ANALYSE UNIVARIÉE
# ==============================================

def analyse_univariee(df, num_cols, cat_cols):
    """
    Réalise une analyse univariée des variables numériques et catégorielles.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        num_cols (list): Liste des colonnes numériques
        cat_cols (list): Liste des colonnes catégorielles
    """
    # Analyse des variables numériques
    print("\n=== ANALYSE DES VARIABLES NUMÉRIQUES ===")
    for col in num_cols:
        # sns.histplot() : Histogramme avec courbe de densité
        plt.figure()
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution de {col}")
        plt.show()

    # Analyse des variables catégorielles
    print("\n=== ANALYSE DES VARIABLES CATÉGORIELLES ===")
    for col in cat_cols:
        # sns.countplot() : Diagramme en barres pour variables catégorielles
        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f"Répartition par {col}")
        plt.xticks(rotation=45)
        plt.show()

# Colonnes à analyser
num_cols = ["temp_moyenne", "nb_evenements", "ensoleillement",
            "distance_aeroport", "jours_sans_pluie", "nuitées"]
cat_cols = ["saison", "mois"]

analyse_univariee(df_clean, num_cols, cat_cols)


# ==============================================
# 7. ANALYSE BIVARIÉE
# ==============================================

def analyse_bivariee(df, num_cols):
    """
    Réalise une analyse bivariée des variables.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        num_cols (list): Liste des colonnes numériques
    """
    # Matrice de corrélation
    print("\n=== MATRICE DE CORRÉLATION ===")
    # df[num_cols].corr() : Calcule les corrélations entre colonnes numériques
    corr_matrix = df[num_cols].corr()

    # sns.heatmap() : Visualisation de la matrice de corrélation
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de Corrélation")
    plt.show()

    # Boxplots par saison
    print("\n=== BOXPLOTS PAR SAISON ===")
    for col in num_cols:
        # sns.boxplot() : Diagramme en boîte montrant la distribution par catégorie
        plt.figure()
        sns.boxplot(data=df, x="saison", y=col)
        plt.title(f"{col} par saison")
        plt.show()

analyse_bivariee(df_clean, num_cols)


# ==============================================
# 8. ANALYSE TEMPORELLE
# ==============================================

def analyse_temporelle(df):
    """
    Analyse l'évolution des nuitées au cours des mois.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
    """
    print("\n=== ÉVOLUTION TEMPORELLE ===")
    # Sélectionner uniquement les colonnes numériques pour le calcul de la moyenne
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ajouter 'mois' qui est utilisé pour le groupby
    if 'mois' not in numeric_cols and 'mois' in df.columns:
        numeric_cols.append('mois')

    # Grouper par mois et calculer la moyenne uniquement pour les colonnes numériques
    df_monthly = df[numeric_cols].groupby("mois").mean().reset_index()

    # sns.lineplot() : Graphique linéaire avec marqueurs
    plt.figure()
    sns.lineplot(data=df_monthly, x="mois", y="nuitées", marker="o")
    plt.title("Évolution moyenne des nuitées par mois")
    mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
            "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    plt.xticks(range(1, 13), mois)
    plt.show()

analyse_temporelle(df_clean)


# ==============================================
# 9. TESTS STATISTIQUES
# ==============================================

def tests_statistiques(df):
    """
    Effectue des tests statistiques sur les données.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
    """
    # Test ANOVA
    print("\n=== TEST ANOVA ===")
    # stats.f_oneway() : Test ANOVA pour comparer les moyennes entre groupes
    groups = [df[df["saison"] == season]["nuitées"]
              for season in df["saison"].unique()]
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion : Différence significative entre les saisons.")
    else:
        print("Conclusion : Pas de différence significative.")

    # Test de corrélation
    print("\n=== TEST DE CORRÉLATION ===")
    # stats.pearsonr() : Test de corrélation de Pearson
    coef, p_val = stats.pearsonr(df["ensoleillement"], df["nuitées"])
    print(f"Corrélation = {coef:.2f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("Conclusion : Corrélation significative.")
    else:
        print("Conclusion : Pas de corrélation significative.")

tests_statistiques(df_clean)


# ==============================================
# 10. MODÉLISATION
# ==============================================

def modelisation(df, features, target):
    """
    Construit et évalue un modèle de régression linéaire.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        features (list): Liste des variables explicatives
        target (str): Variable cible
    """
    print("\n=== MODÉLISATION PAR RÉGRESSION ===")
    # Séparation des données
    X = df[features]
    y = df[target]

    # train_test_split() : Divise les données en ensembles d'entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # LinearRegression() : Crée et entraîne un modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Évaluation
    print(f"R² (Train): {model.score(X_train, y_train):.2f}")
    print(f"R² (Test): {model.score(X_test, y_test):.2f}")

    # Coefficients
    print("\nCoefficients du modèle:")
    for feat, coef in zip(features, model.coef_):
        print(f"{feat}: {coef:.2f}")

modelisation(df_clean,
             ["temp_moyenne", "ensoleillement", "jours_sans_pluie"],
             "nuitées")


# ==============================================
# 11. VISUALISATIONS AVANCÉES
# ==============================================

def visualisations_avancees(df):
    """
    Crée des visualisations avancées des données.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
    """
    print("\n=== VISUALISATIONS AVANCÉES ===")
    # Nuage de points avec régression
    # sns.lmplot() : Nuage de points avec droite de régression par groupe
    plt.figure()
    sns.lmplot(data=df, x="ensoleillement", y="nuitées", hue="saison", height=6)
    plt.title("Relation ensoleillement-nuitées par saison")
    plt.show()

    # Violin plot
    # sns.violinplot() : Combinaison boxplot et estimation de densité
    plt.figure()
    sns.violinplot(data=df, x="saison", y="nuitées")
    plt.title("Distribution des nuitées par saison")
    plt.show()

visualisations_avancees(df_clean)


# ==============================================
# 12. EXÉCUTION PRINCIPALE
# ==============================================

if __name__ == "__main__":
    print("=== DÉBUT DE L'ANALYSE ===")

    # Chargement des données
    df = charger_donnees("jeu_de_donnees_touristiques_1.csv")

    # Exploration initiale
    explorer_donnees(df)

    # Nettoyage
    df_clean = nettoyer_donnees(df)

    # Analyses
    analyse_univariee(df_clean, num_cols, cat_cols)
    analyse_bivariee(df_clean, num_cols)
    analyse_temporelle(df_clean)
    tests_statistiques(df_clean)
    modelisation(df_clean,
                ["temp_moyenne", "ensoleillement", "jours_sans_pluie"],
                "nuitées")
    visualisations_avancees(df_clean)
