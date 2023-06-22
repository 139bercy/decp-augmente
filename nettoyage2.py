import cProfile
import json
import os
import pickle
import logging.handlers
import datetime
import pstats
import re

import numpy as np
import pandas as pd
import itertools
import utils
import recuperation_data

logger = logging.getLogger("main.nettoyage2")
logger.setLevel(logging.DEBUG)
pd.options.mode.chained_assignment = None  # default='warn'

path_to_conf = "confs"
if not (os.path.exists(path_to_conf)):  # Si le chemin confs n'existe pas (dans le cas de la CI et de Saagie)
    os.mkdir(path_to_conf)
if utils.USE_S3:
    res = utils.download_confs()
    if res:
        logger.info("Chargement des fichiers confs depuis le S3")
else:
    logger.info("ERROR Les fichiers de confs n'ont pas pu être chargés")

with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

with open(os.path.join("confs", "var_debug.json")) as f:
    conf_debug = json.load(f)["nettoyage"]

path_to_data = conf_data["path_to_data"]
decp_file_name = conf_data["decp_file_name"]
path_to_data = conf_data["path_to_data"]  # Réécris


def main():
    if utils.USE_S3:
        if not (os.path.exists(path_to_data)):  # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(path_to_data)
        utils.download_data_nettoyage()

    logger.info("Chargement des données")
    if utils.USE_S3:
        # download data from S3
        df = recuperation_data.main()
    else:
        # load data from local
        with open(os.path.join(path_to_data, "decpv2.pkl"), 'rb') as f:
            df = pickle.load(f)

    logger.info("Nettoyage des données")

    df = manage_data_quality(df)


def manage_data_quality(df: pd.DataFrame):
    """
    Cette fonction sépare en deux le dataframe d'entrée. Les données ne respectant pas les formats indiqués par les
    règles de gestion de la DAJ sont mise de côtés. Les règles de gestions sont dans un mail du 15 février 2023.

    /!\
    Dans les règles de gestion, certaine valeur de champ d'identification unique du marché ne sont pas accessibles
    dans la donnée brute. On va donc ne traiter dans cette fonction que les variables accessibles de manières brutes
    et lorsque les règles portent sur des variables non brutes, on appliquera les règles à ce moment-là. (ex : idtitulaire)
    /!\

    Les lignes exclues seront publiées sur data.economie.gouv.fr dans un fichier csv.

    Arguments
    ----------
    df :  le dataframe des données bruts.


    Return
    -----------
    df (dataFrame) : le dataframe des données à enrichir.
    df_badlines (dataFrame) : le dataframe des données exclues.

    """

    # séparation des marchés et des concessions, car traitement différent
    df_concession = df.loc[df['nature'].str.contains('concession', case=False, na=False)]
    df_marche = df.loc[~df['nature'].str.contains('concession', case=False, na=False)]

    df_concession, df_concession_badlines = regles_concession(df_concession)
    df_marche, df_marche_badlines = regles_marche(df_marche)

    print(df_concession.shape[0])
    print(df_concession_badlines.shape[0])

    print(df_marche.shape[0])
    print(df_marche_badlines.shape[0])


    # Concaténation des dataframes pour l'enrigissement (re-séparation après)
    df = pd.concat([df_concession, df_marche])

    logger.info("Pourcentage de mauvaises lignes : " + str(((df_concession_badlines.shape[0] + df_marche_badlines.shape[0]) / df.shape[0]) * 100))

    # Ecriture des données exclues dans un fichier csv
    df_concession_badlines.to_csv(os.path.join(conf_data["path_to_data"], "concession_exclu.csv"), index=False)
    df_marche_badlines.to_csv(os.path.join(conf_data["path_to_data"], "marche_exclu.csv"), index=False)

    return df


def regles_marche(df_marche_: pd.DataFrame) -> pd.DataFrame:
    df_marche_badlines_ = pd.DataFrame(columns=df_marche_.columns)

    def marche_check_empty(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        col_name = ["id", "acheteur.id", "titulaires", "montant",
                    "dureeMois"]  # titulaire contient un dict avec des valeurs dont id
        for col in col_name:
            dfb = pd.concat([dfb, df[~pd.notna(df[col])]])
            df = df[pd.notna(df[col])]
        return df, dfb

    def marche_cpv_object(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        # Si CPV manquant et objet du marché manquant, alors le marché est mis de côté
        dfb = pd.concat(
            [dfb, df[~pd.notna(df["codeCPV"]) | ~pd.notna(df["objet"])]])
        df = df[pd.notna(df["codeCPV"]) | pd.notna(df["objet"])]
        return df, dfb

    def marche_cpv(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        """
        Le CPV comprend 10 caractères (8 pour la racine + 1 pour le séparateur « - » et +1 pour la clé) – format texte pour ne pas supprimer les « 0 » en début de CPV.
        Un code CPV est INEXPLOITABLE s’il n’appartient pas à la liste des codes CPV existants dans la nomenclature européenne 2008 des CPV
        """
        # TODO : Vérifier que le CPV est dans la liste des CPV
        return df, dfb

    def marche_date(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        # Si la date de notification et la date de publication est manquante, alors le marché est mis de côté
        dfb = pd.concat([dfb, df[
            ~pd.notna(df["dateNotification"]) | ~pd.notna(df["datePublicationDonnees"])]])
        df = df[
            pd.notna(df["dateNotification"]) | pd.notna(df["datePublicationDonnees"])]
        return df, dfb

    def marche_dateNotification(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        """
        Format AAAA-MM-JJ
            Si MM<01 ou>12,
            SI JJ<01 ou >31 (voir si possibilité de vérifier le format jour max en fonction du mois et année)
        La date de notification est INEXPLOITABLE si elle ne respecte pas le format, ou si elle ne peut pas être retransformée au format initial (ex : JJ-MM-AAAA)
        Correction si INEXPLOITABLE :
            Si la date de notification du marché est manquante et qu’il existe une date de publication des données essentielles du marché public
            respectant le format AAAA-MM-JJ (ou pouvant être retransformé en ce format) alors il convient d’affecter la date de publication à la date de notification.
        """

        # vérification du format de la date de notification (AAAA-MM-JJ) et correction si besoin création d'un dataframe avec les lignes à corriger
        df["dateNotification"] = pd.to_datetime(df["dateNotification"], format='%Y-%m-%d', errors='ignore')
        format_regex = r'^\d{4}-\d{2}-\d{2}$'
        invalid_dates = df[~df["dateNotification"].str.match(format_regex, na=False)]
        df = df[df["dateNotification"].str.match(format_regex, na=False)]
        invalid_dates["dateNotification"] = invalid_dates["datePublicationDonnees"]
        still_invalid_dates = invalid_dates[~invalid_dates["dateNotification"].str.match(format_regex, na=False)]
        no_more_invalide_dates = invalid_dates[invalid_dates["dateNotification"].str.match(format_regex, na=False)]
        df = pd.concat([df, no_more_invalide_dates])
        dfb = pd.concat([dfb, still_invalid_dates])
        return df, dfb

    df_marche_, df_marche_badlines_ = marche_check_empty(df_marche_, df_marche_badlines_)
    df_marche_, df_marche_badlines_ = marche_cpv_object(df_marche_, df_marche_badlines_)
    df_marche_, df_marche_badlines_ = marche_date(df_marche_, df_marche_badlines_)

    df_marche_, df_marche_badlines_ = check_montant(df_marche_, df_marche_badlines_, "montant")
    df_marche_, df_marche_badlines_ = check_siret(df_marche_, df_marche_badlines_, "acheteur.id")

    df_marche_, df_marche_badlines_ = marche_cpv(df_marche_, df_marche_badlines_)

    df_marche_, df_marche_badlines_ = check_duree_contrat(df_marche_, df_marche_badlines_, 180)
    df_marche_, df_marche_badlines_ = marche_dateNotification(df_marche_, df_marche_badlines_)

    return df_marche_, df_marche_badlines_


def regles_concession(df_concession_: pd.DataFrame) -> pd.DataFrame:

    df_concession_badlines_ = pd.DataFrame(columns=df_concession_.columns)

    def concession_check_empty(df_con: pd.DataFrame, df_bad: pd.DataFrame) -> pd.DataFrame:
        col_name = ["id", "autoriteConcedante.id", "concessionnaires", "objet", "valeurGlobale",
                    "dureeMois"]  # concessionnaires contient un dict avec des valeurs dont id
        for col in col_name:
            df_bad = pd.concat(
                [df_bad, df_con[~pd.notna(df_con[col])]])
            df_con = df_con[pd.notna(df_con[col])]
        return df_con, df_bad

    def concession_date(df_con: pd.DataFrame, df_bad: pd.DataFrame) -> pd.DataFrame:
        # Si la date de début d’exécution et la date de publication est manquante alors le contrat de concession est mis de côté
        df_bad = pd.concat([df_bad, df_con[
            ~pd.notna(df_con["dateDebutExecution"]) | ~pd.notna(df_con["datePublicationDonnees"])]])
        df_con = df_con[
            pd.notna(df_con["dateDebutExecution"]) | pd.notna(df_con["datePublicationDonnees"])]
        return df_con, df_bad

    df_concession_, df_concession_badlines_ = concession_check_empty(df_concession_, df_concession_badlines_)
    df_concession_, df_concession_badlines_ = concession_date(df_concession_, df_concession_badlines_)

    df_concession_, df_concession_badlines_ = check_montant(df_concession_, df_concession_badlines_, "valeurGlobale")
    df_concession_, df_concession_badlines_ = check_siret(df_concession_, df_concession_badlines_, "autoriteConcedante.id")

    df_concession_, df_concession_badlines_ = check_duree_contrat(df_concession_, df_concession_badlines_, 360)

    return df_concession_, df_concession_badlines_


def check_montant(df: pd.DataFrame, dfb: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    La valeur est jugée INEXPLOITABLE
    si :
    -1 La valeur est supérieure à 3 000 000 000€ (Remarque : voir si règles des exceptions à transmettre plus tard).
    -2 Le montant est inférieur à 1€
    -3 Pour un seuil de 100 000 000, il y a
        -1 une succession de mêmes chiffres (ex: 999999999, 888888888, 99999988) les 0 ne sont pas considérés comme des chiffres identiques
        -2 la séquence du montant commençant par 123456789
        Méthodologie ci-dessous :
            Méthode de détection automatique des inexploitables par succession de mêmes chiffres (il convient initialement de passer en caractère le nombre pour le traiter comme une chaîne de caractère pour l’algorithme) :
                (Nombre de répétition du caractère le plus fréquent dans la chaîne de caractère > Nombre de caractères de la chaîne -2)
                & (Caractère le plus fréquent différent de « 0 »)
                & (Les positions du caractère le plus fréquent dans la chaîne de caractère se suivent sans interruption, càd constituent une suite croissante) alors INEXPLOITABLE
            Exemple applicatif : montant de 99999988€. Le « 9 » est l’occurrence la plus fréquente, la chaine de caractère est égale à 8 est donc 8-2 =6. La chaîne de caractère ne contient pas de 0.
            Répétition du « 9 » sans interruption (pas de « 8 » entre deux séries de « 9 »).
            Conclusion : INEXPLOITABLE

    Si INEXPLOITABLE, le contrat est mis de côté.
    """
    # replace string '' by 0
    df[col] = df[col].replace('', 0)
    # change col to float
    df[col] = df[col].astype(float)

    # 1
    dfb = pd.concat([dfb, df[df[col] > 3000000000]])
    df = df[df[col] <= 3000000000]

    # 2
    dfb = pd.concat([dfb, df[df[col] < 1]])
    df = df[df[col] >= 1]

    # 3.1
    # si le même chiffre autre que 0 est répété plus de 6 fois pour les montants supérieur à 100 000 000 alors INEXPLOITABLE
    same_digit_count = df[col].astype(str).apply(lambda x: x.count(x[0]))
    selected_rows = df[(same_digit_count > 6) & (df[col].astype(str).str[0] != "0") & (df[col] > 100000000)]
    dfb = pd.concat([dfb, selected_rows.reset_index(drop=True)])
    df = df.drop(selected_rows.index)

    # 3.2
    # si le montant commence par 123456789 alors INEXPLOITABLE
    dfb = pd.concat([dfb, df[(df[col].astype(str).str[0:9] == "123456789")]])
    df = df[(df[col].astype(str).str[0:9] != "123456789")]

    return df, dfb


def check_siret(df: pd.DataFrame, dfb: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Le SIRET comprend 14 caractères (9 pour le SIREN + 5 pour le NIC) – format texte pour ne pas supprimer les « 0 » en début de Siret.
    L’identifiant autorité concédante est INEXPLOITABLE s’il ne respecte pas le format.
    Si INEXPLOITABLE, le contrat est mis de côté.
    """
    dfb = pd.concat([dfb, df[~df[col].astype(str).str.match(
        "^[0-9]{14}$")]])
    df = df[df[col].astype(str).str.match("^[0-9]{14}$")]

    return df, dfb


def check_id(df: pd.DataFrame, dfb: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    L’identifiant du contrat de concession comprend :
        - 4 caractères pour l’année de notification
        - 1 à 10 caractères pour le numéro interne
        - 2 caractères pour le numéro d’ordre de la modification
    Le numéro d’identification est INEXPLOITABLE s’il ne respecte pas le format.
    """

    def validate_contract_identifier(identifier):
        pattern = r'^\d{4}[A-Z0-9]{1,10}\d{2}$'  # Regex pattern for the identifier format
        return bool(re.match(pattern, identifier))

    dfb = pd.concat([dfb, df[~df[col].astype(str).apply(validate_contract_identifier)]])
    df = df[df[col].astype(str).apply(validate_contract_identifier)]
    return df, dfb


def check_duree_contrat(df: pd.DataFrame, dfb: pd.DataFrame, month: int) -> pd.DataFrame:
    """
    Si durée en mois > month alors INEXPLOITABLE
    Si durée en mois = 0 alors INEXPLOITABLE
    """
    df["dureeMois"] = df["dureeMois"].astype(float)

    dfb = pd.concat([dfb, df[df["dureeMois"] > month]])
    df = df[df["dureeMois"] <= month]

    dfb = pd.concat([dfb, df[df["dureeMois"] == 0]])
    df = df[df["dureeMois"] != 0]

    return df, dfb


if __name__ == '__main__':
    main()
