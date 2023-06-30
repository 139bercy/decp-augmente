import cProfile
import json
import os
import pickle
import logging.handlers
import re

import argparse
import numpy as np
import pandas as pd
import utils
import recuperation_data
import time

logger = logging.getLogger("main.nettoyage2")
logger.setLevel(logging.DEBUG)
pd.options.mode.chained_assignment = None  # default='warn'


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper


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
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--test", help="run script in test mode with a small sample of data")
        args = parser.parse_args()

        with open(os.path.join(path_to_data, "decpv2.pkl"), 'rb') as f:
            df = pickle.load(f)

        if args.test:
            df = df.sample(n=10000, random_state=1)
            logger.info("Mode test activé")

    logger.info("Nettoyage des données")
    manage_data_quality(df)


@measure_execution_time
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

    print("Concession valides : ", str(df_concession.shape[0]))
    print("Concession mauvaises : ", str(df_concession_badlines.shape[0]))
    print("Concession mal rempli % : ", str((df_concession_badlines.shape[0] / (df_concession.shape[0] + df_concession_badlines.shape[0]) * 100)))

    print("Marchés valides : ", str(df_marche.shape[0]))
    print("Marché mauvais : ", str(df_marche_badlines.shape[0]))
    print("Marché mal rempli % : ", str((df_marche_badlines.shape[0] / (df_marche.shape[0] + df_marche_badlines.shape[0]) * 100)))

    # save data to csv files
    df_concession.to_csv(os.path.join(conf_data["path_to_data"], "concession.csv"), index=False)
    df_marche.to_csv(os.path.join(conf_data["path_to_data"], "marche.csv"), index=False)
    df_marche_badlines.to_csv(os.path.join(conf_data["path_to_data"], "marche_exclu.csv"), index=False)
    df_concession_badlines.to_csv(os.path.join(conf_data["path_to_data"], "concession_exclu.csv"), index=False)

    # Concaténation des dataframes pour l'enrigissement (re-séparation après)
    df = pd.concat([df_concession, df_marche])

    return df

@measure_execution_time
def regles_marche(df_marche_: pd.DataFrame) -> pd.DataFrame:
    df_marche_badlines_ = pd.DataFrame(columns=df_marche_.columns)

    @measure_execution_time
    def dedoublonnage_marche(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sont considérés comme doublons des marchés ayant les mêmes valeurs aux champs suivants :
        id,
        idAcheteur,
        idTitulaire,
        dateNotification,
        Montant
        En clair cela signifie que c’est bel et bien le même contrat.
        - Si même (id, idAcheteur, idTitulaire, dateNotification, Montant), regarder datePublicationDonnees, qui correspond à la date d’arrivée de la donnée dans data.gouv. Conserver seulement l’enregistrement ayant la datePublicationDonnees la plus récente.
        - Si même datePublicationDonnees en plus de même jeu de variable, alors regarder le niveau de complétude de chaque enregistrement avec un score ( : compter le nombre de fois où les variables sont renseignées pour chaque enregistrement. Cela constitue un « score »). Prendre l’enregistrement ayant le score le plus élevé.
        - Si même (id, idAcheteur, idTitulaire, dateNotification, Montant, datePublicationDonnees ) et même score, alors garder la dernière ligne du groupe par défaut
        """

        def extract_values(row: list):
            """
            create 9 new columns with the values of the titulaires column

            template for new col name : titulaires_ + col name + _ + value
                - value is number from 1 to 3
                - col name are : typeIdentifiant, id, denominationSociale

            row contains a list of dict, each dict is a titulaires
                - can be empty
                - can contain 1, 2 or 3 titulaires or more keeping only 3 first
                - if 1 value can be a dict and not a list of dict

            :param row: the dataframe row to extract values from
            :return: a new dataframe with the values of the titulaires column, new value are nan if not present
            """
            new_columns = {}

            # create new columns all with nan value
            for value in range(1, 4):
                for col_name in ['denominationSociale', 'id', 'typeIdentifiant']:
                    new_col_name = f'titulaire_{col_name}_{value}'
                    new_columns[new_col_name] = np.nan

            if isinstance(row, list):
                row = row[:3]  # Keep only the first three concession
            else:
                # if row is not a list, then it is empty and for obscure reason script thinks it's a float so returning nan
                return pd.Series(new_columns)

            # fill new columns with values from concessionnaires column if exist
            for value, concession in enumerate(row, start=1):
                # replace value in new_columns by corresponding value in concession
                for col_name in ['denominationSociale', 'id', 'typeIdentifiant']:
                    col_to_fill = f'titulaire_{col_name}_{value}'
                    # col_name is key in concession dict, col_to_fill is key in new_columns dict. get key value in col_name and put it in col_to_fill
                    if concession:
                        new_columns[col_to_fill] = concession.get(col_name, np.nan)

            return pd.Series(new_columns)

        df = df["titulaires"].apply(extract_values).join(df)

        df.drop(columns=["titulaires"], inplace=True)

        logging.info("dedoublonnage_marche")
        print("df_marché avant dédoublonnage : " + str(df.shape))
        # filtre pour mettre la date de publication la plus récente en premier
        df = df.sort_values(by=["datePublicationDonnees"], ascending=False)

        # suppression des doublons en gardant la première ligne donc datePublicationDonnees la plus récente
        dff = df.drop_duplicates(subset=["id", "acheteur.id", "titulaire_id_1", "montant", "dureeMois"], keep="first")

        print("df_marché après dédoublonnage : " + str(dff.shape))
        print("% de doublons marché : ", str((df.shape[0] - dff.shape[0]) / df.shape[0] * 100))
        return dff

    def marche_check_empty(df: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
        col_name = ["id", "acheteur.id", "montant", "titulaire_id_1", "dureeMois"]  # titulaire contient un dict avec des valeurs dont id
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

    @measure_execution_time
    def marche_cpv(df: pd.DataFrame, cpv_2008_df: pd.DataFrame) -> pd.DataFrame:
        """
        Le CPV comprend 10 caractères (8 pour la racine + 1 pour le séparateur « - » et +1 pour la clé) – format texte pour ne pas supprimer les « 0 » en début de CPV.
        Un code CPV est INEXPLOITABLE s’il n’appartient pas à la liste des codes CPV existants dans la nomenclature européenne 2008 des CPV
        Les CPV fonctionnent en arborescence. Le CPV le plus générique est le premier de la liste d’une division. Il y a 45 divisions (03, 09, 14, 15, 16,18…).
        En lisant de gauche à droite, le code CPV le plus générique de la division comportera un « 0 » au niveau du 3ᵉ caractère.
        Ex pour la division 45 : CPV le plus générique : 45000000-7 (travaux de construction)
        Règles :
            - Si la clé du code CPV est manquante et que la racine du code CPV est correcte (8 premiers caractères) alors il convient de compléter avec la clé correspondante issue de la base CPV 2008.
            - Si la racine du code CPV est complète, mais qu’elle n’existe pas dans la base CPV 2008, alors il convient de prendre le code CPV le plus générique de son arborescence.
            - Si la racine du code CPV est correcte, mais que la clé est incorrecte, alors il convient de remplacer par la clé correspondante à la racine issue de la base CPV 2008.
            - Si la racine du code CPV est incomplète, mais qu’au moins les deux premiers caractères du code CPV (la division) sont renseignées correctement, alors il convient de compléter avec le code CPV le plus générique de la division
            - Si le code CPV n’est pas renseigné, mais qu’il y a un objet de marché, il convient de laisser la donnée initiale et de ne pas mettre de côté le marché.
        AUCUN RETRAITEMENT POSSIBLE :
            - Si la racine du code CPV est incomplète, qu’aucun objet de marché n’est présent et que les deux premiers caractères du code CPV sont erronés, alors aucun retraitement n’est possible et l’enregistrement est mis de côté (ex : 111111).
            - Si la racine du code CPV est complète, mais erronée, qu’aucun objet de marché n’est présent et que les deux premiers caractères du code CPV sont erronés, alors aucun retraitement n’est possible et l’enregistrement est mis de côté (ex : 11111111-1).
        Parameters :
            df (pd.DataFrame): dataframe to clean
        Returns :
            df (pd.DataFrame): cleaned dataframe
        """
        def get_cpv_key(cpv_root):
            # check if CPV root exists in CPV 2008 database column "CODE" and only keep the first 8 characters
            cpv_mask = cpv_2008_df["CODE"].str[:8] == cpv_root
            cpv_key = cpv_2008_df.loc[cpv_mask, "CODE"].str[-1].values[0] if cpv_mask.any() else ""
            return cpv_key

        cpv_2008_df["CPV Root"] = cpv_2008_df["CODE"].str[:8]

        # Check if CPV is empty string
        empty_cpv_mask = df['codeCPV'] == ''
        df.loc[empty_cpv_mask, 'CPV'] = df.loc[empty_cpv_mask, 'codeCPV']

        # Check if CPV root is complete
        complete_root_mask = df['codeCPV'].str.len() == 8
        cpv_roots = df.loc[complete_root_mask, 'codeCPV'].str[:8]
        non_existing_roots_mask = ~cpv_roots.isin(cpv_2008_df["CPV Root"].values)
        cpv_roots.loc[non_existing_roots_mask] = cpv_roots.loc[non_existing_roots_mask].str[:2] + '000000'
        df.loc[complete_root_mask, 'CPV'] = cpv_roots + '-' + df.loc[complete_root_mask, 'codeCPV'].str[9:]

        # Check if CPV key is missing only if CPV root is complete
        missing_key_mask = (df['codeCPV'].str.len() >= 8) & (df['codeCPV'].str[9:].isin(['', None]))
        df.loc[missing_key_mask, 'CPV'] = (
            df.loc[missing_key_mask, 'codeCPV'].str[:8].apply(get_cpv_key)
        )

        return df

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

    df_marche_ = dedoublonnage_marche(df_marche_)

    df_marche_, df_marche_badlines_ = marche_check_empty(df_marche_, df_marche_badlines_)
    df_marche_, df_marche_badlines_ = marche_cpv_object(df_marche_, df_marche_badlines_)
    df_marche_, df_marche_badlines_ = marche_date(df_marche_, df_marche_badlines_)

    df_marche_, df_marche_badlines_ = check_montant(df_marche_, df_marche_badlines_, "montant")
    df_marche_, df_marche_badlines_ = check_siret(df_marche_, df_marche_badlines_, "acheteur.id")
    df_marche_, df_marche_badlines_ = check_siret(df_marche_, df_marche_badlines_, "titulaire_id_1")

    df_cpv = pd.read_excel("data/cpv_2008_ver_2013.xlsx", engine="openpyxl")
    df_marche_ = marche_cpv(df_marche_, df_cpv)
     # delete df_cpv to free memory
    del df_cpv

    df_marche_, df_marche_badlines_ = check_duree_contrat(df_marche_, df_marche_badlines_, 180)
    df_marche_, df_marche_badlines_ = marche_dateNotification(df_marche_, df_marche_badlines_)

    return df_marche_, df_marche_badlines_


@measure_execution_time
def regles_concession(df_concession_: pd.DataFrame) -> pd.DataFrame:

    @measure_execution_time
    def dedoublonnage_concession(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sont considérés comme doublons des concessions ayant les mêmes valeurs aux champs suivants :
        id,
        idautoriteConcedante,
        idconcessionnaires,
        dateDebutExecution,
        valeurGlobale.
        En clair cela signifie que c’est bel et bien le même contrat.
        - Si même (id, idautoriteConcedante, idconcessionnaires, dateDebutExecution, valeurGlobale), regarder datePublicationDonnees, qui correspond à la date d’arrivée de la donnée dans data.gouv. Garder datePublicationDonnees la plus récente.
        - Si même datePublicationDonnees en plus de même jeu de variable, alors regarder le niveau de complétude de chaque enregistrement avec un score ( : compter le nombre de fois où les variables sont renseignées pour chaque enregistrement. Cela constitue un « score »). Prendre l’enregistrement ayant le score le plus élevé.
        - Si même (id, idautoriteConcedante, idconcessionnaires, dateDebutExecution, valeurGlobale, datePublicationDonnees) et même score, alors garder la dernière ligne du groupe.
        """

        def extract_values(row: list):
            """
            create 9 new columns with the values of the concessionnaires column

            template for new col name : concessionnaire_ + col name + _ + value
                - value is number from 1 to 3
                - col name are : denominationSociale, id, typeIdentifiant

            row contains a list of dict, each dict is a concessionnaire
                - can be empty
                - can contain 1, 2 or 3 concessionnaires or more keeping only 3 first
                - if 1 value can be a dict and not a list of dict

            :param row: the dataframe row to extract values from
            :return: a new dataframe with the values of the concessionnaires column, new value are nan if not present
            """
            new_columns = {}

            # create new columns all with nan value
            for value in range(1, 4):
                for col_name in ['denominationSociale', 'id', 'typeIdentifiant']:
                    new_col_name = f'concessionnaire_{col_name}_{value}'
                    new_columns[new_col_name] = np.nan

            if isinstance(row, list):
                # how is the list of concessionnaires
                # if contain a dict where key is exactly : concessionnaire, then the list we want is the value of this dict key
                if 'concessionnaire' in row[0].keys():
                    row = [item['concessionnaire'] for item in row]
                row = row[:3]  # Keep only the first three concession
            else:
                # if row is not a list, then it is empty and for obscure reason script thinks it's a float so returning nan
                return pd.Series(new_columns)

            # le traitement ici à lieux car comme on dit : "Garbage in, garbage out" mais on est gentil on corrige leurs formats -_-
            # check if row is a list of list of dict, if so, keep only the first list
            if isinstance(row[0], list):
                row = row[0]

            # fill new columns with values from concessionnaires column if exist
            for value, concession in enumerate(row, start=1):
                # replace value in new_columns by corresponding value in concession
                for col_name in ['denominationSociale', 'id', 'typeIdentifiant']:
                    col_to_fill = f'concessionnaire_{col_name}_{value}'
                    # col_name is key in concession dict, col_to_fill is key in new_columns dict. get key value in col_name and put it in col_to_fill
                    if concession:
                        new_columns[col_to_fill] = concession.get(col_name, np.nan)

            return pd.Series(new_columns)


        df = df["concessionnaires"].apply(extract_values).join(df)

        df.drop(columns=["concessionnaires"], inplace=True)

        logging.info("dedoublonnage_concession")
        print("df_concession_ avant dédoublonnage : " + str(df_concession_.shape))
        # filtre pour mettre la date de publication la plus récente en premier
        df = df.sort_values(by=["datePublicationDonnees"], ascending=[False])

        # suppression des doublons en gardant la première ligne donc datePublicationDonnees la plus récente
        dff = df.drop_duplicates(subset=["id", "autoriteConcedante.id", "dateDebutExecution", "concessionnaire_id_1","valeurGlobale"],
                                                            keep="first")
        print("df_concession_ après dédoublonnage : " + str(df.shape))
        print("% doublon concession : ", str((df.shape[0] - dff.shape[0]) / df.shape[0] * 100))
        return dff

    df_concession_badlines_ = pd.DataFrame(columns=df_concession_.columns)

    def concession_check_empty(df_con: pd.DataFrame, df_bad: pd.DataFrame) -> pd.DataFrame:
        col_name = ["id", "autoriteConcedante.id", "concessionnaire_id_1", "objet", "valeurGlobale",
                    "dureeMois"]
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

    def concession_dateDebutExecution(df: pd.DataFrame) -> pd.DataFrame:
        """
        Format AAAA-MM-JJ
            Si MM<01 ou>12,
            SI JJ<01 ou >31 (voir si possibilité de vérifier le format jour max en fonction du mois et année)
        Si la date de début d’exécution du contrat de concession est manquante et qu’il existe une date de publication des données d’exécution, respectant le format AAAA-MM-JJ (ou pouvant être retransformé en ce format) alors il convient d’affecter la date de publication à la date de début d’exécution.
        """

        # vérification du format de la date de notification (AAAA-MM-JJ) et correction si besoin création d'un dataframe avec les lignes à corriger
        df["dateDebutExecution"] = pd.to_datetime(df["dateDebutExecution"], format='%Y-%m-%d', errors='ignore')
        df["datePublication"] = pd.to_datetime(df["datePublication"], format='%Y-%m-%d', errors='ignore')

        # si la date de début d'exécution n'est pas au format AAAA-MM-JJ regarder la date de publication et si elle est au format AAAA-MM-JJ alors mettre la date de publication dans la date de début d'exécution
        df.loc[(df["dateDebutExecution"].isnull()) & (df["datePublication"].notnull()), "dateDebutExecution"] = df["datePublication"]

        return df

    df_concession_ = dedoublonnage_concession(df_concession_)

    df_concession_, df_concession_badlines_ = concession_check_empty(df_concession_, df_concession_badlines_)
    df_concession_, df_concession_badlines_ = concession_date(df_concession_, df_concession_badlines_)

    df_concession_, df_concession_badlines_ = check_montant(df_concession_, df_concession_badlines_, "valeurGlobale")
    df_concession_, df_concession_badlines_ = check_siret(df_concession_, df_concession_badlines_, "autoriteConcedante.id")
    df_concession_, df_concession_badlines_ = check_siret(df_concession_, df_concession_badlines_, "concessionnaire_id_1")

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
