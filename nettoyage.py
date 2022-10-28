import json
import os
import pickle
import logging.handlers
from random import random
import numpy as np
import pandas as pd
import itertools
from pandas import json_normalize

import cProfile
import pstats


pd.options.mode.chained_assignment = None  # default='warn'

with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)
path_to_data = conf_data["path_to_data"]
decp_file_name = conf_data["decp_file_name"]

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

with open(os.path.join("confs", "var_debug.json")) as f:
    conf_debug = json.load(f)["nettoyage"]

logger = logging.getLogger("main.nettoyage")
logger.setLevel(logging.DEBUG)


def main():
    check_reference_files()
    logger.info("Ouverture du fichier decp.json")
    with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
        data = json.load(json_data)


    # Modification pour un prendre subset de données 

    if conf_debug["subset"]:
        n_data = len(data["marches"])
        n_subset = conf_debug["n_subset"]
        logger.info(
            "Subset étant True on se restreint à un dataframe de taille n_subset soit  {} lignes choisis aléatoirement".format(
                n_subset))
        if conf_debug["debug"]:
            logger.info("Mode debug, pas d'alea")
            seed = conf_debug["seed"]
            np.random.seed(seed)
            random_i = list(np.random.choice(n_data, n_subset))
        else:
            random_i = list(np.random.choice(n_data, n_subset))

        accessed_mapping = map(data['marches'].__getitem__, random_i)
        accessed_list = list(accessed_mapping)
        data['marches'] = accessed_list

    logger.info("Début du traitement: Conversion des données en pandas")
    df = manage_modifications(data)
    logger.info("Fin du traitement")

    df = regroupement_marche_complet(df)

    logger.info("Début du traitement: Gestion des titulaires")
    df = (df.pipe(manage_titulaires)
          .pipe(manage_duplicates)
          .pipe(manage_amount)
          .pipe(manage_missing_code)
          .pipe(manage_region)
          .pipe(manage_date)
          .pipe(correct_date)
          .pipe(data_inputation)
          .pipe(replace_char)
          )
    logger.info("Fin du traitement")

    logger.info("Creation csv intermédiaire: decp_nettoye.csv")
    with open('df_nettoye', 'wb') as df_nettoye:
        # Export présent pour faciliter l'utilisation du module enrichissement.py
        pickle.dump(df, df_nettoye)
    df.to_csv("decp_nettoye.csv")
    logger.info("Ecriture du csv terminé")


def check_reference_files():
    """
    Vérifie la présence des fichiers datas nécessaires, dans le dossier data.
        StockEtablissement_utf8.csv, cpv_2008_ver_2013.xlsx, geoflar-communes-2015.csv,
        departement2020.csv, region2020.csv, StockUniteLegale_utf8.csv
    """
    path_data = conf_data["path_to_data"]

    l_key_useless = ["path_to_project", "path_to_data", "path_to_cache", "cache_bdd_insee",
                     "cache_not_in_bdd_insee",
                     "cache_bdd_legale",
                     "cache_not_in_bdd_legale"]

    path = os.path.join(os.getcwd(), path_data)
    for key in list(conf_data.keys()):
        if key not in l_key_useless:
            logger.info(f'Test du fichier {conf_data[key]}')
            mask = os.path.exists(os.path.join(path, conf_data[key]))
            if not mask:
                logger.error(f"Le fichier {conf_data[key]} n'existe pas")
                raise ValueError(f"Le fichier data: {conf_data[key]} n'a pas été trouvé")


def manage_titulaires(df: pd.DataFrame):
    # Ecriture dans les logs
    logger.info(f"Nombre de marché sans titulaires: {sum(df['titulaires'].isnull())}. "
                f"Remplacé par la valeur du concessionnaire")
    logger.info(f"Nombre de marché sans montant: {sum(df['montant'].isnull())}. Remplacé par la valeur globale")
    logger.info(f"Nombre de marché sans identifiant acheteur: {sum(df['acheteur.id'].isnull())}. "
                f"Remplacé par l'identifiatn de l'autorité Concedante")
    logger.info(f"Nombre de marché sans nom d'acheteur: {sum(df['acheteur.nom'].isnull())}. "
                f"Remplacé par le nom de l'autorité Concédante")

    # Gestion différences concessionnaires / titulaires
    df.titulaires = np.where(df["titulaires"].isnull(), df.concessionnaires, df.titulaires)
    df.montant = np.where(df["montant"].isnull(), df.valeurGlobale, df.montant)
    df['acheteur.id'] = np.where(df['acheteur.id'].isnull(), df['autoriteConcedante.id'], df['acheteur.id'])
    df['acheteur.nom'] = np.where(df['acheteur.nom'].isnull(), df['autoriteConcedante.nom'], df['acheteur.nom'])
    useless_columns = ['dateSignature', 'dateDebutExecution', 'valeurGlobale', 'donneesExecution', 'concessionnaires',
                        'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom',
                        'idtech', "id_technique"]
    df.drop(columns=useless_columns, inplace=True)

    # Récupération des données titulaires
    df = df[~(df['titulaires'].isna())]

    # Création d'une colonne nbTitulairesSurCeMarche.
    # Cette colonne sera retravaillé dans la fonction detection_accord_cadre
    df.loc[:, "nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x: len(x))

    df_titulaires = pd.concat([pd.DataFrame.from_records(x) for x in df['titulaires']],
                              keys=df.index).reset_index(level=1, drop=True)
    df_titulaires.rename(columns={"id": "idTitulaires"}, inplace=True)
    df = df.drop('titulaires', axis=1).join(df_titulaires).reset_index(drop=True)

    return df


def manage_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permet la suppression des eventuels doublons pour un subset précis du dataframe

    Retour:
        pd.DataFrame
    """
    logger.info("Début du traitement: Suppression des doublons")
    nb_ligne_avant_suppression = len(df)
    df.drop_duplicates(subset=['source', '_type', 'nature', 'procedure', 'dureeMois',
                               'datePublicationDonnees', 'lieuExecution.code', 'lieuExecution.typeCode',
                               'lieuExecution.nom', 'id', 'objet', 'codeCPV', 'dateNotification', 'montant',
                               'formePrix', 'acheteur.id', 'acheteur.nom', 'typeIdentifiant', 'idTitulaires',
                               'denominationSociale'],
                       keep='first',
                       inplace=True)
    df.reset_index(inplace=True, drop=True)
    nb_ligne_apres_suppresion = len(df)

    # Ecriture dans les logs
    logger.info(f"Nombre de lignes doublons supprimées: {nb_ligne_avant_suppression - nb_ligne_apres_suppresion}")

    # Correction afin que ces variables soient représentées identiquement
    df['formePrix'] = np.where(df['formePrix'].isna(), np.nan, df['formePrix'])
    df['formePrix'] = np.where('Ferme, actualisable' == df['formePrix'], 'Ferme et actualisable', df['formePrix'])
    df['procedure'] = np.where('Appel d’offres restreint' == df['procedure'], "Appel d'offres restreint",
                               df['procedure'])

    logger.info("Fin du traitement")
    return df


def is_false_amount(x: float, threshold: int = 5) -> bool:
    """
    On cherche à vérifier si les parties entières des montants sont composées d'au moins 5 threshold fois le meme chiffre
    (hors 0).
    Exemple pour threshold = 5: 999 999 ou 222 262.
    Ces montants seront considérés comme faux
    """
    # Création d'une liste compteur
    d = [0] * 10
    str_x = str(abs(int(x)))
    for c in str_x:
        # On compte le nombre de fois que chaque chiffre apparait
        d[int(c)] += 1
    for counter in d[1:]:
        if counter > threshold:
            return True
    return False


def manage_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Travail sur la détection des montants erronés. Ici inférieur à 200, supérieur à 9.99e8 et
    si la partie entière du montant est composé d'au moins 5 fois le même chiffre hors 0.
    Exemple de montant érronés:
        - 999 999
        - 222 522

    Retour:
        pd.DataFrame
    """

    logger.info("Début du traitement: Détection et correction des montants aberrants")
    # Identifier les outliers - travail sur les montants
    df["montant"] = pd.to_numeric(df["montant"], downcast='float') # Passage en float32 plutôt que 64
    df['montantCalcule'] = df["montant"]
    df['montantCalcule'].fillna(0, inplace=True)
    # variable témoin pour les logs

    values_montant_calcul = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero = values_montant_calcul[0] if (0 in values_montant_calcul.keys()) else 0

    # Détection des montants "1 chiffre"
    df["montantCalcule"] = df["montantCalcule"].apply(lambda x: 0 if is_false_amount(x) else abs(x))

    values_montant_calcul2 = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero_2 = values_montant_calcul2[0] if (0 in values_montant_calcul2.keys()) else 0
    logger.info(f"{n_montant_calcul_equal_zero_2 - n_montant_calcul_equal_zero} montant(s) correspondaient à des"
                f"suites d'un seul chiffre. Exemple: 9 999 999")
    
    #Actualisation de la variable après la modification de df
    values_montant_calcul = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero = values_montant_calcul[0] if (0 in values_montant_calcul.keys()) else 0

    # Définition des bornes inf et sup et traitement
    borne_inf = 200.0
    borne_sup = 9.99e8
    df["montantCalcule"] = df["montantCalcule"] / df["nbTitulairesSurCeMarche"]
    df['montantCalcule'] = np.where(df['montantCalcule'] <= borne_inf, 0, df['montantCalcule'])

    
    #Actualisation de la variable après la modification de df
    values_montant_calcul2 = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero_2 = values_montant_calcul2[0] if (0 in values_montant_calcul2.keys()) else 0
    logger.info(f"{n_montant_calcul_equal_zero_2 - n_montant_calcul_equal_zero}"
                f" montant(s) étaient inférieurs à la borne inf {borne_inf}")
    #Actualisation de la variable après la modification de df
    values_montant_calcul = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero = values_montant_calcul[0] if (0 in values_montant_calcul.keys()) else 0
    df['montantCalcule'] = np.where(df['montantCalcule'] >= borne_sup, 0, df['montantCalcule'])
    #Actualisation de la variable après la modification de df
    values_montant_calcul2 = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero_2 = values_montant_calcul2[0] if (0 in values_montant_calcul2.keys()) else 0
    logger.info(f"{n_montant_calcul_equal_zero_2 - n_montant_calcul_equal_zero} montant(s) étaient supérieurs à "

                f"la borne sup: {borne_sup}")
    # Colonne supplémentaire pour indiquer si la valeur est estimée ou non
    df['montantEstime'] = np.where(df['montantCalcule'] != df.montant, True, False)
    # Ecriture dans la log

    values_montant_calcul2 = df.montantCalcule.value_counts()
    n_montant_calcul_equal_zero_2 = values_montant_calcul2[0] if (0 in values_montant_calcul2.keys()) else 0
    logger.info(f"Au total, {n_montant_calcul_equal_zero_2} montant(s) "

                f"ont été corrigé (on compte aussi les montants vides).")
    logger.info("Fin du traitement")
    return df


def manage_missing_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mise en qualité des variables identifiantes.
    Cf Readme (copier coller la partie du Readme correspondante)

    Retour:
        pd.DataFrame
    """

    logger.info("Début du traitement: Gestion des Id null")
    # Ecriture dans les logs
    logger.info("Nombre d'identifiant manquants et remplacés: {}".format(sum(df["id"].isnull())))
    logger.info("Nombre de code CPV manquants et remplacés: {}".format(sum(df["codeCPV"].isnull())))

    # Les id et codeCPV manquants sont remplacés par des '0'. Dans le cas de id, la fonction [insérer nom de la fonction
    # présente dans la MR Gestion_ID]
    # Permet le retraitement de la variable pour la rendre unique
    df.id = np.where(df["id"].isnull(), '0000000000000000', df.id)
    df.codeCPV = np.where(df["codeCPV"].isnull(), '00000000', df.codeCPV)

    # Nettoyage des caractères spéciaux dans codes idTitulaires
    logger.info("Nettoyage des idTitualires")
    caracteres_speciaux_dict = conf_glob["nettoyage"]["caractere_speciaux"]
    mask = (df.typeIdentifiant == 'SIRET') | \
           (df.typeIdentifiant.isnull()) | \
           (df.typeIdentifiant == 'nan')
    df.idTitulaires[mask].replace(caracteres_speciaux_dict, inplace=True)
    df.idTitulaires = np.where(df.idTitulaires == '', np.NaN, df.idTitulaires)
    # Ecriture dans les logs
    logger.info(f"Nombre d'identifiant titualire ou un traitement sur les caractères spéciaux a été fait: {sum(mask)}")

    # Récupération code NIC: 5 dernier chiffres du Siret <- idTitulaires
    logger.info("Récupération du code NIC")
    df.idTitulaires = df.idTitulaires.astype(str)
    df['nic'] = df["idTitulaires"].str[-5:]
    df.nic = np.where(~df["nic"].str.isdigit(), np.NaN, df.nic)
    df['nic'] = df.nic.astype(str)

    # Récupération de ce qu'on appelle la division du marché selon la nomenclature européenne CPV.
    # Ajout du nom de la division
    logger.info("Récupération de la division du code CPV.")
    df.codeCPV = df.codeCPV.astype(str)
    df["CPV_min"] = df["codeCPV"].str[:2]
    df["natureObjet"] = "Fournitures"
    df.loc[df["CPV_min"] == '45', 'natureObjet'] = 'Travaux'
    df.loc[df["CPV_min"] > '45', 'natureObjet'] = 'Services'

    # Mise en forme des données vides
    logger.info("Mise en forme des données vides de la colonne denominationSociale")
    df.denominationSociale = np.where(
        (df.denominationSociale == 'N/A') | (df.denominationSociale == 'null'),
        np.NaN, df.denominationSociale)
    logger.info('Fin du traitement')

    return df


def manage_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout des libellés Régions/Département pour le lieu d'execution du marché

    Retour:
        pd.DataFrame
    """

    logger.info("Début du traitement: Attribution et correction des régions/déprtements (code + libelle). "
                "Zone d'execution du marché")
    # Régions / Départements #
    # Création de la colonne pour distinguer les départements
    logger.info("Création de la colonne département Execution")
    df['codeDepartementExecution'] = df['lieuExecution.code'].str[:3]
    liste_correspondance = conf_glob["nettoyage"]["DOM2name"]
    df['codeDepartementExecution'].replace(liste_correspondance, inplace=True)

    df['codeDepartementExecution'] = df['codeDepartementExecution'].str[:2]

    liste_correspondance_2 = conf_glob["nettoyage"]["name2DOMCOM"]
    df['codeDepartementExecution'].replace(liste_correspondance_2, inplace=True)

    # Vérification si c'est bien un code département
    liste_cp = conf_glob["nettoyage"]["code_CP"].split(',') \
               + [str(i) for i in list(np.arange(10, 96, 1))]
    df['codeDepartementExecution'] = np.where(~df['codeDepartementExecution'].isin(liste_cp), np.NaN,
                                              df['codeDepartementExecution'])

    # Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)
    df['lieuExecution.typeCode'] = np.where(df['lieuExecution.typeCode'].isna(), np.NaN, df['lieuExecution.typeCode'])
    df['codeDepartementExecution'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN,
                                              df['codeDepartementExecution'])

    # Récupération des codes régions via le département
    logger.info("Ajout des code regions pour le lieu d'execution")
    path_dep = os.path.join(path_to_data, conf_data["departements-francais"])
    departement = pd.read_csv(path_dep, sep=",", usecols=['dep', 'reg', 'libelle'], dtype={"dep": str, "reg": str,
                                                                                           "libelle": str})
    df['codeDepartementExecution'] = df['codeDepartementExecution'].astype(str)
    df = pd.merge(df, departement, how="left",
                  left_on="codeDepartementExecution", right_on="dep")
    df.rename(columns={"reg": "codeRegionExecution"}, inplace=True)
    # On supprime la colonne dep, doublon avec codeDepartementExecution
    del df["dep"]
    # Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
    df['codeRegionExecution'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'],
                                         df['codeRegionExecution'])
    df['codeRegionExecution'] = df['codeRegionExecution'].astype(str)
    # Vérification des codes région
    liste_reg = conf_glob["nettoyage"]["code_reg"].split(',')  # 98 = collectivité d'outre mer

    df['codeRegionExecution'] = np.where(~df['codeRegionExecution'].isin(liste_reg), np.NaN, df['codeRegionExecution'])
    # Identification du nom des régions
    df['codeRegionExecution'] = df['codeRegionExecution'].astype(str)

    # Import de la base region de l'Insee
    logger.info("Ajout du libelle des regions d'execution")
    path_reg = os.path.join(path_to_data, conf_data["region-fr"])
    region = pd.read_csv(path_reg, sep=",", usecols=["reg", "libelle"], dtype={"reg": str, "libelle": str})
    region.columns = ["reg", "libelle_reg"]

    df = pd.merge(df, region, how="left",
                  left_on="codeRegionExecution", right_on="reg")
    df.rename(columns={"libelle_reg": "libelleRegionExecution"}, inplace=True)
    # On supprime la colonne reg, doublon avec codeRegionExecution
    del df["reg"]
    logger.info("Fin du traitement")
    return df


def manage_date(df: pd.DataFrame) -> pd.DataFrame:
    """
        Récupération de l'année de notification du marché ainsi que son mois à partir de la variable dateNotification.

    Retour:
        - pd.DataFrame
    """

    logger.info("Début du traitement: Récupération de l'année et du mois du marché public + "
                "Correction des années aberrantes")
    # Date / Temps #
    # ..............Travail sur les variables de type date
    df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
    df.dateNotification = df.dateNotification.str[0:10]
    # On récupère l'année de notification
    logger.info("Récupération de l'année")
    df['anneeNotification'] = df.dateNotification.str[0:4]
    df['anneeNotification'] = df['anneeNotification'].astype(float)
    # On supprime les erreurs (0021 ou 2100 par exemple)
    df['dateNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['dateNotification'])
    df['dateNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['dateNotification'])
    df['anneeNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['anneeNotification'])
    df['anneeNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['anneeNotification'])
    logger.info("Au total, {} marchés avaient une année érronée".format(sum(df["anneeNotification"].isna())))
    df['anneeNotification'] = df.anneeNotification.astype(str).str[:4]

    # On récupère le mois de notification
    logger.info("Récupération du mois")
    df['moisNotification'] = df.dateNotification.str[5:7]
    df.datePublicationDonnees = np.where(df.datePublicationDonnees == '', np.NaN, df.datePublicationDonnees)
    logger.info(f"Au total, {sum(df['datePublicationDonnees'].isna())} marchés n'ont pas de date de publication des"
                "données connue")
    logger.info("Fin du traitement")

    return df


def correct_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Travail sur les durées des contrats. Recherche des durées exprimées en Jour et non pas en mois

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Correction de la variable dureeMois.")
    # On cherche les éventuelles erreurs mois -> jours
    df['montantCalcule'] = df['montantCalcule'].astype(np.int32) # 32 au lieu de 64 pour l'espace mémoire
    df['dureeMois'] = df['dureeMois'].astype(np.int32) # 32 au lieu de 64
    mask = ((df['montantCalcule'] == df['dureeMois'])
            | (df['montantCalcule'] / df['dureeMois'] < 100)
            | (df['montantCalcule'] / df['dureeMois'] < 1000) & (df['dureeMois'] >= 12)
            | ((df['dureeMois'] == 30) & (df['montantCalcule'] < 200000))
            | ((df['dureeMois'] == 31) & (df['montantCalcule'] < 200000))
            | ((df['dureeMois'] == 360) & (df['montantCalcule'] < 10000000))
            | ((df['dureeMois'] == 365) & (df['montantCalcule'] < 10000000))
            | ((df['dureeMois'] == 366) & (df['montantCalcule'] < 10000000))
            | ((df['dureeMois'] > 120) & (df['montantCalcule'] < 2000000)))

    df['dureeMoisEstimee'] = np.where(mask, "True", "False")

    # On corrige pour les colonnes considérées comme aberrantes, on divise par 30 (nombre de jours par mois)
    df['dureeMoisCalculee'] = np.where(mask, round(df['dureeMois'] / 30, 0), df['dureeMois'])
    # Comme certaines valeurs atteignent zero, on remplace par un mois
    # Il y a une valeur négative
    df['dureeMoisCalculee'] = np.where(df['dureeMoisCalculee'] <= 0, 1, df['dureeMoisCalculee'])
    logger.info(f"Au total, {sum(mask)} duree de marché en mois ont été jugées rentrées en jour et non en mois.")
    logger.info("Fin du traitement")

    return df


def data_inputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permet une estimation de la dureeMois grace au contenu de la commande (codeCPV).
    Dans le cas des Fournitures et des Services (toutes les commandes hors Travaux),
    si la durée est supérieur à 10 ans, alors on impute par la médiane des durées pour le même codeCPV

    Retour
        - pd.DataFrame
    """
    logger.info("Début du tritement: Imputation de la variable dureeMois")
    df_intermediaire = df[["objet", "dureeMois", "dureeMoisEstimee", "dureeMoisCalculee", "CPV_min", "montantCalcule"]]
    # On fait un groupby sur la division des cpv (CPV_min) afin d'obtenir toutes les durees par division
    df_group = pd.DataFrame(df_intermediaire.groupby(["CPV_min"])["dureeMoisCalculee"])
    # On cherche à obtenir la médiane par division de CPV
    df_group.columns = ["CPV_min", "listeDureeMois"]
    df_group["mediane_dureeMois_CPV"] = df_group.listeDureeMois.apply(np.median)
    # La liste des duree exacte est inutile: on supprime
    df_group.drop("listeDureeMois", axis=1, inplace=True)
    # On ajoute provisoirement la colonne mediane_dureeMois
    df = pd.merge(df, df_group, how="left", left_on="CPV_min", right_on="CPV_min", copy=False)
    # 120 = 10 ans. duree arbitraire jugée trop longue.
    # En l'etat, on ne touche pas au duree concernant la catégorie Travaux. identifié par le codeCPV_min == 45.
    mask = ((df.dureeMoisCalculee > 120) & (df.CPV_min != '45'))
    logger.info(f"Au total, {sum(mask)} duree en mois sont encore supérieures à 120 "
                f"(et les marchés ne concernent pas le monde des travaux)")
    df.dureeMoisCalculee = np.where(mask, df.mediane_dureeMois_CPV, df.dureeMoisCalculee)
    # On modifie au passage la colonne dureeMoisEstimee
    df['dureeMoisEstimee'] = np.where(mask, "True", df.dureeMoisEstimee)
    # La mediane n'est pas utile dans le df final: on supprime
    df.drop("mediane_dureeMois_CPV", axis=1, inplace=True)

    logger.info("Fin du traitement")
    return df


def replace_char(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour mettre en qualité le champ objetMarche

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Remplacement des caractères mal converti")
    # Remplacement brutal du caractère (?) par XXXXX
    df['objet'] = df['objet'].str.replace('�', 'XXXXX')
    logger.info("Fin du traitement")
    return df


def create_value_number(x):
    """Retourne le max parmis les élements de l'obejt d'entrée, si il y a un Nan, retourne Nan 
    """
    if x.isna().any():
        value_number = 0  # Essentiel pour la construction de df_avec_bon_id. Sinon ça crash
    else:
        value_number = max(x)
    return value_number

def create_new_index(x):
    return list(x.index)


def regroupement_marche_complet(df):
    # On regroupe selon l objet du marché. Attention, objet pas forcément unique mais idMarche ne l'est pas non plus.
    df_group = pd.DataFrame(df[["objet", "datePublicationDonnees", "montant", "id"]] .groupby(["objet",
                                                    "datePublicationDonnees", "montant"])["id"])
    df_group['new_index'] = df_group[1].apply(create_new_index)
    df_group["nbTit"] = df_group[1].apply(lambda x: len(x)) # On compte le nombre d'id dans chaque ligne df group et ainsi on a le nombre de titulaires
    df_group['bon_id'] = df_group[1].apply(create_value_number) # On récupère l'ID le plus haut parmis les doublons (au sens objet et datePublicationDonnees)
    flat_indx = list(itertools.chain(*df_group['new_index'].values))
    flat_nbtit = list(itertools.chain(*[[x]*x for x in df_group["nbTit"].values])) # On applati la liste, si on a 2 titulaires il faut donc avoir deux 2 à la suite dans la liste d'où le itertools.chain
    flat_id = list(itertools.chain(*[[x]*y for (x,y) in zip(df_group["bon_id"].values, df_group["nbTit"].values)]))
    dict_data = {"nombreTitulaireSurMarchePresume" : flat_nbtit, "id": flat_id}
    df_reconstruct = pd.DataFrame(index=flat_indx, data=dict_data)
    
    df_reconstruct['id'].replace(0, pd.NA, inplace=True) # On renomme remplace ici les 0 par des Nan, pas plus tôt pour deux raisons 
    # Lorsqu'on flat les listes les Nan ne sont pas itérables
    # Cependant on a bien besoin de forcer le typage en pd.NA pour la suite de la pipeline
    df["nombreTitulaireSurMarchePresume"] = False # Je créé la colonne dans df pour que les deux colonnes soit update avec la methode update()
    df.update(df_reconstruct)
    df['nombreTitulaireSurMarchePresume'].replace(0,np.nan, inplace=True) # Ajout artificiel pour se caler sur le format de la fonction regroupement_marche_complet() initiale

    return df


def indice_marche_avec_modification(data: dict) -> list:
    """
    Renvoie la liste des indices des marchés contenant une modification

    Retour:
        - list
    """
    liste_indices = []
    for i in range(len(data['marches'])):
        # Ajout d'un identifiant technique -> Permet d'avoir une colonne id unique par marché
        data["marches"][i]["id_technique"] = i
        if data["marches"][i]["modifications"]:
            liste_indices += [i]
    return liste_indices


def recuperation_colonne_a_modifier(data: dict, liste_indices: list) -> dict:
    """
    Renvoie les noms des differentes colonnes recevant une modification
    sous la forme d'un dictionnaire: {Nom_avec_modification: Nom_sans_modification}

    Retour:
        dict
    """
    liste_colonne = []
    colonne_to_modify = {}
    for indice in liste_indices:
        # colonne_modifiees = list(data["marches"][indice]["modifications"][0].keys())
        for col in data["marches"][indice]["modifications"][0].keys():
            if "Modification" not in col:
                col += "Modification"
            if col not in liste_colonne:
                liste_colonne += [col]
    for col in liste_colonne:
        if "Modification" in col and col != "objetModification":
            name_col = col.replace("Modification", "")
            colonne_to_modify[col] = name_col
        else:
            colonne_to_modify[col] = col
    return colonne_to_modify


def prise_en_compte_modifications(df: pd.DataFrame, col_to_normalize: str = 'modifications'):
    """
    La fonction json_normalize de pandas ne permet pas de spliter la clef modifications automatiquement.
    Cette fonction permet de le faire
    En entrée : La sortie json_normalize de pandas. (avec une colonne modifications)
    Le dataframe en entrée est directement modifié dans la fonction.
    """
    # Check colonne modifications.
    if col_to_normalize not in df.columns:
        raise ValueError(f"Il n'y a aucune colonne du nom de {col_to_normalize} dans le dataframe entrée en paramètre")
    to_normalize = df[col_to_normalize]  # Récupération de la colonne à splitter
    df["booleanModification"] = 0
    for i in range(len(to_normalize)):
        json_modification = to_normalize[i]
        if json_modification:  # dans le cas ou des modifications ont été apportées
            for col in json_modification[0].keys():
                col_init = col
                # Formatage du nom de la colonne
                if "Modification" not in col:
                    col += "Modification"
                if col not in df.columns:  # Cas ou on tombe sur le premier marche qui modifie un champ
                    df[col] = ""  # Initialisation dans le df initial
                df[col][i] = json_modification[0][col_init]
                df["booleanModification"][i] = 1  # Création d'une booléenne pour simplifier le subset pour la suite


def split_dataframe(df: pd.DataFrame, sub_data: pd.DataFrame, modalite: str) -> tuple:
    """
    Définition de deux dataFrame.
        - Le premier qui contiendra uniquement les lignes avec modification, pour le marché ayant pour objet modalite
        - Le second contiendra l'ensemble des lignes correspondant au marché isolé dans le df1 qui ont pour objet
          modalite

        :param df: la source totale des données
        :param sub_data: le sous-ensemble correspondant à l'ensemble des marchés avec une modification
        :param modalite: la modalité sur laquelle on veut filtrer

        Retour:
            tuple (pd.DataFrame, pd.DataFrame)
    """
    # Premier df: Contenant les lignes d'un marche avec des colonnes modifications non vide
    marche = sub_data[sub_data.objet == modalite]
    marche = marche.sort_values(by='id')
    # Second dataframe: Dans le df complet, récupération des lignes correspondant au marché récupéré
    date = marche.datePublicationDonnees.iloc[0]
    # A concaténer ?
    marche_init = df[df.objet == modalite]
    marche_init = marche_init[marche_init.datePublicationDonnees == date]
    return marche, marche_init


def fusion_source_modification(raw: pd.DataFrame, df_source: pd.DataFrame, col_modification: list,
                               dict_modification: dict) -> pd.DataFrame:
    """
    Permet de fusionner les colonnes xxxModification et sa colonne.
    raw correspond à une ligne du df_source
    Modifie le df_source

    Retour:
        pd.DataFrame
    """
    for col in col_modification:
        col_init = dict_modification[col]
        if raw[col] != '':
            df_source[col_init].loc[raw.name] = raw[col]
    return df_source


def regroupement_marche(df: pd.DataFrame, dict_modification: dict) -> pd.DataFrame:
    """
    Permet de recoder la variable identifiant.
    Actuellement: 1 identifiant par déclaration (marché avec ou sans modification)
    Un marché peut être déclaré plusieurs fois en fonction du nombre d'entreprise. Si 2 entreprises sur
    En sortie: id correspondra à un identifiant unique pour toutes les lignes composants un marché SI il a eu une
    modification.
    Modification inplace du df source

    Retour:
        pd.DataFrame
    """
    df["idtech"] = ""
    subdata_modif = df[df.booleanModification == 1]  # Tout les marchés avec les modifications
    liste_objet = list(subdata_modif.objet.unique())
    df_to_concatene = pd.DataFrame()  # df vide pour la concaténation
    logger.info("Au total, {} marchés sont concernés par au moins une modification".format(len(liste_objet)))
    for objet_marche in liste_objet:
        # Récupération du dataframe modification et du dataframe source
        marche, marche_init = split_dataframe(df, subdata_modif, objet_marche)
        for j in range(len(marche)):
            marche_init = fusion_source_modification(marche.iloc[j], marche_init, dict_modification.keys(),
                                                     dict_modification)
        marche_init["idtech"] = marche.iloc[-1].id_technique
        df_to_concatene = pd.concat([df_to_concatene, marche_init], copy=False)
    df.update(df_to_concatene)
    # Attention aux id.
    df["idMarche"] = np.where(df.idtech != "", df.idtech, df.id_technique)
    return df


def manage_modifications(data: dict) -> pd.DataFrame:
    """
    Conversion du json en pandas et incorporation des modifications

    Retour:
        pd.DataFrame
    """
    l_indice = indice_marche_avec_modification(data)
    dict_modification = recuperation_colonne_a_modifier(data, l_indice)
    df = json_normalize(data['marches'])
    df = df.astype(conf_glob["nettoyage"]['type_col_nettoyage'], copy=False)
    prise_en_compte_modifications(df)
    df["idtech"] = df["id_technique"].copy()
    df['idMarche'] = df["id_technique"].copy()
    #df = regroupement_marche(df, dict_modification)
    return df


if __name__ == "__main__":
    if conf_debug["debug"]:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        with open('df_nettoye', 'rb') as df_nettoye:
            df = pickle.load(df_nettoye)
            init_len = len(df)
        with open("profilingSnettoyage_opti_size{}.txt".format(init_len), "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats('ncalls')
            ps.sort_stats('cumulative')
            ps.print_stats()
    else:
        main()
