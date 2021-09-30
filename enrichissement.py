import csv
import json
import os
import pickle
import logging
import numpy as np
import pandas as pd
from geopy.distance import distance, Point

logger = logging.getLogger("main.enrichissement")
logger.setLevel(logging.DEBUG)


with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

path_to_data = conf_data["path_to_data"]


def main():
    with open("df_nettoye", "rb") as df_nettoye:
        df = pickle.load(df_nettoye)

    df = df.astype(conf_glob["enrichissement"]["type_col_enrichissement"], copy=False)
    df = (df.pipe(enrichissement_siret)
          .pipe(enrichissement_cpv)
          .pipe(reorganisation)
          .pipe(enrichissement_geo)
          .pipe(apply_luhn)
          .pipe(manage_column_final)
          )

    logger.info("Début du traitement: Ecriture du csv final: decp_augmente")
    df.to_csv("enrichissement_arrondissementenrichissement_arrondissement.csv", quoting=csv.QUOTE_NONNUMERIC, sep=";")
    logger.info("Fin du traitement")


def enrichissement_siret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement du dataFrame avec la base réferentiel entreprise pour les Acheteur ET les Etablissement

    Return:
        - pd.DataFrame
    """
    # Transformation de la colonne idTitulaire en colonne siret
    df_ = df.copy()
    dictCorrespondance = conf_glob["enrichissement"]["abrev2nom"]
    # Travail sur l'id des titulaires
    df_["idTitulaires"] = np.where(~df_["idTitulaires"].str.isdigit(), "", df_.idTitulaires)
    df_ = df_.rename(columns={"idTitulaires": "siretEtablissement"})
    # Travail sur l'id des acheteurs
    df_ = df_.rename(columns={"acheteur.id": "idAcheteur"})
    path = os.path.join(path_to_data, conf_data["geo_comp"])
    columns = [
        "siret",
        "siren",
        "codePostalEtablissement",
        "libelleCommuneEtablissement",
        "codeCommuneEtablissement",
        "activitePrincipaleEtablissement",
        "numeroVoieEtablissement",
        "typeVoieEtablissement",
        "libelleVoieEtablissement",
        "nic",
        "categorieEntreprise",
        "nicSiegeUniteLegale"]  # Colonne à utiliser dans la base Siren

    result = pd.DataFrame(columns=columns)
    chunksize = 1000000
    for gm_chunk in pd.read_csv(path, chunksize=chunksize, sep=",", encoding="utf-8", usecols=columns, dtype=str):
        resultTemp = pd.merge(df_, gm_chunk, left_on="siretEtablissement", right_on="siret", copy=False)
        # rename les colonnes de gm_chunk
        gm_chunk = gm_chunk.rename(columns={"codePostalEtablissement": "codePostalAcheteur",
                                            "libelleCommuneEtablissement": "libelleCommuneAcheteur",
                                            "codeCommuneEtablissement": "codeCommuneAcheteur"})
        gm_chunk = gm_chunk.drop(columns=["activitePrincipaleEtablissement",
                                          "numeroVoieEtablissement",
                                          "typeVoieEtablissement",
                                          "libelleVoieEtablissement",
                                          "nic",
                                          "nicSiegeUniteLegale",
                                          "categorieEntreprise",
                                          "siren"])
        resultTemp = pd.merge(resultTemp, gm_chunk, left_on="idAcheteur", right_on="siret", copy=False)
        # jointure entre resltTemp et gm_chunk
        result = pd.concat([result, resultTemp], axis=0, copy=False)
        del resultTemp
    # Travail sur l'adresseEtablissement
    result["typeVoieEtablissement"] = result["typeVoieEtablissement"].apply(lambda x: dictCorrespondance[x] if x in dictCorrespondance.keys() else x)
    result["adresseEtablissement"] = result[["numeroVoieEtablissement", "typeVoieEtablissement", "libelleVoieEtablissement"]].astype(str).agg(" ".join, axis=1)
    result = result.drop(columns=["siret", "numeroVoieEtablissement", "typeVoieEtablissement", "libelleVoieEtablissement", "siret_y"])
    # Remplacement des null par NC dans la CatégorieEntreprise
    result["categorieEntreprise"] = np.where(result["categorieEntreprise"].isnull(), "NC", result["categorieEntreprise"])
    return result


def manage_column_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renommage de certaines colonnes et trie des colonnes.

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Reorganisation du dataframe final")
    with open(os.path.join("confs", "var_to_export.json")) as f:
        conf_export = json.load(f)
    colonne_to_export = []

    for key in conf_export["export"].keys():
        if conf_export["export"][key] == 1:
            colonne_to_export += [key]
    df = df.reindex(columns=colonne_to_export)
    df = df.rename(columns={
        "natureObjet": "natureObjetMarche",
        "categorieEntreprise": "categorieEtablissement",
    })
    return df


# Algorithme de Luhn

def is_luhn_valid(x: int) -> bool:
    """
    Application de la formule de Luhn à un nombre
    Permet la verification du numero SIREN et Siret d'un acheteur/etablissement

    Retour:
        - bool
    """
    try:
        luhn_corr = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
        list_number_in_x = [int(i) for i in list(str(x))]
        l2 = [luhn_corr[i] if (index + 1) % 2 == 0 else i for index, i in enumerate(list_number_in_x[::-1])]
        if sum(l2) % 10 == 0:
            return True
        elif str(x)[:9] == "356000000":  # SIREN de la Poste
            if sum(list_number_in_x) % 5 == 0:
                return True
        return False
    except:
        return False


def apply_luhn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Application de la formule de Luhn sur les siren/siret

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Vérification Siren/Siret par formule de Luhn")
    # Application sur les siren des Acheteur
    df['siren1Acheteur'] = df["idAcheteur"].str[:9]
    df_SA = pd.DataFrame(df['siren1Acheteur'])
    df_SA = df_SA.drop_duplicates(subset=['siren1Acheteur'], keep='first')
    df_SA['sirenAcheteurValide'] = df_SA['siren1Acheteur'].apply(is_luhn_valid)
    df = pd.merge(df, df_SA, how='left', on='siren1Acheteur', copy=False)
    logger.info("Nombre de Siren Acheteur jugé invalide:{}".format(len(df) - sum(df.sirenAcheteurValide)))
    del df['siren1Acheteur']
    del df_SA
    # Application sur les siren des établissements
    df['siren2Etablissement'] = df.sirenEtablissement.str[:]
    df_SE = pd.DataFrame(df['siren2Etablissement'])
    df_SE = df_SE.drop_duplicates(subset=['siren2Etablissement'], keep='first')
    df_SE['sirenEtablissementValide'] = df_SE['siren2Etablissement'].apply(is_luhn_valid)
    df = pd.merge(df, df_SE, how='left', on='siren2Etablissement', copy=False)
    logger.info("Nombre de Siren Etablissement jugé invalide:{}".format(len(df) - sum(df.sirenEtablissementValide)))
    del df['siren2Etablissement']
    del df_SE
    # Application sur les siret des établissements
    df['siret2Etablissement'] = df.siretEtablissement.str[:]
    df_SE2 = pd.DataFrame(df['siret2Etablissement'])
    df_SE2 = df_SE2.drop_duplicates(subset=['siret2Etablissement'], keep='first')
    df_SE2['siretEtablissementValide'] = df_SE2['siret2Etablissement'].apply(is_luhn_valid)
    # Merge avec le df principal
    df = pd.merge(df, df_SE2, how='left', on='siret2Etablissement', copy=False)
    logger.info("Nombre de Siret Etablissement jugé invalide:{}".format(len(df) - sum(df.siretEtablissementValide)))
    del df["siret2Etablissement"]
    del df_SE2
    # On rectifie pour les codes non-siret
    df.siretEtablissementValide = np.where(
        (df.typeIdentifiantEtablissement != 'SIRET'),
        "Non valable",
        df.siretEtablissementValide)
    return df


def enrichissement_cpv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Récupération des codes CPV formatés.

    Return:
        - pd.Dataframe
    """
    # Importation et mise en forme des codes/ref CPV
    logger.info("Début du traitement: Enrichissement cpv")
    path = os.path.join(path_to_data, conf_data["cpv_2008_ver_2013"])
    refCPV = pd.read_excel(path, usecols=['CODE', 'FR'])
    refCPV.columns = ['CODE', 'refCodeCPV']
    refCPV_min = pd.DataFrame.copy(refCPV, deep=True)
    refCPV_min["CODE"] = refCPV_min.CODE.str[0:8]
    refCPV_min = refCPV_min.drop_duplicates(subset=['CODE'], keep='first')
    refCPV_min.columns = ['CODEmin', 'FR2']
    # Merge avec le df principal
    df = pd.merge(df, refCPV, how='left', left_on="codeCPV", right_on="CODE", copy=False)
    # del refCPV
    df = pd.merge(df, refCPV_min, how='left', left_on="codeCPV", right_on="CODEmin", copy=False)
    del refCPV_min
    # Garde uniquement la colonne utile / qui regroupe les nouvelles infos
    df.refCodeCPV = np.where(df.refCodeCPV.isnull(), df.FR2, df.refCodeCPV)
    df.drop(columns=["FR2", "CODE", "CODEmin"], inplace=True)
    df = pd.merge(df, refCPV, how='left', left_on="refCodeCPV", right_on="refCodeCPV", copy=False)
    del refCPV
    # Rename la variable CODE en codeCPV
    df.rename(columns={"codeCPV": "codeCPV_Original",
              "CODE": "codeCPV"}, inplace=True)
    return df


def reorganisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mise en qualité du dataframe

    Return:
        - pd.Dataframe
    """
    logger.info("Début du traitement: Reorganisation du dataframe")
    # Ajustement de certaines colonnes
    df.codePostalEtablissement = df.codePostalEtablissement.astype(str).str[:5]
    df.codePostalAcheteur = df.codePostalAcheteur.astype(str).str[:5]
    df.codeCommuneEtablissement = df.codeCommuneEtablissement.astype(str).str[:5]
    df.codeCommuneAcheteur = df.codeCommuneAcheteur.astype(str).str[:5]

    df.anneeNotification = df.anneeNotification.astype(str)
    df.codeDepartementExecution = df.codeDepartementExecution.astype(str)

    # codePostal est enlevé pour le moment car est un code départemental
    df.drop(columns=["uid", "uuid", "denominationSociale_x", 'siret'], inplace=True, errors="ignore")

    # Réorganisation des colonnes et de leur nom
    column_mapping = {
        'id': "id",
        '_type': "type",
        'objet': "objetMarche",
        'lieuExecution.code': "lieuExecutionCode",
        'lieuExecution.typeCode': "lieuExecutionTypeCode",
        'lieuExecution.nom': "lieuExecutionNom",
        'acheteur.id': "idAcheteur",
        'acheteur.nom': "nomAcheteur",
        'typeIdentifiant': "typeIdentifiantEtablissement",
        'idTitulaires': "siretEtablissement",
        'denominationSociale_y': "denominationSocialeEtablissement",
        'nic': "nicEtablissement",
        'CPV_min': "codeCPV_division",
        'siren': "sirenEtablissement",
        'refCodeCPV': "referenceCPV"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Rectification codePostalAcheteur et codeCommuneAcheteur
    df["codePostalAcheteur"] = df["codePostalAcheteur"].apply(fix_codegeo)
    df["codeCommuneAcheteur"] = df["codeCommuneAcheteur"].apply(fix_codegeo)
    df["codePostalEtablissement"] = df["codePostalEtablissement"].apply(fix_codegeo)
    df["codeCommuneEtablissement"] = df["codeCommuneEtablissement"].apply(fix_codegeo)
    # Petites corrections sur lieuExecutionTypeCode et nature
    list_to_correct = ["lieuExecutionTypeCode", "nature"]
    for column in list_to_correct:
        df[column] = df[column].str.upper()
        df[column] = df[column].str.replace("É", "E")
    return df


def fix_codegeo(code: str) -> str:
    """
    Correction de l'erreur ou le code 01244 soit considérer comme l'entier 1244
    code doit etre un code commune/postal

    Return:
        - str
    """
    if len(code) == 4:
        code = "0" + code
    if "." in code[:5]:
        return "0" + code[:4]
    return code[:5]


def enrichissement_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement du dataFrame avec la base réferentiel géographique pour les Acheteur ET les Etablissement

    Return:
        - pd.DataFrame
    """
    path = os.path.join(path_to_data, conf_data["geo_ref"])
    columns = ["SUPERFICIE", "POPULATION", "coordonnees_commune", "code_commune", "code_departement", "nom_departement", 'code_region', 'nom_region', "code_arrondissement", "nom_arrondissement"]
    geo_ref = pd.read_csv(path, sep=";", usecols=columns)
    # jointure sur communeEtablissement
    df = pd.merge(df, geo_ref, how='left', left_on="codeCommuneEtablissement", right_on="code_commune", copy=False)
    df = df.rename(columns={"SUPERFICIE": "superficieCommuneEtablissement",
                            "POPULATION": "populationCommuneEtablissement",
                            "coordonnees_commune": "geolocCommuneEtablissement",
                            "code_departement": "codeDepartementEtablissement",
                            "nom_departement": "libelleDepartementEtablissement",
                            'code_region': "codeRegionEtablissement",
                            'nom_region': "libelleRegionEtablissement",
                            "code_arrondissement": "codeArrondissementEtablissement",
                            "nom_arrondissement": "libelleArrondissementEtablissement"})
    df.drop(columns="code_commune", inplace=True)
    # jointure sur communeAcheteur
    df = pd.merge(df, geo_ref, how='left', left_on="codeCommuneAcheteur", right_on="code_commune", copy=False)
    df = df.rename(columns={"SUPERFICIE": "superficieCommuneAcheteur",
                            "POPULATION": "populationCommuneAcheteur",
                            "coordonnees_commune": "geolocCommuneAcheteur",
                            "code_departement": "codeDepartementAcheteur",
                            "nom_departement": "libelleDepartementAcheteur",
                            'code_region': "codeRegionAcheteur",
                            'nom_region': "libelleRegionAcheteur",
                            "code_arrondissement": "codeArrondissementAcheteur",
                            "nom_arrondissement": "libelleArrondissementAcheteur"})
    df.drop(columns="code_commune", inplace=True)
    return df


def get_distance(row: pd.DataFrame) -> float:
    """
    Calcul des distances entre l'acheteur et l'établissement qui répond à l'offre

    Return:
        - float
    """
    try:
        x = Point(row.longitudeCommuneAcheteur, row.latitudeCommuneAcheteur)
        y = Point(row.longitudeCommuneEtablissement, row.latitudeCommuneEtablissement)

        return distance(x, y).km
    except ValueError:
        return None


if __name__ == "__main__":
    main()
