import csv
import json
import os
import pickle
import logging
import numpy as np
import pandas as pd
import cProfile
import pstats
from geopy.distance import distance, Point

logger = logging.getLogger("main.enrichissement")
logger.setLevel(logging.DEBUG)


with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

with open(os.path.join("confs", "var_debug.json")) as f:
    conf_debug = json.load(f)["enrichissement"]

path_to_data = conf_data["path_to_data"]
path_to_cache = conf_data["path_to_cache"]


def main():
    with open('df_nettoye', 'rb') as df_nettoye:
        df = pickle.load(df_nettoye)

    
    df = df.astype(conf_glob["enrichissement"]["type_col_enrichissement"], copy=False)
    df = (df.pipe(cache_management_insee)
          .pipe(enrichissement_siret)
          .pipe(enrichissement_cpv)
          .pipe(enrichissement_acheteur)
          .pipe(reorganisation)
          .pipe(enrichissement_geo)
          .pipe(enrichissement_type_entreprise)
          .pipe(apply_luhn)
          .pipe(enrichissement_departement)
          .pipe(enrichissement_arrondissement)
          .pipe(concat_unduplicate_and_caching_hash)
          .pipe(manage_column_final)
          .pipe(change_sources_name)
          )

    logger.info("Début du traitement: Ecriture du csv final: decp_augmente")
    df.to_csv("decp_augmente.csv", quoting=csv.QUOTE_NONNUMERIC, sep=";")
    # Mise en cache pour être ré_utilisé.
    if conf_debug["debug"]:
        with open('df_new_augmente', 'wb') as df_augmente:
            # Export présent pour faciliter la comparaison
            pickle.dump(df, df_augmente)
    logger.info("Fin du traitement")

def concat_unduplicate_and_caching_hash(df):
    """
    Cette fonction concatène ensemble les dataframes (celui du flux et celui du Stock).
    Dédoublonne le dataframe (lorsqu'on aura les infos du dédoublonnage)
    Pourquoi ne pas concaténer uniquement à la fin du processus enrichissement ?
    Dans la fonction suivante, manage_column_final on garde les colonnes souhaitées et on drop les autres.
    Si un jour on choisit d'exporter une nouvelle colonne il est intéressant d'avoir en cache le dataframe entier.
    Sinon on doit tout recalculer.
    """
    print("concat cache", df.shape)
    # concat
    path_to_df_cache = os.path.join(path_to_data, conf_data['cache_df'])
    file_cache_exists = os.path.isfile(path_to_df_cache)
    if file_cache_exists :
        with open(os.path.join(path_to_data, conf_data['cache_df']), "rb") as file_cache:
            df_cache = pickle.load(file_cache)
        df = pd.concat([df, df_cache]).reset_index(drop=True)

    # Save DataFrame pour la prochaine fois
    with open(os.path.join(path_to_data, conf_data['cache_df']), "wb") as file_cache:
            pickle.dump(df, file_cache)
    return df

def manage_column_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renommage de certaines colonnes et trie des colonnes.

    Retour:
        - pd.DataFrame
    """
    print('manage column fina', df.shape)
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


def extraction_departement_from_code_postal(code_postal: str) -> str:
    """
    Renvoie le code postal en prenant en compte les territoires outre-mer
    code_postal est un str

    Retour:
        str
    """
    try:
        code = code_postal[:2]
        if code == "97" or code == "98":
            code = code_postal[:3]
        return code
    except IndexError:
        return "00"


def jointure_base_departement_region() -> pd.DataFrame:
    """
    Permet la jointure entre la base departement de l'Insee (dossier data) et la base region de l'Insee

    Retour:
        - pd.DataFrame
    """
    # Import de la base département
    path_dep = os.path.join(path_to_data, conf_data["departements-francais"])
    departement = pd.read_csv(path_dep, sep=",", usecols=['dep', 'reg', 'libelle'], dtype={"dep": str, "reg": str, "libelle": str})
    # Import de la base Région
    path_reg = os.path.join(path_to_data, conf_data["region-fr"])
    region = pd.read_csv(path_reg, sep=",", usecols=["reg", "libelle"], dtype={"reg": str, "libelle": str})
    region.columns = ["reg", "libelle_reg"]
    # Merge des deux bases
    df_dep_reg = pd.merge(departement, region, how="left", left_on="reg", right_on="reg", copy=False)
    df_dep_reg.columns = ["code_departement", "code_region", "Nom", "Region"]
    df_dep_reg.code_region = np.where(df_dep_reg.code_region.isin(["1", "2", "3", "4", "6"]), "0" + df_dep_reg.code_region, df_dep_reg.code_region)
    return df_dep_reg


def enrichissement_departement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout des variables région et departement dans decp. Ces deux variables concernent les acheteurs et les établissements

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Ajout des libelle departement/Region pour les acheteurs et les etabissements")
    logger.info("Début de la jointure entre les deux csv Insee: Departements et Regions")
    df_dep_reg = jointure_base_departement_region()
    logger.info("Fin de la jointure")
    # Creation de deux variables récupérant le numéro du departement
    df["departementAcheteur"] = df["codePostalAcheteur"].apply(extraction_departement_from_code_postal)
    df["departementEtablissement"] = df["codePostalEtablissement"].apply(extraction_departement_from_code_postal)
    # Fusion entre Numero et numero de departement pour recuperer le nom et ou la region (pour etablissement)
    df_dep_reg.code_departement = df_dep_reg.code_departement.astype(str)
    df = pd.merge(df, df_dep_reg, how="left", left_on="departementAcheteur", right_on="code_departement", copy=False)
    df = df.rename(columns={
                   'Nom': "libelleDepartementAcheteur",
                   'Region': "libelleRegionAcheteur",
                   'code_region': "codeRegionAcheteur"
                   })
    df = df.drop(["code_departement"], axis=1)
    df = pd.merge(df, df_dep_reg, how="left", left_on="departementEtablissement", right_on="code_departement", copy=False)
    df = df.rename(columns={
                   'Nom': "libelleDepartementEtablissement",
                   'Region': "libelleRegionEtablissement",
                   'code_region': "codeRegionEtablissement"
                   })
    df = df.drop(["code_departement"], axis=1)
    return df


def enrichissement_arrondissement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout du code Arrondissement à partir du code commune et du libelle du Arrondissement à partir de son code.
    L'Arrondissement correspond à la zone géographique raccrochée à une sous-prefecture

    Retour:
        - pd.DataFrame
    """
    logger.info("Début du traitement: Ajout des codes/libelles des arrondissements pour les acheteurs et les etablissements")
    df = get_code_arrondissement(df)
    df = get_libelle_arrondissement(df)
    return df


def get_code_arrondissement(df):
    """
    Ajout de la colonne code Arrondissement à partir du code commune

    Retour:
        - pd.DataFrame
    """
    path_to_commune = os.path.join(path_to_data, conf_data["commune-fr"])
    commune = pd.read_csv(path_to_commune, sep=",", usecols=['TYPECOM', 'COM', 'ARR'], dtype={"COM": str, "ARR": str})
    commune = commune[commune.TYPECOM == "COM"]
    commune.drop(['TYPECOM'], axis=1)
    df = df.merge(commune, how="left", left_on="codeCommuneAcheteur", right_on="COM", copy=False)
    df = df.drop(["COM"], axis=1)
    df = df.rename(columns={"ARR": "codeArrondissementAcheteur"})
    df = pd.merge(df, commune, how="left", left_on="codeCommuneEtablissement", right_on="COM", copy=False)
    df = df.drop(["COM"], axis=1)
    df = df.rename(columns={"ARR": "codeArrondissementEtablissement"})
    return df


def get_libelle_arrondissement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout de la colonne libelle Arrondissement à partir du code Arrondissement

    Retour:
        - pd.DataFrame"""
    path_to_arrondissement = os.path.join(path_to_data, conf_data["arrondissement-fr"])
    arrondissement = pd.read_csv(path_to_arrondissement, sep=",", usecols=['ARR', 'LIBELLE'], dtype={"ARR": str, "LIBELLE": str})
    df = pd.merge(df, arrondissement, how="left", left_on="codeArrondissementAcheteur", right_on="ARR", copy=False)
    df = df.drop(["ARR"], axis=1)
    df = df.rename(columns={"LIBELLE": "libelleArrondissementAcheteur"})
    df = pd.merge(df, arrondissement, how="left", left_on="codeArrondissementEtablissement", right_on="ARR", copy=False)
    df = df.drop(["ARR"], axis=1)
    df = df.rename(columns={"LIBELLE": "libelleArrondissementEtablissement"})
    return df


def actualiser_cache_entreprise(df_to_analyse: pd.DataFrame, path_to_db: str, dfcache: pd.DataFrame, columns: list, dtypes: list, chunksize=1000000):
    """
    La fonction permet d'actualiser le cache pour enrichissement_entreprise qui utilise la base StockUniteLegale.

    Arguments
    ----------
    df_to_analyse (Series) avec une unique colonne

    Returns
    --------------
    dfcache (DataFrame) correspondant au cache
    df_to_analys correspondant aux lignes qui n'ont pas matchés avec la base csv
    
    """
    for gm_chunk in pd.read_csv(path_to_db, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns, dtype=dtypes):
        
        #Spécificité de cette fonction, on doit pre process la donnée du chunk pour la comparer
        gm_chunk["nicSiegeUniteLegale"] = gm_chunk["nicSiegeUniteLegale"].astype(str).str.zfill(5)
        gm_chunk["siren"] = gm_chunk["siren"].astype(str).str\
            .cat(gm_chunk["nicSiegeUniteLegale"].astype(str), sep='')
        gm_chunk.rename(columns={"siren":"siretEtablissement"}, inplace= True) # Si on ne renomme pas ici ça va perturber la suite du process en créant une autre colonne siretEtablissement par la suite
        # Ajouter à df cache les infos qu'il faut
        matching = gm_chunk.loc[gm_chunk.siretEtablissement.isin(df_to_analyse.tolist())].copy() # La copie du dataframe qui match parmis le chunk en cours
        df_to_analyse = df_to_analyse[~df_to_analyse.isin(matching.siretEtablissement.tolist())] 
        dfcache = dfcache.append(matching)

    return dfcache, df_to_analyse

def enrichissement_type_entreprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement des données avec la catégorie de l'entreprise. Utilisation de la base StockUniteLegale de l'Insee

    Retour:
        - pd.DataFrame
    """
    print("type entreprise", df.shape)
    logger.info('début enrichissement_type_entreprise')
    df = df.astype(conf_glob["enrichissement"]["type_col_enrichissement_siret"], copy=False)
    # Recuperation de la base
    path = os.path.join(path_to_data, conf_data["base_ajout_type_entreprise"])
    # La base est volumineuse. Pour "optimiser la mémoire", on va segmenter l'import
    usecols=["siren", "categorieEntreprise", "nicSiegeUniteLegale"]
    dtype={"siren": 'string', "categorieEntreprise": 'string', "nicSiegeUniteLegale": 'string'}
    path_cache = os.path.join(path_to_cache, conf_data["cache_bdd_legale"])
    path_cache_not_in_bdd = os.path.join(path_to_cache, conf_data["cache_not_in_bdd_legale"])
    cache_siren_not_found_exists = os.path.isfile(path_cache_not_in_bdd)
    if cache_siren_not_found_exists:
        list_siret_not_found = loading_cache(path_cache_not_in_bdd)
    else:
        list_siret_not_found = []
    mask_siren_valid_not_found = df.siretEtablissement.isin(list_siret_not_found)
    dfSIRET_valide_notfound = df[mask_siren_valid_not_found]
    # On retire les siret valides mais non trouvés lors des précédents passages du df.
    df = df[~mask_siren_valid_not_found]
    cache_exists = os.path.isfile(path_cache)
    if cache_exists:
        dfcache = loading_cache(path_cache)
        dfcache = dfcache.rename(columns={"siren" : "siretEtablissement"})
        series_siren_not_in_cache, series_siren_in_cache = split_on_column_match(df, dfcache, column="siretEtablissement")        
        need_refresh_cache = not(series_siren_not_in_cache.empty)
        if need_refresh_cache:
            logger.info("Enrichissement type entreprise: Besoin d'actualiser cache")
            dfcache, series_siren_valid_but_not_found_in_bdd = actualiser_cache_entreprise(series_siren_not_in_cache, path, dfcache, columns=usecols, dtypes=dtype)
            dfcache = dfcache.rename(columns={"siren" : "siretEtablissement"})
            dfcache = dfcache.drop_duplicates(subset=['siretEtablissement'], keep='first')
            #Update cache de la lsite des sirets valides mais non trouvés
            list_siret_not_found += series_siren_valid_but_not_found_in_bdd.tolist()

            # Actualise les caches
            write_cache(dfcache, path_cache)
            write_cache(list_siret_not_found, path_cache_not_in_bdd)
    else:
        # crécupérer le dataframe correspondant au cache
        dfcache, series_siren_valid_but_not_found_in_bdd = actualiser_cache_entreprise(df.siretEtablissement, path, dfcache=pd.DataFrame(), columns=usecols, dtypes=dtype)
        dfcache = dfcache.drop_duplicates(subset=['siretEtablissement'], keep='first') # La colonne s'appelle encore siren dans le cache
        # Créer les cache
        write_cache(dfcache, path_cache)
        write_cache(series_siren_valid_but_not_found_in_bdd.tolist(), path_cache_not_in_bdd)
    dfcache.rename(columns={"siren": "siretEtablissement"}, inplace=True)
    # # Jointure sur le Siret entre df et to_add
    df = df.merge(
        dfcache[['categorieEntreprise', 'siretEtablissement']], how='left', on='siretEtablissement', copy=False)
    df["categorieEntreprise"] = np.where(df["categorieEntreprise"].isnull(), "NC", df["categorieEntreprise"])
    df = pd.concat([df,dfSIRET_valide_notfound])
    logger.info('fin enrichissement_type_entreprise\n')
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
    #del df_SE2enrichissement_siret
    del df_SE2

    # On rectifie pour les codes non-siret
    df.siretEtablissementValide = np.where(
        (df.typeIdentifiantEtablissement != 'SIRET'),
        "Non valable",
        df.siretEtablissementValide)
    return df


def enrichissement_siret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement des données via les codes siret/siren

    Retour:
        - pd.DataFrame
    """
    print('enr siret', df.shape)
    logger.info("Début du traitement: Enrichissement siret")
    dfSIRET = get_siretdf_from_original_data(df)
    archiveErrorSIRET = getArchiveErrorSIRET()

    logger.info("Enrichissement insee en cours...")
    path_to_bdd_insee = os.path.join(path_to_data, conf_data["base_sirene_insee"])
    path_to_cache_insee = os.path.join(path_to_cache, conf_data["cache_bdd_insee"])
    path_to_cache_not_in_insee = os.path.join(path_to_cache, conf_data["cache_not_in_bdd_insee"])
    enrichissementInsee, nanSiren = get_enrichissement_insee(dfSIRET, path_to_bdd_insee, path_to_cache_insee, path_to_cache_not_in_insee)
    logger.info("Enrichissement insee fini")

    logger.info("Enrichissement infogreffe en cours...")
    enrichissementScrap = get_enrichissement_scrap(nanSiren, archiveErrorSIRET)
    del archiveErrorSIRET
    logger.info("enrichissement infogreffe fini")

    logger.info("Concaténation des dataframes d'enrichissement...")
    dfenrichissement = get_df_enrichissement(enrichissementScrap, enrichissementInsee)
    del enrichissementScrap
    del enrichissementInsee
    logger.info("Fini")

    # Ajout au df principal
    df = pd.merge(df, dfenrichissement, how='outer', left_on="idTitulaires", right_on="siret", copy=False)
    del dfenrichissement
    return df


def get_siretdf_from_original_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utilisation d'un dataframe intermediaire pour traiter les Siret unique

    Retour:
        - pd.DataFrame
    """

    dfSIRET = pd.DataFrame.copy(df[['idTitulaires', 'typeIdentifiant', 'denominationSociale']])
    dfSIRET = dfSIRET.drop_duplicates(subset=['idTitulaires'], keep='first')
    dfSIRET.reset_index(inplace=True, drop=True)
    dfSIRET.idTitulaires = dfSIRET.idTitulaires.astype(str)

    dfSIRET["idTitulaires"] = np.where(~dfSIRET["idTitulaires"].str.isdigit(), '00000000000000', dfSIRET.idTitulaires)

    dfSIRET.reset_index(inplace=True, drop=True)

    dfSIRET.rename(columns={
        "idTitulaires": "siret",
        "typeIdentifiant": "siren"}, inplace=True)
    dfSIRET.siren = dfSIRET.siret.str[:9] # 9 = taille du Siren
    dfSIRET.denominationSociale = dfSIRET.denominationSociale.astype(str)

    return dfSIRET


def getArchiveErrorSIRET() -> pd.DataFrame:
    """
    Récupération des siret erronés

    Retour:
        - pd.DataFrame
    """
    archiveErrorSIRET = pd.DataFrame(columns=['siret', 'siren', 'denominationSociale'])
    logger.info('Aucune archive d\'erreur')
    return archiveErrorSIRET


def split_on_column_match(dfSIRET: pd.DataFrame, dfcache: pd.DataFrame, column: str):
    """
    La fonction retourne deux series, la premier correspond aux éléments de dfSIRET n'étant pas dans dfcache (selon la colonne en entrée).
    Le second correspond aux éléments étant dans le cache.
    

    Arguments
    -------------
    dfSIRET : dataframe a analyser
    dfcache : dataframe correspondant au cache
    column : nom de la colonne sur laquelle on split en deux dfSIRET
    
    Returns
    -----------
    Deux series.
    """
    mask_boolean_on_column = dfSIRET.loc[:, str(column)].isin(dfcache.loc[:, str(column)].tolist())
    return dfSIRET.loc[~mask_boolean_on_column, str(column)].copy(), dfSIRET.loc[mask_boolean_on_column, str(column)].copy()


def loading_cache(path_to_cache):
    with open(path_to_cache, 'rb') as df_cache:
        df = pickle.load(df_cache)
    return df

def actualiser_cache(dfSiret_to_add, path_to_db, dfcache, columns, dtypes, chunksize=1000000):
    """
    La fonction parcourt la bdd Insee par chunk. Pour chaque chunk on regarde si il y a des correspondances de siret entre dfSiret_to_add et la bdd insee.
    Si il y a un match, on ajoute alors les lignes de la bdd insee au dataframe cache. Sinon c'est que les siret sont à la fois valide, mais non présent dans le cache.
    On les sépare (on retire ceux trouvé de dfSIRET_to_add) pour pouvoir les mettre dans un second cache.
    
    
    Arguments
    ----------
    dfSiret_to_add (Series) avec une unique colonne
    path_to_db (string)
    dfcache (DataFrame) correspondant au cache.
    columns (list) :  Liste des colonnes du CSV qu'on va lire
    dtypes (dict) :  dictionnaire des types utilisés pour le CSV

    Returns
    --------------
    dfcache (DataFrame) : le nouveau cache actualisé
    dfSiret_to_add (DataFrame) : correspond aux sirets non trouvé dans la BdD
    """
    for gm_chunk in pd.read_csv(path_to_db, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns, dtype=dtypes):
        # Ajouter à df cache les infos qu'il faut
        matching = gm_chunk.loc[gm_chunk.siret.isin(dfSiret_to_add.tolist())].copy() # La copie du dataframe qui match parmis le chunk en cours
        dfSiret_to_add = dfSiret_to_add[~dfSiret_to_add.isin(matching.siret.tolist())] 
        dfcache = dfcache.append(matching)
    return dfcache, dfSiret_to_add

def write_cache(dfcache, path_to_cache):
    with open(path_to_cache, 'wb') as pathcache:
        pickle.dump(dfcache, pathcache)
    logger.info("Cache sauvegardé")
    return None

def cache_management_insee(df, key_columns_df=["idTitulaires", "acheteur.id"], key_columns_csv="siret"):
    """
    La fonction met en cache les lignes en provenance de la bdd insee (stockEtablissement)
    que 'lon utilise pour enrichir les données dans enrichissement_insee et encrihcissement_acheteurs

    Il y a deux fonctions qui pointent vers la bdd insee (StockEtablissement).
    Les deux se basent sur le mot "siret", or à chaque fois deux colonnes différentent sont renommés en siret pour être join.
    La première version de cache qui construisait le cache sur un siret correspondant à une des deux colonnes n'était donc pas satisfaisant.

    On va alors créer un cache complet en amont


    Arguments
    df : DataFrame à analyser
    key_columns_df, les colonnes qu'on va considérer comme clefs du df
    key_columns_csv, la colonnes qu'on va considérer comme clef du csv

    """
    print("cache manageent insee", df.shape)
    # Création des variables ici plutôt que de les mettre dans l'appelle de pipe au début du fichier avec les noms à rallonger

    path_to_bdd_insee = os.path.join(path_to_data, conf_data["base_sirene_insee"])
    path_to_cache_insee = os.path.join(path_to_cache, conf_data["cache_bdd_insee"])
    path_to_cache_not_in_insee = os.path.join(path_to_cache, conf_data["cache_not_in_bdd_insee"])
    columns = [
    'siren',
    'nic',
    'siret',
    'typeVoieEtablissement',
    'libelleVoieEtablissement',
    'codePostalEtablissement',
    'libelleCommuneEtablissement',
    'codeCommuneEtablissement',
    'activitePrincipaleEtablissement',
    'nomenclatureActivitePrincipaleEtablissement']  # Colonne à utiliser dans la base Siren
    
    dtypes = {
        'siret': 'string',
        'typeVoieEtablissement': 'string',
        'libelleVoieEtablissement': 'string',
        'codePostalEtablissement': 'string',
        'libelleCommuneEtablissement': 'string',
        'codeCommuneEtablissement': 'object',
    }

    df_keys = pd.concat([df.loc[:, "idTitulaires"], df.loc[:, "acheteur.id"]])
    df_keys = df_keys.to_frame().rename(columns={0: "siret"})
    mask_siret_not_valid = (~df_keys.siret.apply(is_luhn_valid)) | (df_keys.siret == '00000000000000')
    dfSIRET_siret_not_valid = df_keys[mask_siret_not_valid]
    dfSIRET_siret_not_valid['sirenAcheteurValide'] = False
    df_keys = df_keys[~mask_siret_not_valid]
    #Le second cache des siret valide not found est traité en premier.
    cache_siret_not_found_exists = os.path.isfile(path_to_cache_not_in_insee)
    if cache_siret_not_found_exists:
        sirets_not_found = loading_cache(path_to_cache_not_in_insee)
    else:
        sirets_not_found = []
    mask_siret_valid_not_found = df_keys.siret.isin(sirets_not_found)
    dfSIRET_valide_notfound = df_keys[mask_siret_valid_not_found]
    # On retire les siret valides mais non trouvés lors des précédents passages du df.
    df_keys = df_keys[~mask_siret_valid_not_found]
    
    cache_exists = os.path.isfile(path_to_cache_insee)
    if cache_exists:
        dfcache = loading_cache(path_to_cache_insee)
        series_SIRET_not_in_cache, seriesSIRETincache = split_on_column_match(df_keys, dfcache, column="siret")
        need_refresh_cache = not(series_SIRET_not_in_cache.empty)
        if need_refresh_cache:
            logger.info("Enrichissement avec insee : Besoin d'actualiser cache")
            # Ceux pas dans le cache, ajouter au cache leur correspondant bddinsee
            dfcache, series_siret_valid_but_not_found_in_bdd = actualiser_cache(series_SIRET_not_in_cache, path_to_bdd_insee, dfcache, columns=columns, dtypes=dtypes)
            dfcache = dfcache.drop_duplicates(subset=['siret'], keep='first')
            #Update cache de la lsite des sirets valides mais non trouvés
            sirets_not_found += series_siret_valid_but_not_found_in_bdd.tolist()
            # Actualise les caches
            write_cache(dfcache, path_to_cache_insee)
            write_cache(sirets_not_found, path_to_cache_not_in_insee)
    else : 
        logger.info("Enrichissement avec insee : Création du cache")
        # crécupérer le dataframe correspondant au cache
        dfcache, series_siret_valid_but_not_found_in_bdd = actualiser_cache(df_keys.siret, path_to_bdd_insee, dfcache=pd.DataFrame(), columns=columns, dtypes=dtypes)
        dfcache = dfcache.drop_duplicates(subset=['siret'], keep='first')
        # Créer les cache
        write_cache(dfcache, path_to_cache_insee)
        write_cache(series_siret_valid_but_not_found_in_bdd.tolist(), path_to_cache_not_in_insee)
    return df

def get_enrichissement_insee(dfSIRET: pd.DataFrame, path_to_data: str, path_to_cache_bdd: str, path_to_cache_not_in_bdd: str) -> list:
    """
    Ajout des informations Adresse/Activité des entreprises via la base siren Insee par un système de double cache.
    Pour bien comprendre la fonction, il y a plusieurs cas possibles concernant un SIRET. Il y a le cas où le siret est valide et match avec la bdd insee (on gère ça avec le premier cache)
    Le cas où le siret est invalide ou OOOOOOOO (siret artificiel inscrit en amont dans enrichissement.py) il n'y a aucune chance de trouver ça dans la bdd insee donc.
    Le dernier cas où un siret est valide mais pas présent en bdd, pour ceux-ci on créé un second cache.
    Dans un cache (dfcache) sont stockés les informations en rpovenance de la bdD insee que l'on gère 
    Dans le second cache (sirets_not_found) sont stockés les siret valides que l'on doit gérer mais qui ne sotn pas dans la bdd insee

    Arguments
    -----------------
    dfSIRET 
    path_to_data chemin vers le fichier csv de la BdD Insee Etablissement_utf8
    path_to_cache_bdd chemin vers le cache du fichier csv de la BdD Insee 'Etablissement_utf8'
    path_to_cache_not_inbdd:  chemin vers le cache des siret non trouvé dans le fichier csv de la BdD Insee Etablissement_utf8


    Returns
    --------------
        - list:
            - list[0]: pd.DataFrame -- données principales
            - list[1]: pd.DataFrame -- données où le SIRET n'est pas renseigné/invalide/pas trouvé

    """
    # Traitement pour le cache. Si le siret n'est pas valide ou non renseigné, on va aller chercher dans le cache. Or on veut pas ça, donc on le gère en amont du cache.
    # Ceux qui ont un siret non valide on les vire de df SIRET, on les récupèrera plus tard.
    mask_siret_not_valid = (~dfSIRET.siret.apply(is_luhn_valid)) | (dfSIRET.siret == '00000000000000')
    dfSIRET_siret_not_valid = dfSIRET[mask_siret_not_valid]
    dfSIRET = dfSIRET[~mask_siret_not_valid]
    
    # Traitons les caches maintenant

    #Le second cache des siret valide not found est traité en premier.
    cache_siret_not_found_exists = os.path.isfile(path_to_cache_not_in_bdd)
    if cache_siret_not_found_exists:
        sirets_not_found = loading_cache(path_to_cache_not_in_bdd)
    else:
        sirets_not_found = []
    
    mask_siret_valid_not_found = dfSIRET.siret.isin(sirets_not_found)
    dfSIRET_valide_notfound = dfSIRET[mask_siret_valid_not_found]
    # On retire les siret valides mais non trouvés lors des précédents passages du df.
    dfSIRET = dfSIRET[~mask_siret_valid_not_found]
    
    cache_exists = os.path.isfile(path_to_cache_bdd)
    if cache_exists: # Il devrait pas ne pas exister, donc on rentrera toujours dans la boucle. Je laisse la condition pour qu'on comprenne si ça bug un jour en connectant à la CI etc.
        logger.info("Chargement du cache")
        dfcache = loading_cache(path_to_cache_bdd)
        # regarder les siret dans le cache, ceux pas dans le cache on va passer à travers la bdd pour les trouver. Ceux qui ne sont pas dans la BdD sont sauvés dans un 2e cache.
    enrichissement_insee_siret = pd.merge(dfSIRET, dfcache, how='left', on=['siret'], copy=False)
    enrichissement_insee_siret.rename(columns={"siren_x": "siren"}, inplace=True)
    enrichissement_insee_siret.drop(columns=["siren_y"], axis=1, inplace=True)
    df_nan_siret = pd.concat([enrichissement_insee_siret[enrichissement_insee_siret.activitePrincipaleEtablissement.isnull()], dfSIRET_siret_not_valid, dfSIRET_valide_notfound])
    enrichissement_insee_siret = enrichissement_insee_siret[
        enrichissement_insee_siret.activitePrincipaleEtablissement.notnull()]
    df_nan_siret = df_nan_siret.loc[:, ["siret", "siren", "denominationSociale"]]

    # Concaténation des deux resultats
    enrichissementInsee = enrichissement_insee_siret
    df_nan_siret = df_nan_siret.iloc[:, :3]
    df_nan_siret.reset_index(inplace=True, drop=True)
    return [enrichissementInsee, df_nan_siret]
    


def get_enrichissement_scrap(nanSiren: pd.DataFrame, archiveErrorSIRET: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement des données restantes et récupération des siret n'ayant aps de correspondance

    Return:
        - pd.DataFrame
    """

    # ....... Solution complémentaire pour ceux non-identifié dans la BDD
    columns = [
        'index',
        'rue',
        'siret',
        'ville',
        'typeEntreprise',
        'codeType',
        'detailsType',
        'SIRETisMatched']

    filter = 10
    nanSiren = nanSiren.iloc[:filter, :]

    df_scrap = pd.DataFrame(columns=columns)
    # Récupération des résultats
    nanSiren.reset_index(inplace=True)
    resultat = pd.merge(nanSiren, df_scrap, on='index', copy=False)
    resultatScrap1 = resultat[resultat.rue != ' ']

    # Données encore manquantes
    dfDS = resultat[resultat.rue == ' ']
    dfDS = dfDS.iloc[:, 1:4]
    dfDS.columns = ['siret', 'siren', 'denominationSociale']
    dfDS.reset_index(inplace=True, drop=True)
    df_scrap2 = pd.DataFrame(
        columns=['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched'])

    # Récupération des résultats
    dfDS.reset_index(inplace=True)
    resultat = pd.merge(dfDS, df_scrap2, on='index', copy=False)
    resultatScrap2 = resultat[resultat.rue != ' ']

    ###############################################################################
    # Enregistrement des entreprises n'ayant aucune correspondance
    errorSIRET = resultat[
        (resultat.siret_y == '') | (resultat.siret_y == '') | (resultat.siret_y == ' ') | (resultat.siret_y.isnull())]
    errorSIRET = errorSIRET[['siret_x', 'siren', 'denominationSociale']]
    errorSIRET.columns = ['siret', 'siren', 'denominationSociale']
    errorSIRET.reset_index(inplace=True, drop=True)
    errorSIRET = pd.concat([errorSIRET, archiveErrorSIRET], axis=0, copy=False)
    errorSIRET = errorSIRET.drop_duplicates(subset=['siret', 'siren', 'denominationSociale'], keep='first')
    errorSIRET.to_csv('errorSIRET.csv', sep=';', index=False, header=True, encoding='utf-8')
    ###############################################################################

    # On réuni les résultats du scraping
    enrichissementScrap = pd.concat([resultatScrap1, resultatScrap2], copy=False)
    return enrichissementScrap


def get_df_enrichissement(enrichissementScrap: pd.DataFrame, enrichissementInsee: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement géographique grace aux données Insee

    Returns:
        - pd.DataFrame
    """
    # Arrangement des colonnes
    # Gestion bdd insee
    enrichissementInsee.reset_index(inplace=True, drop=True)
    listCorrespondance = conf_glob["enrichissement"]["abrev2nom"]

    enrichissementInsee['typeVoieEtablissement'].replace(listCorrespondance, inplace=True)
    enrichissementInsee['rue'] = \
        (enrichissementInsee.typeVoieEtablissement + ' ' + enrichissementInsee.libelleVoieEtablissement)

    enrichissementInsee['activitePrincipaleEtablissement'] = enrichissementInsee[
        'activitePrincipaleEtablissement'].str.replace(".", "")

    # Gestion bdd scrap
    enrichissementScrap.reset_index(inplace=True, drop=True)
    enrichissementScrap["codePostal"] = np.nan
    enrichissementScrap["commune"] = np.nan
    enrichissementScrap.codePostal = enrichissementScrap.codePostal.astype(str)
    enrichissementScrap.commune = enrichissementScrap.ville.astype(str)
    enrichissementScrap.rue = enrichissementScrap.rue.astype(str)

    enrichissementScrap["codePostal"] = enrichissementScrap.ville.str[0:7]
    enrichissementScrap["codePostal"] = enrichissementScrap["codePostal"].str.replace(" ", "")
    enrichissementScrap["commune"] = enrichissementScrap.ville.str[7:]

    enrichissementScrap.drop(columns=["index", "siret_x", "ville", "typeEntreprise",
                                      "detailsType", "SIRETisMatched", "siret_y"],
                             inplace=True, errors="ignore")
    enrichissementInsee.drop(columns=["nic", "typeVoieEtablissement", "libelleVoieEtablissement",
                                      "nomenclatureActivitePrincipaleEtablissement"],
                             inplace=True, errors="ignore")

    # Renomme les colonnes
    enrichissementScrap.rename(columns={
        'rue': 'adresseEtablissement',
        'codeType': 'codeTypeEtablissement',
        'codePostal': 'codePostalEtablissement',
        'commune': 'communeEtablissement'
    }, inplace=True, errors="ignore")
    enrichissementInsee.rename(columns={
        'libelleCommuneEtablissement': 'communeEtablissement',
        'activitePrincipaleEtablissement': 'codeTypeEtablissement',
        'rue': 'adresseEtablissement'
    }, inplace=True, errors="ignore")

    enrichissementInsee = enrichissementInsee[[
        'siret',
        'siren',
        'denominationSociale',
        'codePostalEtablissement',
        'communeEtablissement',
        'codeCommuneEtablissement',
        'codeTypeEtablissement',
        'adresseEtablissement']]

    # df final pour enrichir les données des entreprises
    dfenrichissement = pd.concat([enrichissementInsee, enrichissementScrap], copy=False)
    dfenrichissement = dfenrichissement.astype(str)
    # On s'assure qu'il n'y ai pas de doublons
    dfenrichissement = dfenrichissement.drop_duplicates(subset=['siret'], keep=False)

    return dfenrichissement


def enrichissement_cpv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Récupération des codes CPV formatés.

    Return:
        - pd.Dataframe
    """
    print("CPV", df.shape)
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


def enrichissement_acheteur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichissement des données des acheteurs via les codes siret/siren.
    Dans cette fonction du pipeline, on ré-utilise la Bdd insee (et donc on va ré utiliser les caches utilisés pour enrichissement_insee)

    Return:
        - pd.DataFrame
    """
    print('acheteur', df.shape)
    # StockEtablissement_utf8 et les caches
    path_to_cache_bdd = os.path.join(path_to_cache, conf_data["cache_bdd_insee"])
    path_to_cache_not_in_bdd = os.path.join(path_to_cache, conf_data["cache_not_in_bdd_insee"])
    
    # Chargement des caches qui existent forcément. Donc pas de test sur leur existence
    sirets_not_found = loading_cache(path_to_cache_not_in_bdd)
    mask_siret_valid_not_found = df.siret.isin(sirets_not_found)
    df_siret_valid_not_found = df[mask_siret_valid_not_found]
    df = df[~mask_siret_valid_not_found]
    mask_siret_not_valid = (~df.siret.apply(is_luhn_valid)) | (df.siret == '00000000000000')
    dfSIRET_siret_not_valid = df[mask_siret_not_valid]
    df = df[~mask_siret_not_valid]

    logger.info("Début du traitement: Enrichissement acheteur")
    dfAcheteurId = df['acheteur.id'].to_frame()
    dfAcheteurId = dfAcheteurId.rename(columns={'acheteur.id' : 'siret'})
    dfAcheteurId = dfAcheteurId.drop_duplicates(keep='first')
    dfAcheteurId.reset_index(inplace=True, drop=True)
    dfAcheteurId = dfAcheteurId.astype(str)

    
    dfcache = loading_cache(path_to_cache_bdd)
    
    # Colonnes utiles l'enrichissement
    usecols=['siret', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement']
    dfcache_acheteur = dfcache[usecols].copy()
    enrichissementAcheteur =  pd.merge(dfAcheteurId, dfcache_acheteur, on="siret", copy=False)
    enrichissementAcheteur = enrichissementAcheteur.drop_duplicates(subset=['siret'], keep='first')
    enrichissementAcheteur.columns = ['acheteur.id', 'codePostalAcheteur', 'libelleCommuneAcheteur',
                                      'codeCommuneAcheteur']
    enrichissementAcheteur = enrichissementAcheteur.drop_duplicates(subset=['acheteur.id'], keep='first')

    df = pd.merge(df, enrichissementAcheteur, how='left', on='acheteur.id', copy=False)
    df = pd.concat([df, dfSIRET_siret_not_valid, df_siret_valid_not_found])

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
    Ajout des géolocalisations des entreprises

    Return:
        - pd.DataFrame
    """
    print("geo", df.shape)
    logger.info("Début du traitement: Enrichissement geographique")
    # Enrichissement latitude & longitude avec adresse la ville
    df.codeCommuneAcheteur = df.codeCommuneAcheteur.astype(object)
    df.codeCommuneEtablissement = df.codeCommuneEtablissement.astype(object)
    df_villes = get_df_villes()
    # Ajout des données Acheteur
    df = pd.merge(df, df_villes, how='left', left_on="codeCommuneAcheteur", right_on="codeCommune", copy=False)
    df.rename(columns={"superficie": "superficieCommuneAcheteur",
                       "population": "populationCommuneAcheteur",
                       "latitude": "latitudeCommuneAcheteur",
                       "longitude": "longitudeCommuneAcheteur"},
              inplace=True)
    df.drop(columns="codeCommune", inplace=True)
    # Ajout des données Etablissements
    df = pd.merge(df, df_villes, how='left', left_on="codeCommuneEtablissement", right_on='codeCommune', copy=False)
    df.rename(columns={"superficie": "superficieCommuneEtablissement",
                       "population": "populationCommuneEtablissement",
                       "latitude": "latitudeCommuneEtablissement",
                       "longitude": "longitudeCommuneEtablissement"},
              inplace=True)
    df.drop(columns="codeCommune", inplace=True)

    # Calcul de la distance entre l'acheteur et l'etablissement
    df['distanceAcheteurEtablissement'] = df.apply(get_distance, axis=1)

    # Remise en forme des colonnes géo-spatiales
    cols = ["longitudeCommuneAcheteur",
            "latitudeCommuneAcheteur",
            "longitudeCommuneEtablissement",
            "latitudeCommuneEtablissement"]
    df[cols] = df[cols].astype(str)
    df['geolocCommuneAcheteur'] = df.latitudeCommuneAcheteur + ',' + df.longitudeCommuneAcheteur
    df['geolocCommuneAcheteur'] = np.where(
        df['geolocCommuneAcheteur'] == 'nan,nan', np.NaN, df['geolocCommuneAcheteur'])

    df['geolocCommuneEtablissement'] = df.latitudeCommuneEtablissement + ',' + df.longitudeCommuneEtablissement
    df['geolocCommuneEtablissement'] = np.where(
        df['geolocCommuneEtablissement'] == 'nan,nan', np.NaN, df['geolocCommuneEtablissement'])
    df.reset_index(inplace=True, drop=True)
    return df


def get_df_villes() -> pd.DataFrame:
    """
    Récupération des informations sur les communes (superficie/population)

    Return:
        - pd.DataFrame
    """
    path = os.path.join(path_to_data, conf_data["base_geoflar"])
    df_villes = pd.read_csv(path, sep=';', header=0, error_bad_lines=False,
                            usecols=['INSEE_COM', 'Geo Point', 'SUPERFICIE', 'POPULATION'])

    # Suppression des codes communes sans point geo
    df_villes = df_villes[(df_villes['INSEE_COM'].notnull()) & (df_villes['Geo Point'].notnull())]
    df_villes.reset_index(inplace=True, drop=True)

    # Séparation de la latitude et la longitude depuis les info géo
    df_villes['Geo Point'] = df_villes['Geo Point'].astype(str)
    df_sep = pd.DataFrame(df_villes['Geo Point'].str.split(',', 1, expand=True))
    df_sep.columns = ['latitude', 'longitude']

    # Fusion des lat/long dans le df
    df_villes = df_villes.join(df_sep)

    # Suppression des colonnes inutiles
    df_villes.drop(columns=["Geo Point"], inplace=True, errors="ignore")

    # Renommer les variables
    df_villes.rename(columns={"INSEE_COM": 'codeCommune',
                              "POPULATION": 'population',
                              "SUPERFICIE": 'superficie'},
                     inplace=True)

    # Conversion des colonnes
    df_villes.population = df_villes.population.astype(float)
    df_villes.codeCommune = df_villes.codeCommune.astype(object)
    df_villes.latitude = df_villes.latitude.astype(float)
    df_villes.longitude = df_villes.longitude.astype(float)

    return df_villes


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


def change_sources_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a dataframe in input, and based on a dict will rename the source
    """
    print('change sources', df.shape)
    dict_replace_name_source = {
        "data.gouv.fr_aife": "API AIFE",
        "data.gouv.fr_pes": "PES Marchés",
        "marches-publics.info": "AWS-Achat",
        "megalis-bretagne": "Megalis Bretagne",
        "atexo-maximilien": "Maximilien IdF",
        "ternum-bfc": "Territoires numériques BFC",
        "e-marchespublics": "Dematis",
        "grandlyon": "Grand Lyon"
    }
    df["source"].replace(dict_replace_name_source, inplace=True)
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
        with open("profilingSenrichissement_size{}.txt".format(init_len), "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats('ncalls')
            ps.sort_stats('cumulative')
            ps.print_stats()
    else:
        main()