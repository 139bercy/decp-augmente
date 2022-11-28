import pandas as pd
import os
import numpy as np
import pickle
import json
import datetime
import logging.handlers
from pandas.util import hash_pandas_object
from pandas import json_normalize



with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)
path_to_data = conf_data["path_to_data"]
decp_file_name = conf_data["decp_file_name"]

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

with open(os.path.join("confs", "var_debug.json")) as f:
    conf_debug = json.load(f)["nettoyage"]

logger = logging.getLogger("main.gestion_flux")
logger.setLevel(logging.DEBUG)

def main():
    check_reference_files()
    logger.info("Ouverture du fichier decp.json d'aujourd'hui")
    with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
        data = json.load(json_data)
    
    df_decp = json_normalize(data['marches'])
    print('original', df_decp.shape)
    logger.info("Séparation du DataFrame en deux : marchés avec et sans modifications")

    df_modif, df_no_modif = split_dataframes_according_to_modifications(df_decp)

    #Gestion de la partie avec les modifications
    logger.info("Création clef de hash pour les marchés ayant des modifications de decp.json")
    df_modif = create_hash_key_for_modifications(df_modif)
    logger.info("Comparaison des clefs de hash calculées avec celles correspondant aux lignes modifications déjà enrichies.")
    df_modif_to_process, df_modif_processed = differenciate_according_to_hash(df_modif,conf_data["hash_modifications"])
    #Sauvegarde clefs de hache
    with open(os.path.join(path_to_data, conf_data["hash_modifications"]), "wb") as f:
        pickle.dump(df_modif.hash_key, f)
    #Gestion de la partie sans les modifications
    logger.info("Création clef de hash pour les marchés n'ayant pas de modifications de decp.json")
    df_no_modif = create_hash_key_for_no_modification(df_no_modif)
    logger.info("Comparaison des clefs de hash calculées avec celles correspondant aux lignes déjà enrichies.")
    df_no_modif_to_process, df_no_modif_processed = differenciate_according_to_hash(df_no_modif, conf_data["hash_no_modifications"])
    #Sauvegarde clefs de hache
    with open(os.path.join(path_to_data, conf_data["hash_no_modifications"]), "wb") as f:
        pickle.dump(df_no_modif.hash_key, f)
    print('Shape no modif to process puis process')
    print(df_no_modif_to_process.shape)
    print('\n', df_no_modif_processed.shape)
    print('Shape modif to process puis process')
    print(df_modif_to_process.shape)
    print('\n', df_modif_processed.shape)
    # Concaténation des dataframes à processer et mise de côté ceux déjà processé
    df_to_process = pd.concat([df_no_modif_to_process, df_modif_to_process]).reset_index(drop=True)
    #Sauvegarde du Dataframe à processer, et donc à envoyer en entrée de nettoyage
    with open("df_flux", "wb") as file:
        pickle.dump(df_to_process, file)
    return None


def concat_modifications(dictionaries : list):
    """
    Parfois, certains marché ont plusieurs modifications (la colonne modification est une liste de dictionnaire).
    Jusqu'alors, seul le premier élément de la liste (et donc la première modification) était pris en compte. 
    Cette fonction met à jour le premier dictionnaire de la liste. Ainsi les modifications considérées par la suite seront bien les dernières.

    Arguments
    ------------
    dictionnaries (list) liste des dictionnaires de modifications

    Returns
    ----------
    Une liste d'un élément : le dictionnaire des modifications à considérer.

    """
    dict_original = dictionaries[0]
    for dict in dictionaries: # C'st une boucle sur quelques éléments seulement, ça devrait pas poser trop de problèmes.
        dict_original.update(dict)
    return [dict_original]

def explode_according_to_keys(df :pd.DataFrame ,keys):
    """
    Cette fonction retourne un dataframe avec autant de colonne que de clef dans keys. 
    La colonne est complétée si une information est trouvé, Nan sinon.

    Arguments
    ---------
    df le dataframe composé d'une colonne (une série donc) qu'on va exploser
    keys un array/une liste de clef
    """
    df_explode = pd.DataFrame()
    for key in keys:
        df_explode[key] = df.apply(lambda x:x[0].get(key)) #x[0] car après concat_modifications le dictionnaire des modifications update est à la position 0 de la liste
    return df_explode

def transform_titulaires(x):
    """Les titulaires sont encore des listes de dictionnaires, des formats d'objets mutables en python et donc non hashable.
    On va donc récupérer uniquement l'id des titulaires pour en faire un tuple, qui lui est hashable.


    Returns
    -----------
    un tuple des id des titulaires ou None si l'objet d'entrée est None.

    """
    if type(x)==list: # On peut avoir un None également
        try :
            return tuple([y.get("id") for y in x])
        except : # J'ai vu 3 lignes qui ont un format de donnée étrange : une liste de liste d'un élément. Pour le considérer je mets un try except car a priori le except arrivera très rarement.
            return tuple([y.get("id") for y in x[0]])
    else :
        return x

def create_hash_key_for_modifications(df_decp_modif : pd.DataFrame):
    """
    Cette fonction génère une clef de hash pour le dataframe en entrée, celui des modifications qui a ses spécificités.

    Arguments
    ---------
    df_decp_modif le dataframe contenant uniquement les modifications


    Returns
    ------------
    df_decp_modif enrichi de la clef de hachage.

    """
    df_decp_modif['modif_up'] = df_decp_modif.modifications.apply(concat_modifications) # On rassemble les modifications 
    columns_modification = df_decp_modif.modif_up.apply(lambda x:list(x[0].keys())).explode().unique() # Permet de récupérer toutes les clefs possibles même si le format évolue
    print('cool')
    # On sauvegarde coluns_modification pour le réutiliser dans nettoyage
    with open("columns_modifications", "wb") as file_modif:
        pickle.dump(columns_modification, file_modif)
    df_modification_explode = explode_according_to_keys(df_decp_modif.modif_up, columns_modification)
    # A ce stade, les titulaires sont encore des listes de dictionnaires, donc non hashables. Transformons-les.
    df_modification_explode['titulaires_transfo'] = df_modification_explode.loc[:, "titulaires"].apply(transform_titulaires)
    subset_to_hash_modif = conf_glob["gestion_flux"]["subset_for_hash_modifications"]
    # Mettre le subset_to_hash_modif dans un JSON externable ?
    hash_modif = hash_pandas_object(df_modification_explode.loc[:, subset_to_hash_modif], index=False) # index doit toujours rester à False, sinon la clef de hash prends en compte l'index (ce qu'on ne veut pas)
    df_decp_modif['hash_key'] = hash_modif

    return df_decp_modif

def differenciate_according_to_hash(df : pd.DataFrame, path_to_hash_pickle, hash_column="hash_key"):
    """
    Cette fonction permet de différencier les nouvelles lignes en comparant les clefs de hash calculées pour le decp actuellement récupéré avec les clefs de hash déjà en mémoire.
    Elle ré-écrit également le cache de mémoire avec l'ensemble des clefs (anciennes+nouvelles).

    Arguments
    ----------
    df

    Returns
    ----------
    Deux DataFrames, l'un avec les lignes à traiter, l'autre avec les lignes déjà traitées.
    """
    path_to_hash_cache = os.path.join(path_to_data, path_to_hash_pickle)
    exists_path = os.path.isfile(path_to_hash_cache)
    if exists_path : 
        with open(path_to_hash_cache, "rb") as file_hash_modif:
            hash_processed = pickle.load(file_hash_modif)
        mask_hash_to_process = df.loc[:, str(hash_column)].isin(hash_processed)

        return df[~mask_hash_to_process], df[mask_hash_to_process]
    else:
        return df, pd.DataFrame()

def split_dataframes_according_to_modifications(df_decp : pd.DataFrame):
    """
    Cette fonction renvoie deux dataframes.
    Le premier contient les lignes ayant des modifications.
    Le second les lignes sans modifications.
    Pourquoi ce choix ? 
    Lorsqu'un marché déjà existant est actualisé, rien ne change sauf le contenu de la colonne modification.
    Contenu qui a un format particulier donc on souhaitait le traiter à part.

    """
    mask_modifications = df_decp.modifications.apply(len)>0
    df_decp_modif = df_decp[mask_modifications]
    df_decp_no_modif = df_decp[~mask_modifications]

    return df_decp_modif, df_decp_no_modif

def create_hash_key_for_no_modification(df : pd.DataFrame):
    """
    Cette fonction calcule les clefs de hachage pour le dataframe d'entrée et ajoute cette information au dataframe.

    Arguments
    ---------
    df (pd.DataFrame) : DataFrame des lignes sans modifications.

    Returns
    ---------
    le dataframe d'entrée enrichi de la colonne contenant les clefs de hachage
    """
    subset_to_hash_no_modif = conf_glob["gestion_flux"]["subset_for_hash_no_modifications"]
    hash_keys = hash_pandas_object(df.loc[:, subset_to_hash_no_modif], index=False)
    df['hash_key'] = hash_keys
    logger.info("Cache des clefs de hachage actualisé")

    return df



def check_reference_files():
    """
    Vérifie la présence des fichiers datas nécessaires, dans le dossier data.
        StockEtablissement_utf8.csv, cpv_2008_ver_2013.xlsx, geoflar-communes-2015.csv,
        departement2020.csv, region2020.csv, StockUniteLegale_utf8.csv
    """
    path_data = conf_data["path_to_data"]

    useless_keys = ["path_to_project", "path_to_data", "path_to_cache", "cache_bdd_insee",
                     "cache_not_in_bdd_insee","cache_bdd_legale",
                     "cache_not_in_bdd_legale","cache_df",
                     "hash_modifications", "hash_no_modifications"]

    path = os.path.join(os.getcwd(), path_data)
    for key in list(conf_data.keys()):
        if key not in useless_keys:
            logger.info(f'Test du fichier {conf_data[key]}')
            mask = os.path.exists(os.path.join(path, conf_data[key]))
            if not mask:
                logger.error(f"Le fichier {conf_data[key]} n'existe pas")
                raise ValueError(f"Le fichier data: {conf_data[key]} n'a pas été trouvé")


if __name__ == "__main__":
    main()