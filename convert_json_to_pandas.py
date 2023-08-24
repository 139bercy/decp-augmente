from pandas import json_normalize
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm  # Import tqdm

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)


def manage_modifications(data: dict) -> pd.DataFrame:
    """
    Conversion du json en pandas et incorporation des modifications

    Retour:
        pd.DataFrame
    """
    L_indice = indice_marche_avec_modification(data)
    dict_modification = recuperation_colonne_a_modifier(data, L_indice)
    df = json_normalize(data['marches'])
    # Replace empty strings with NaN (Not a Number) and convert to float
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.astype(conf_glob["nettoyage"]['type_col_nettoyage'], copy=False)
    prise_en_compte_modifications(df)
    #df = regroupement_marche(df, dict_modification)
    # save df to pickle
    df.to_pickle(os.path.join("data", "dfafterconvertsmall.pkl"))
    return df


def indice_marche_avec_modification(data: dict) -> list:
    """
    Renvoie la liste des indices des marchés contenant une modification

    Retour:
        - list
    """
    liste_indices = []
    for i in range(len(data["marches"])):
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
        raise ValueError("Il n'y a aucune colonne du nom de {} dans le dataframe entrée en paramètre".format(col_to_normalize))
    to_normalize = df[col_to_normalize]  # Récupération de la colonne à splitter
    df["booleanModification"] = 0
    for i in range(len(to_normalize)):
        json_modification = to_normalize[i]
        if json_modification != []:  # dans le cas ou des modifications ont été apportées
            for col in json_modification[0].keys():
                col_init = col
                # Formatage du nom de la colonne
                if "Modification" not in col:
                    col += "Modification"
                if col not in df.columns:  # Cas ou on tombe sur le premier marche qui modifie un champ
                    df[col] = ""  # Initialisation dans le df initial
                df[col][i] = json_modification[0][col_init]
                df["booleanModification"][i] = 1  # Création d'une booléenne pour simplifier le subset pour la suite


def regroupement_marche(df: pd.DataFrame, dict_modification: dict) -> pd.DataFrame:
    """
    Permet de recoder la variable identifiant.
    Actuellement: 1 identifiant par déclaration (marché avec ou sans modification)
    Un marché peut être déclaré plusieurs fois en fonction du nombre d'entreprise. Si 2 entreprises sur
    En sortie: id correspondra à un identifiant unique pour toutes les lignes composants un marché SI il a eu une modification
    Modification inplace du df source

    Retour:
        pd.DataFrame
    """
    df["idtech"] = ""
    subdata_modif = df[df.booleanModification == 1]  # Tout les marchés avec les modifications
    liste_objet = list(subdata_modif.objet.unique())
    df_to_concatene = pd.DataFrame()  # df vide pour la concaténation
    for objet_marche in tqdm(liste_objet, desc="Processing objet marchés"):
        # Récupération du dataframe modification et du dataframe source
        marche, marche_init = split_dataframe(df, subdata_modif, objet_marche)
        for j in range(len(marche)):
            marche_init = fusion_source_modification(marche.iloc[j], marche_init, dict_modification.keys(), dict_modification)
        marche_init["idtech"] = marche.iloc[-1].id_technique
        df_to_concatene = pd.concat([df_to_concatene, marche_init], copy=False)
    df.update(df_to_concatene)
    # Attention aux id.
    df["idMarche"] = np.where(df.idtech != "", df.idtech, df.id_technique)
    return df


def split_dataframe(df: pd.DataFrame, sub_data: pd.DataFrame, modalite: str) -> tuple:
    """
    Définition de deux dataFrame.
        - Le premier qui contiendra uniquement les lignes avec modification, pour le marché ayant pour objet modalite
        - Le second contiendra l'ensemble des lignes correspondant au marché isolé dans le df1 qui ont pour objet modalite

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
    return (marche, marche_init)


def fusion_source_modification(raw: pd.DataFrame, df_source: pd.DataFrame, col_modification: list, dict_modification: dict) -> pd.DataFrame:
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


def regroupement_marche_(df: pd.DataFrame, dict_modification: dict) -> pd.DataFrame:
    # Assuming you've generated the 'booleanModification' column in df

    # Create a dictionary to store the modified marché dataframes
    modified_marches = {}

    # Group data by "objet" and loop through each group
    for objet_marche, group in tqdm(df[df["booleanModification"] == 1].groupby("objet"), desc="Processing objet marchés"):
        # Initialize the modified marché dataframe with the last row of the group
        modified_marche = group.iloc[-1].copy()

        # Apply modifications from the dict_modification
        modified_marche = fusion_source_modification(modified_marche, group, dict_modification.keys(),
                                                     dict_modification)

        # Store the modified marché dataframe
        modified_marches[objet_marche] = modified_marche

    # Update the original dataframe with the modified marché dataframes
    for objet_marche, modified_marche in tqdm(modified_marches.items(), desc="Updating original dataframe"):
        mask = df["objet"] == objet_marche
        df.loc[mask] = modified_marche

    # Update idMarche based on idtech or id_technique
    df["idMarche"] = np.where(df["idtech"] != "", df["idtech"], df["id_technique"])

    return df
