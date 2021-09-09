import json
import os
import pickle
import logging
import logging.handlers
import numpy as np
import pandas as pd
from pandas import json_normalize



with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)
path_to_data = conf_data["path_to_data"]
decp_file_name = conf_data["decp_file_name"]

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

logger = logging.getLogger("main.nettoyage")
logger.setLevel(logging.DEBUG)




def main():
    check_reference_files(conf_data)
    logger.info("Ouverture du fichier decp.json")
    with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
        data = json.load(json_data)

    logger.info("Début du traitement: Conversion des données en pandas")
    df = manage_modifications(data)
    logger.info("Fin du traitement")

    df = regroupement_marche_complet(df)

    logger.info("Début du traitement: Gestion des titulaires")
    df = manage_titulaires(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Suppression des doublons")
    df = manage_duplicates(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Déection et correction des montants aberrants")
    df = manage_amount(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Gestion des Id null")
    df = manage_missing_code(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Attribution et correction des régions/déprtements (code + libelle). Zone d'execution du marché")
    df = manage_region(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Récupération de l'année et du mois du marché public + Correction des années aberrantes")
    df = manage_date(df)
    logger.info("Fin du traitement")
    
    logger.info("Début du traitement: Correction de la variable dureeMois.")
    df = correct_date(df)
    logger.info("Fin du traitement")

    logger.info("Début du tritement: Imputation de la variable dureeMois")
    df = data_inputation(df)
    logger.info("Fin du traitement")

    logger.info("Début du traitement: Remplacement des caractères mal converti")
    # suppression des caractères mal encodés
    df = replace_char(df)
    logger.info("Fin du traitement")

    logger.info("Creation csv intermédiaire: decp_nettoye.csv")
    with open('df_nettoye', 'wb') as df_nettoye:
        pickle.dump(df, df_nettoye)  # Export présent pour faciliter l'utilisation du module enrichissement.py
    df.to_csv("decp_nettoye.csv")
    logger.info("Ecriture du csv terminé")


def check_reference_files(conf_data):
    """Vérifie la présence des fichiers datas nécessaires, dans le dossier data.
        StockEtablissement_utf8.csv, cpv_2008_ver_2013.xlsx, "geoflar-communes-2015.csv", departement2020.csv, region2020.csv, StockUniteLegale_utf8.csv"""
    path_to_data = conf_data["path_to_data"]
    L_key_useless = ["path_to_project", "path_to_data"]
    path = os.path.join(os.getcwd(), path_to_data)
    for key in list(conf_data.keys()):
        if key not in L_key_useless:
            logger.info('Test du fichier {}'.format(conf_data[key]))
            mask = os.path.exists(os.path.join(path, conf_data[key]))
            if not mask:
                logger.error("Le fichier {} n'existe pas".format(conf_data[key]))
                raise ValueError("Le fichier data: {} n'a pas été trouvé".format(conf_data[key]))


def manage_titulaires(df):
    # Ecriture dans les logs
    logger.info("Nombre de marché sans titulaires: {}. Remplacé par la valeur du concessionnaire".format(sum(df["titulaires"].isnull())))
    logger.info("Nombre de marché sans montant: {}. Remplacé par la valeur globale".format(sum(df["montant"].isnull())))
    logger.info("Nombre de marché sans identifiant acheteur: {}. Remplacé par l'identifiatn de l'autorité Concedante".format(sum(df["acheteur.id"].isnull())))
    logger.info("Nombre de marché sans nom d'acheteur: {}. Remplacé par le nom de l'autorité Concédante".format(sum(df["acheteur.nom"].isnull())))

    # Gestion différences concessionnaires / titulaires
    df.titulaires = np.where(df["titulaires"].isnull(), df.concessionnaires, df.titulaires)
    df.montant = np.where(df["montant"].isnull(), df.valeurGlobale, df.montant)
    df['acheteur.id'] = np.where(df['acheteur.id'].isnull(), df['autoriteConcedante.id'], df['acheteur.id'])
    df['acheteur.nom'] = np.where(df['acheteur.nom'].isnull(), df['autoriteConcedante.nom'], df['acheteur.nom'])
    donneesInutiles = ['dateSignature', 'dateDebutExecution', 'valeurGlobale', 'donneesExecution', 'concessionnaires',
                       'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom', 'idtech', "id_technique"]
    df.drop(columns=donneesInutiles, inplace=True)

    # Récupération des données titulaires
    df["titulaires"].fillna('0', inplace=True)
    df = df[df['titulaires'] != '0']

    # Création d'une colonne nbTitulairesSurCeMarche. Cette colonne sera retravaillé dans la fonction detection_accord_cadre
    df.loc[:, "nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x: len(x))

    df_titulaires = pd.concat([pd.DataFrame.from_records(x) for x in df['titulaires']],
                              keys=df.index).reset_index(level=1, drop=True)
    df_titulaires.rename(columns={"id": "idTitulaires"}, inplace=True)
    df = df.drop('titulaires', axis=1).join(df_titulaires).reset_index(drop=True)

    return df


def manage_duplicates(df):
    """Permet la suppression des eventuels doublons pour un subset du dataframe"""
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
    logger.info("Nombre de lignes doublons supprimées: {}".format(nb_ligne_avant_suppression - nb_ligne_apres_suppresion))

    # Correction afin que ces variables soient représentées identiquement
    df['formePrix'] = np.where(df['formePrix'].isna(), np.nan, df['formePrix'])
    df['formePrix'] = np.where('Ferme, actualisable' == df['formePrix'], 'Ferme et actualisable', df['formePrix'])
    df['procedure'] = np.where('Appel d’offres restreint' == df['procedure'], "Appel d'offres restreint", df['procedure'])

    return df


def is_false_amount(x, threshold=5):
    """On cherche à vérifier si des montants ne sont composés que d'un seul chiffre. exemple: 999 999.
    Ces montants seront considérés comme faux"""
    d = [0] * 10
    str_x = str(abs(int(x)))
    for c in str_x:
        d[int(c)] += 1
    for counter in d[1:]:
        if counter > threshold:
            return True
    return False


def manage_amount(df):
    """Travail sur la détection des montants erronés. Ici inférieur à 200, supérieur à 9.99e8 et composé d'un seul chiffre: 999 999"""
    # Identifier les outliers - travail sur les montants
    df["montant"] = pd.to_numeric(df["montant"])
    df['montantOriginal'] = df["montant"]
    df['montant'].fillna(0, inplace=True)
    # variable témoin pour les logs
    nb_montant_egal_zero = df.montant.value_counts()[0]
    df["montant"] = df["montant"].apply(lambda x: 0 if is_false_amount(x) else abs(x))

    logger.info("{} montant(s) correspondaient à des suites d'un seul chiffre. Exemple: 9 999 999".format(df.montant.value_counts()[0] - nb_montant_egal_zero))
    nb_montant_egal_zero = df.montant.value_counts()[0]
    borne_inf = 200.0
    borne_sup = 9.99e8
    df["montant"] = df["montant"] / df["nbTitulairesSurCeMarche"]
    df['montant'] = np.where(df['montant'] <= borne_inf, 0, df['montant'])
    logger.info("{} montant(s) étaient inférieurs à la borne inf {}".format(df.montant.value_counts()[0] - nb_montant_egal_zero, borne_inf))
    nb_montant_egal_zero = df.montant.value_counts()[0]
    df['montant'] = np.where(df['montant'] >= borne_sup, 0, df['montant'])
    logger.info("{} montant(s) étaient supérieurs à la borne sup: {}".format(df.montant.value_counts()[0] - nb_montant_egal_zero, borne_sup))
    df = df.rename(columns = {"montant": "montantCalcule"})
    # Colonne supplémentaire pour indiquer si la valeur est estimée ou non
    df['montantEstime'] = np.where(df['montantCalcule'] != df.montantOriginal, True, False)
    # Ecriture dans la log
    logger.info("Au total, {} montant(s) ont été corrigé (on compte aussi les montants vides).".format(sum(df.montantEstime)))
    return df


def manage_missing_code(df):
    """Travail sur les variables d'identifiant """
    # Ecriture dans les logs
    logger.info("Nombre d'identifiant manquants et remplacés: {}".format(sum(df["id"].isnull())))
    logger.info("Nombre de code CPV manquants et remplacés: {}".format(sum(df["codeCPV"].isnull())))

    df.id = np.where(df["id"].isnull(), '0000000000000000', df.id)
    df.codeCPV = np.where(df["codeCPV"].isnull(), '00000000', df.codeCPV)

    # Nettoyage des codes idTitulaires
    logger.info("Nettoyage des idTitualires")
    caracteres_speciaux_dict = conf_glob["nettoyage"]["caractere_speciaux"]
    mask = (df.typeIdentifiant == 'SIRET') | \
           (df.typeIdentifiant.isnull()) | \
           (df.typeIdentifiant == 'nan')
    df.idTitulaires[mask].replace(caracteres_speciaux_dict, inplace=True)
    df.idTitulaires = np.where(df.idTitulaires == '', np.NaN, df.idTitulaires)
    # Ecriture dans les logs
    logger.info("Nombre d'identifiant titualire ou un traitement sur les caractères spéciaux a été fait: {}".format(sum(mask)))

    # Récupération code NIC: 5 dernier chiffres du Siret <- idTitulaires
    logger.info("Récupération du code NIC")
    df.idTitulaires = df.idTitulaires.astype(str)
    df['nic'] = df["idTitulaires"].str[-5:]
    df.nic = np.where(~df["nic"].str.isdigit(), np.NaN, df.nic)
    df['nic'] = df.nic.astype(str)

    # Gestion code CPV
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

    return df


def manage_region(df):
    """Ajout des libellés Régions/département pour le lieu d'execution du marché"""
    # Régions / Départements #
    # Création de la colonne pour distinguer les départements
    logger.info("Création de la colonne département Execution")
    df['codeDepartementExecution'] = df['lieuExecution.code'].str[:3]
    listCorrespondance = conf_glob["nettoyage"]["DOM2name"]
    df['codeDepartementExecution'].replace(listCorrespondance, inplace=True)

    df['codeDepartementExecution'] = df['codeDepartementExecution'].str[:2]

    listCorrespondance2 = conf_glob["nettoyage"]["name2DOMCOM"]
    df['codeDepartementExecution'].replace(listCorrespondance2, inplace=True)

    # Vérification si c'est bien un code département
    listeCP = conf_glob["nettoyage"]["code_CP"].split(',') \
        + [str(i) for i in list(np.arange(10, 96, 1))]
    df['codeDepartementExecution'] = np.where(~df['codeDepartementExecution'].isin(listeCP), np.NaN, df['codeDepartementExecution'])

    # Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)
    df['lieuExecution.typeCode'] = np.where(df['lieuExecution.typeCode'].isna(), np.NaN, df['lieuExecution.typeCode'])
    df['codeDepartementExecution'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN, df['codeDepartementExecution'])

    # Récupération des codes régions via le département
    logger.info("Ajout des code regions pour le lieu d'execution")
    path_dep = os.path.join(path_to_data, conf_data["departements-francais"])
    departement = pd.read_csv(path_dep, sep=",", usecols=['dep', 'reg', 'libelle'], dtype={"dep": str, "reg": str, "libelle": str})
    df['codeDepartementExecution'] = df['codeDepartementExecution'].astype(str)
    df = pd.merge(df, departement, how="left",
                  left_on="codeDepartementExecution", right_on="dep")
    df.rename(columns={"reg": "codeRegionExecution"}, inplace=True)
    # On supprime la colonne dep, doublon avec codeDepartementExecution
    del df["dep"]
    # Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
    df['codeRegionExecution'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'], df['codeRegionExecution'])
    df['codeRegionExecution'] = df['codeRegionExecution'].astype(str)
    # Vérification des codes région
    listeReg = conf_glob["nettoyage"]["code_reg"].split(',')  # 98 = collectivité d'outre mer

    df['codeRegionExecution'] = np.where(~df['codeRegionExecution'].isin(listeReg), np.NaN, df['codeRegionExecution'])
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
    return df


def manage_date(df):
    """Travail sur les Dates. Récupération de l'année de notification du marché ainsi que son mois"""
    # Date / Temps #
    # ..............Travail sur les variables de type date
    df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
    df.dateNotification = df.dateNotification.str[0:10]
    # On récupère l'année de notification
    logger.info("Récupération de l'année")
    df['anneeNotification'] = df.dateNotification.str[0:4]
    # Pour les annee non renseignée (égale à '') on les mets à 0000, cas traité 3 lignes plus tard 
    # df['anneeNotification'] = np.where(df.anneeNotification == "", "0000", df.anneeNotification)
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
    logger.info("Au total, {} marchés n'ont pas de date de publication des données connue".format(sum(df["datePublicationDonnees"].isna())))

    return df


def correct_date(df):
    """Travail sur les durées des contrats. Recherche des durées exprimées en Jour et non pas en mois"""
    # On cherche les éventuelles erreurs mois -> jours
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
    df['dureeMoisCalculee'] = np.where(df['dureeMoisCalculee'] <= 0, 1, df['dureeMoisCalculee'])  # Il y a une valeur négative
    logger.info("Au total, {} duree de marché en mois ont été jugées rentrées en jour et non en mois.".format(sum(mask)))

    return df


def data_inputation(df):
    """Permet une estimation de la dureeMois (pour les durees évidemment fausses) grace au contenu de la commande (codeCPV)"""
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
    logger.info("Au total, {} duree en mois sont encore supérieures à 120 (et les marchés ne concernent pas le monde des travaux)".format(sum(mask)))
    df.dureeMoisCalculee = np.where(mask, df.mediane_dureeMois_CPV, df.dureeMoisCalculee)
    # On modifie au passage la colonne dureeMoisEstimee
    df['dureeMoisEstimee'] = np.where(mask, "True", df.dureeMoisEstimee)
    # La mediane n'est pas utile dans le df final: on supprime
    df.drop("mediane_dureeMois_CPV", axis=1, inplace=True)
    return df


def replace_char(df):
    """Fonction pour mettre en qualité le champ objet"""
    # Remplacement brutal du caractère (?) par un espace
    df['objet'] = df['objet'].str.replace('�', 'XXXXX')
    return df

# Pour la PR
# Cette fonction doit s'insérer après le traitement de la branche Add feature modifications

def regroupement_marche_complet(df):
    """la colonne id n'est pas unique. Cette fonction permet de la rendre unique en regroupant 
    les marchés en fonction de leur objets/date de publication des données et montant.
    Ajoute dans le meme temps la colonne nombreTitulaireSurMarchePresume"""
    # Creation du sub DF necessaire
    df_titulaires = pd.DataFrame()
    df_to_update = pd.DataFrame()
    df_intermediaire = df[["objet", "datePublicationDonnees", "montant", "id"]] # "datePublicationDonnees",
    # On regroupe selon l objet du marché. Attention, objet n est pas forcément unique mais idMarche ne l'est pas non plus.
    df_group = pd.DataFrame(df_intermediaire.groupby(["objet",
                                                      "datePublicationDonnees", "montant"])["id"])
    # Initialisation du resultat sous forme de liste
    # df_group a un multi_index objet-datePublicationDonnees
    index = df_group.index
    for i in range(len(df_group)):
        # dataframe contenant les id d'un meme marche
        ids_to_modify = df_group[1].iloc[i]
        # Contient les index des lignes d'un meme marché. Utile pour le update
        new_index = list(ids_to_modify.index)
        # Création du dataframe avec id en seule colonne et comme index les index dans le df initial
        df_avec_bon_id = pd.DataFrame(len(new_index)*[max(ids_to_modify)], index=new_index, columns = ["id"])
        # Création d'un dataframe intermédiaire avec comme colonne nombreTitulaireSurMarchePresume
        df_nbtitulaires = pd.DataFrame(len(new_index)*[len(new_index)], index=new_index, columns = ["nombreTitulaireSurMarchePresume"])
        df_to_update = pd.concat([df_to_update, df_avec_bon_id])
        # Dataframe permettant de faire la jointure pour ajouter la colonne nombreTitulaireSurMarchePresume dans le df initial
        df_titulaires = pd.concat([df_titulaires, df_nbtitulaires])
        # Arbitraire, faire un update a chaque étape rallonge le temps d'execution, un update final idem.
        if len(df_to_update) > 50000:
            df.update(df_to_update)
            df_to_update = pd.DataFrame()
    # Ajout effectif de nombreTitulaireSurMarchePresume
    df = df.merge(df_titulaires, how='left', left_index=True, right_index=True)
    df.update(df_to_update)
    return df


# Partie pour la Review
## Objectif: Récupérer un dictionnaire contenant le passage colonne actuel / colonne subissant la modification
## Utilisation de data: l'objet json.
def indice_marche_avec_modification(data):
    """ Renvoie la liste des indices des marchés contenant une modification """
    liste_indices = []
    for i in range(len(data["marches"])):
        data["marches"][i]["id_technique"] = i  # Ajout d'un identifiant technique -> Permet d'avoir une colonne id unique par marché
        if data["marches"][i]["modifications"]:
            liste_indices += [i]
    return liste_indices


def recuperation_colonne_a_modifier(data, liste_indices):
    """ Renvoie les noms des differentes colonnes recevant une modification 
    sous la forme d'un dictionnaire: {Nom_avec_modification: Nom_sans_modification} """
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

## Objectif Numéro 2: Ajouter les modifications présentes dans le json à la clef modification 

def prise_en_compte_modifications(df, col_to_normalize='modifications'):
    """La fonction json_normalize de pandas ne permet pas de spliter la clef modifications automatiquement. 
    Cette fonction permet de le faire
    En entrée : La sortie json_normalize de pandas. (avec une colonne modifications)
    Le dataframe en entrée est directement modifié dans la fonction."""
    # Check colonne modifications. 
    if col_to_normalize not in df.columns:
        raise ValueError("Il n'y a aucune colonne du nom de {} dans le dataframe entrée en paramètre".format(col_to_normalize))
    to_normalize = df[col_to_normalize] #Récupération de la colonne à splitter
    df["booleanModification"] = 0
    for i in range(len(to_normalize)):
        json_modification = to_normalize[i]
        if json_modification != []: # dans le cas ou des modifications ont été apportées
            # col_to_add = list(json_modification[0].keys()) #Récupération des noms de colonnes
            for col in json_modification[0].keys():
                col_init = col
                # Formatage du nom de la colonne
                if "Modification" not in col:
                        col += "Modification"
                if col not in df.columns: # Cas ou on tombe sur le premier marche qui modifie un champ
                    df[col] = "" # Initialisation dans le df initial
                df[col][i] = json_modification[0][col_init]
                df["booleanModification"][i] = 1 #Création d'une booléenne pour simplifier le subset pour la suite

## Objectif numéro 3: Définition des fonctions nécéssaire pour l'identification des marchés/mis à jour de marchés

def split_dataframe(df, sub_data, modalite):
    """Définition de deux dataFrame.
        - Le premier qui contiendra uniquement les lignes avec modification, pour le marché ayant pour objet modalite
        - Le second contiendra l'ensemble des lignes correspondant au marché isolé dans le df1 qui ont pour objet modalite
        
        :param df: la source totale des données
        :param sub_data: le sous-ensemble correspondant à l'ensemble des marchés avec une modification
        :param modalite: la modalité sur laquelle on veut filtrer

        :return: un tuple (premier_df, second_df)
    """
    #Premier df: Contenant les lignes d'un marche avec des colonnes modifications non vide
    marche = sub_data[sub_data.objet == modalite]
    marche = marche.sort_values(by = 'id')
    #Second dataframe: Dans le df complet, récupération des lignes correspondant au marché récupéré
    date = marche.datePublicationDonnees.iloc[0]
    # A concaténer ? 
    marche_init = df[df.objet == modalite]
    marche_init = marche_init[marche_init.datePublicationDonnees == date]
    return (marche, marche_init)
    

# Partie fusion des datas modifiées
def fusion_source_modification(raw, df_source, col_modification, dict_modification):
    """ Permet de fusionner les colonnes xxxModification et sa colonne.
    raw correspond à une ligne du df_source
    Modifie le df_source
    """
    for col in col_modification:
        col_init = dict_modification[col]
        if raw[col] != '':
            df_source[col_init].loc[raw.name] = raw[col]
    return df_source

#Fonction Finale

def regroupement_marche(df, dict_modification):
    """Permet de recoder la variable identifiant. 
    Actuellement: 1 identifiant par déclaration (marché initial / modification sur marché/ un marché peut être déclaré plusieurs fois en fonction du nombre d'entreprise)
    En sortie: idMarche correspondra à un identifiant unique pour toutes les lignes composants un marché SI il a eu une modification
    Modification inplace du df source
    """
    df["idtech"] = ""
    #col_modification = list(dict_modification.keys())
    subdata_modif = df[df.booleanModification == 1] # Tout les marchés avec les modifications
    liste_objet = list(subdata_modif.objet.unique())
    df_to_concatene = pd.DataFrame() #df vide pour la concaténation
    logger.info("Au total, {} marchés sont concernés par au moins une modification".format(len(liste_objet)))
    for objet_marche in liste_objet:
        #Récupération du dataframe modification et du dataframe source
        marche, marche_init = split_dataframe(df, subdata_modif, objet_marche)
        for j in range(len(marche)):
            marche_init = fusion_source_modification(marche.iloc[j], marche_init, dict_modification.keys(), dict_modification)
        marche_init["idtech"] = marche.iloc[-1].id_technique
        df_to_concatene = pd.concat([df_to_concatene, marche_init], copy = False)
    df.update(df_to_concatene)
    #Attention aux id. 
    df.rename(columns = {"id": "id_source"}, inplace = True)
    df["id"] = np.where(df.idtech != "", df.idtech, df.id_technique)
    return df

def manage_modifications(data):
    """Conversion du json en pandas et incorporation des modifications"""
    L_indice = indice_marche_avec_modification(data)
    dict_modification = recuperation_colonne_a_modifier(data, L_indice)
    df = json_normalize(data['marches'])
    df = df.astype(conf_glob["nettoyage"]['type_col_nettoyage'], copy=False)
    prise_en_compte_modifications(df)
    df = regroupement_marche(df, dict_modification)
    return df
# Fin de la Review


if __name__ == "__main__":
    main()
