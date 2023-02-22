import json
import os
import pickle
import logging.handlers
from random import random
import datetime
import numpy as np
import pandas as pd
import itertools
from pandas import json_normalize
import cProfile
import pstats
import utils

logger = logging.getLogger("main.nettoyage")
logger.setLevel(logging.DEBUG)
pd.options.mode.chained_assignment = None  # default='warn'

path_to_conf = "confs"
if not(os.path.exists(path_to_conf)): # Si le chemin confs n'existe pas (dans le cas de la CI et de Saagie)
    os.mkdir(path_to_conf)
if utils.USE_S3:
    res = utils.download_confs()
    if res :
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
path_to_data = conf_data["path_to_data"] # Ré écris

def main():

    if utils.USE_S3:
        if not(os.path.exists(path_to_data)): # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(path_to_data)
        utils.download_data_nettoyage()

    # Chargement du fichier flux
    logger.info("Récupération du flux")
    today = datetime.date.today()
    flux_file = "df_flux"
    flux_file = utils.retrieve_lastest(utils.s3.meta.client, flux_file)
    file_nettoye_today = "df_nettoye" + "-" + today.strftime("%Y-%m-%d") + ".pkl"
    if utils.USE_S3:
        logger.info(" Fichier Flux chargé depuis S3")
        df_flux = utils.get_object_content(flux_file)
    else : 
        print('Chargement en local')
        with open(flux_file, "rb") as flux_file:
            df_flux = pickle.load(flux_file)
    print('BUCKET visé : ', utils.BUCKET_NAME)
    # SI il n'y a pas d'ajout de données.
    if df_flux.empty :
        print('Flux vide')
        if utils.USE_S3:
            utils.write_object_file_on_s3(file_nettoye_today, df_flux)
        else : 
            with open('df_nettoye.pkl', 'wb') as df_nettoye:
                # Export présent pour faciliter l'utilisation du module enrichissement.py
                pickle.dump(df_flux, df_nettoye)
        logger.info("Flux vide")
        return df_flux

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
    df = manage_modifications(df_flux)
    df = manage_data_quality(df)
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
    print(df.columns)
    logger.info("Creation csv intermédiaire: decp_nettoye.csv")
    if utils.USE_S3 : 
        utils.write_object_file_on_s3(file_nettoye_today, df)
    else:
        with open(file_nettoye_today, 'wb') as df_nettoye:
            # Export présent pour faciliter l'utilisation du module enrichissement.py
            pickle.dump(df, df_nettoye)
    df.to_csv("df_nettoye" + "-" + today.strftime("%Y-%m-%d") + ".csv")
    logger.info("Ecriture du csv terminé")


def manage_data_quality(df : pd.DataFrame):
    """
    Cette fonction sépare en deux le dataframe d'entrée. Les données ne respectant pas les formats indiquées par les 
    règles de gestion de la DAJ sont mise de côtés. Les règles de gestions sont dans un mail du 15 février 2023.
    /!\ Dans les règles de gestion, certaines valeur de champ d'identification unique du marché ne sont pas accessibles 
    dans la donnée brut. On va donc ne traiter dans cette fonction que les variables accessibles de manières brutes
    et lorsque les règles portent sur des variables non brutzq on appliquera les règles à ce moment là. (ex : idtitulaire)
    /!\\
    
    Arguments
    ----------
    df :  le dataframe des données bruts.


    Return 
    -----------
    df (dataFrame) : le dataframe des données à enrichir.

    """
    




    pass

def check_reference_files():
    """
    Vérifie la présence des fichiers datas nécessaires, dans le dossier data.
        StockEtablissement_utf8.csv, cpv_2008_ver_2013.xlsx, geoflar-communes-2015.csv,
        departement2020.csv, region2020.csv, StockUniteLegale_utf8.csv
    """
    path_data = conf_data["path_to_data"]

    useless_keys = ["path_to_project", "path_to_data", "path_to_cache", "cache_bdd_insee",
                     "cache_not_in_bdd_insee",
                     "cache_bdd_legale",
                     "cache_not_in_bdd_legale", "cache_acheteur_bdd_legale", "cache_acheteur_not_in_bdd_legale"]

    path = os.path.join(os.getcwd(), path_data)
    for key in list(conf_data.keys()):
        if key not in useless_keys:
            logger.info(f'Test du fichier {conf_data[key]}')
            mask = os.path.exists(os.path.join(path, conf_data[key]))
            if not mask:
                logger.error(f"Le fichier {conf_data[key]} n'existe pas")
                raise ValueError(f"Le fichier data: {conf_data[key]} n'a pas été trouvé")


def found_values_in_dic(x, name:str):
    try:
        return x.get(name)
    except: # Parfois il arrive que x soit un Nan, mais c'est un cas assez rare d'où le try except.
        return None
def create_columns_titulaires_fast(df, column="titulaires"):
    """
    Explose le contenu du dataframe d'entrée à le colonne column puis créé une nouvelle colonne pour chaque clef explosée.
    Nécessite une unicité des index
    
    Arguments
    ---------
    df 
    column (strign) colonne dans laquelle se trouve l'objet à exploser pour créer les colonnes

    Returns
    --------
    Le même dataframe avec les informations extraites de la colonne column
    
    """
    df_explode = df[column].explode() # Very quick
    df_explode['typeIdentifiant'] = df_explode.apply(found_values_in_dic, args=(["typeIdentifiant"]))
    df_explode['id'] = df_explode.apply(found_values_in_dic, args=(["id"]))
    df_explode['denominationSociale'] = df_explode.apply(found_values_in_dic, args=(["denominationSociale"]))
    # On converti en dataframe pour faciliter le merge
    df_explode = pd.DataFrame(data={"typeIdentifiant" : df_explode["typeIdentifiant"], \
                                'idTitulaires' : df_explode['id'].iloc[:-1], \
                                'denominationSociale' : df_explode['denominationSociale'].iloc[:-2]})
    df = df.merge(df_explode, left_index=True, right_index=True)
    return df

def deal_with_many_titulaires(df_with_cotitulaires : pd.DataFrame, n_cotit=3):
    """
    Cette fonction renvoie un dictionnaire. Chaque élément du dictionnaire est un dataframe composé des informations du cotitulaires numéro n, n étant la clef du dictionnaire.
    Cette fonction peut être amélioré temporellement, le .apply(pd.series) est très long. Prendre exemple sur create_columns_titulaires_fast, même si c'est plus compliqué dû au doublon
    Peut être suffit il simplement de remonter le mask de duplicated.

    Arguments
    ----------
    df un Dataframe avec les marchés présentants plusieurs cotitulaires
    n_cotit (int) , nombe de cotitulaires dont il faut extraire les informations
    """
    df_with_cotitulaires_and_columns = df_with_cotitulaires['titulaires'].explode().apply(pd.Series) # On explose les lignes
    df_with_cotitulaires_and_columns['index'] = df_with_cotitulaires_and_columns.index # On créé la colonne index pour pouvoir ne récupérer que le premier 
    mask_duplicated = df_with_cotitulaires_and_columns.duplicated(subset=['index'], keep='first') # La première occurence est False, les autres sont True
    df_with_cotitulaires_titulaires = df_with_cotitulaires_and_columns.loc[~mask_duplicated, ["id", "denominationSociale", "typeIdentifiant"]]
    df_with_cotitulaires_titulaires.rename(columns={"id": "idTitulaires"}, inplace=True)
    dict_df_with_cotitulaires = {}
    dict_df_with_cotitulaires[0] = df_with_cotitulaires_titulaires
    # Récupérer les titulaires secondes et recommencer l'opération de dedoublonnage sur l'index pour cotitulaires 1, 2 et 3
    c_cotitulaires = 1
    while c_cotitulaires <=3: # 3 cotitulaires max : règle métier
        df_with_cotitulaires_and_columns = df_with_cotitulaires_and_columns.loc[mask_duplicated] # On ne récupère que les doublons sans la première occurence. Le cotitulaire 1 est alors le premier duplicata 
        mask_duplicated = df_with_cotitulaires_and_columns.duplicated(subset=['index'], keep='first') # La première occurence est False, les autres sont True
        df_with_cotitulaires_c = df_with_cotitulaires_and_columns.loc[~mask_duplicated, ["typeIdentifiant", "id", "denominationSociale"]]
        df_with_cotitulaires_c = df_with_cotitulaires_c.rename(columns={"id": f"id_cotitulaire{c_cotitulaires}",\
                                                                        "denominationSociale" : f"denominationSociale_cotitulaire{c_cotitulaires}",\
                                                                        "typeIdentifiant":f"typeIdentifiant_cotitulaire{c_cotitulaires}"})
        dict_df_with_cotitulaires[c_cotitulaires] = df_with_cotitulaires_c
        c_cotitulaires += 1

    return dict_df_with_cotitulaires



def manage_titulaires(df: pd.DataFrame):
    """
    Cette fonction gère les titulaires des marchés qu'ils soient uniques ou multiples. 
    D'un point de vue métier/data : Les titulaires d'un marchés sont sous formes de JSON (dictionnaire) dans la colonne titulaires. 
    L'immense majorité des JSON titulaires est ainsi : {'typeIndentifiant' : value, 'id': value, 'denominationSociale': value}
    On veut éclater cette colonne, c-à-d ne plus avoir une colonne avec un JSON mais créer de nouvelles colonnes avec les valeurs de ce JSON

    L'autre point de cette fonction est de gérer les marchés lorsqu'il y a plusieurs titulaires. Avant on créait des lignes pour chaque nouveau titulaire, maintenant
    on a des nvls colonnes pour les cotitulaires. On garde l'unicité de 1 ligne = 1 marché qui était perdu avant.
    """
    df = df[~(df['titulaires'].isna() & df['concessionnaires'].isna())]
    df.titulaires = np.where(df["titulaires"].isnull(), df.concessionnaires, df.titulaires)
    df.montant = np.where(df["montant"].isnull(), df.valeurGlobale, df.montant)
    df['acheteur.id'] = np.where(df['acheteur.id'].isnull(), df['autoriteConcedante.id'], df['acheteur.id'])
    df['acheteur.nom'] = np.where(df['acheteur.nom'].isnull(), df['autoriteConcedante.nom'], df['acheteur.nom'])
    useless_columns = ['dateSignature', 'dateDebutExecution', 'valeurGlobale', 'donneesExecution', 'concessionnaires',
                        'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom',
                        'idtech', "id_technique"]
    df.drop(columns=useless_columns, inplace=True)

    # Création d'une colonne nbTitulairesSurCeMarche.
    # Cette colonne sera retravaillé dans la fonction detection_accord_cadre
    df.loc[:, "nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x: len(x))

    # Gérer le cas pour un seul titulaires
    df_one_titulaires = df[df['nbTitulairesSurCeMarche']==1].copy()
    df_one_titulaires = create_columns_titulaires_fast(df_one_titulaires)
    df_one_titulaires.rename(columns={"id_y": "idTitulaires", "id_x": "id"}, inplace=True)

    # Dans le cas de plusieurs titulaires
    df_with_cotitulaires = df[df["nbTitulairesSurCeMarche"]>1].copy() # On ne garde que les dataframes avec des cotitulaires 
    dict_df_with_cotitulaires = deal_with_many_titulaires(df_with_cotitulaires) # Cette fonction peut être amélioré, cependant comme 
    df_cotitulaires = pd.concat([x for x in dict_df_with_cotitulaires.values()], axis=1) # On recolle les différents co titulaires d'un même marché

    df_with_cotitulaires = pd.concat([df_with_cotitulaires, df_cotitulaires], axis=1)

    ### Reconstruction du datframe final qui est l'aggrégration des df_one_titulaires et dfcotitulaires
    return pd.concat([df_one_titulaires, df_with_cotitulaires], axis=0)


def manage_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permet la suppression des eventuels doublons pour un subset précis du dataframe

    Retour:
        pd.DataFrame
    """
    logger.info(f"Taille dataframe avant   manage_duplicates{df.shape}")
    logger.info("Début du traitement: Suppression des doublons")
    nb_ligne_avant_suppression = len(df)
    df.sort_values(by="source", inplace=True) # Pourquoi ? La partie métier (Martin Douysset) a demandé à ce qu'en cas de doublon sur plusieurs sources, ceux de l'AIFE
    # (la première en ordre alphabéitque soit conservés).
    # Donc on sort by source et on drop duplicates en gardant les first.
    df.reset_index(drop=True, inplace=True)
    assert df.loc[0, "source"] == "data.gouv.fr_aife"
    df.drop_duplicates(subset=['_type', 'nature', 'procedure', 'dureeMois',
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

    # On ne veut plus convertir en int. Mais plutôt utiliser round. 
    df['montant'] = df['montant'].apply(lambda x:round(x) if str(x).isdigit() else x) # Pourquoi on n'utilise pas directement round de pandas ?
    # Car on ne gagne pas beaucoup en rapidité et la méthode pandas laisse le format float. Alors qu'on veut un display int.
    

    def detect_inexploitable(montant):
        """
        Cette fonction indique si un montant est exploitable ou non selon l'algorithme spécifié par la DAJ
        """
        montant_str = str(int(float(montant)))  # Valeur du int non comprise
        if float(montant) > 3000000000 or float(montant) < 1:
            return True # True car inexploitable
        elif montant_str.startswith('123456789'):
            return True
        threshold = len(montant_str)-2
        for i_caractere in range(len(montant_str)):
            begin_caract = montant_str[i_caractere]
            subset = montant_str[i_caractere:i_caractere+threshold]
            unique_caract_subset = ''.join(set(subset))
            if (len(unique_caract_subset)==1) and (begin_caract!='0'):
                return True 
        else:
            return False
    
    df['montant_inexploitable'] = df.montant.apply(lambda x:detect_inexploitable(x))
    

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
    # Ajout d'un replace(' ') car sinon on ne prenait pas correctement en compte tous les departements d'un format spécifique
    liste_cp = conf_glob["nettoyage"]["code_CP"].replace(' ', '').split(',') \
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
    departement['dep'] = departement['dep'].astype(str)
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
    liste_reg = conf_glob["nettoyage"]["code_reg"].replace(' ', '').split(',')  # 98 = collectivité d'outre mer
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
    logger.info(f"Taille dataframe avant manage_date {df.shape}")
    logger.info("Début du traitement: Récupération de l'année et du mois du marché public + "
                "Correction des années aberrantes")
    # Date / Temps #
    # ..............Travail sur les variables de type date
    df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
    df.dateNotification = df.dateNotification.str[0:10]
    # On récupère l'année de notification
    logger.info("Récupération de l'année")
    df['anneeNotification'] = df.dateNotification.str[0:4]
    mask_only_digits = df['anneeNotification'].apply(lambda x:str(x).isdigit())
    df['anneeNotification'] = np.where(mask_only_digits, df['anneeNotification'], 0) # Safe casting car parfois on a des formats lunaires.
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
    df['moisNotification'] = df.moisNotification.astype(str).str[:4]
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
    """Retourne le max parmis les élements de l'objet d'entrée, si il y a un Nan, retourne Nan 
    """
    if x.isna().any():
        value_number = 0  # Essentiel pour la construction de df_avec_valid_id. Sinon ça crash
    else:
        value_number = max(x)
    return value_number


def regroupement_marche_complet(df):
    """la colonne id n'est pas unique. Cette fonction permet de la rendre unique en regroupant
    les marchés en fonction de leur objets/date de publication des données et montant.
    Ajoute dans le meme temps la colonne nombreTitulaireSurMarchePresume"""
    logger.info(f"Taille dataframe avant regroupement_marche_complet {df.shape}")
    # On regroupe selon l objet du marché. Attention, objet pas forcément unique mais idMarche ne l'est pas non plus.
    df_group = pd.DataFrame(df[["objet", "datePublicationDonnees", "montant", "id"]] .groupby(["objet",
                                                    "datePublicationDonnees", "montant"])["id"])
    df_group['new_index'] = df_group[1].apply(lambda x:list(x.index))
    df_group["nbTit"] = df_group[1].apply(lambda x: len(x)) # On compte le nombre d'id dans chaque ligne df group et ainsi on a le nombre de titulaires
    df_group['valid_id'] = df_group[1].apply(create_value_number) # On récupère l'ID le plus haut parmis les doublons (au sens objet et datePublicationDonnees)
    flat_indx = list(itertools.chain(*df_group['new_index'].values))
    flat_nbtit = list(itertools.chain(*[[x]*x for x in df_group["nbTit"].values])) # On applati la liste, si on a 2 titulaires il faut donc avoir deux 2 à la suite dans la liste d'où le itertools.chain
    flat_id = list(itertools.chain(*[[x]*y for (x,y) in zip(df_group["valid_id"].values, df_group["nbTit"].values)]))
    dict_data = {"nombreTitulaireSurMarchePresume" : flat_nbtit, "id": flat_id}
    df_reconstruct = pd.DataFrame(index=flat_indx, data=dict_data)
    
    df_reconstruct['id'].replace(0, pd.NA, inplace=True) # On renomme remplace ici les 0 par des Nan, pas plus tôt pour deux raisons 
    # Lorsqu'on flat les listes les Nan ne sont pas itérables
    # Cependant on a bien besoin de forcer le typage en pd.NA pour la suite de la pipeline
    df["nombreTitulaireSurMarchePresume"] = 0 # Je créé la colonne dans df pour que les deux colonnes soit update avec la methode update()
    df.update(df_reconstruct)
    df['nombreTitulaireSurMarchePresume'].replace(0,np.nan, inplace=True) # Ajout artificiel pour se caler sur le format de la fonction regroupement_marche_complet() initiale

    return df


def recuperation_colonne_a_modifier() -> dict:
    """
    Renvoie les noms des differentes colonnes recevant une modification
    sous la forme d'un dictionnaire: {Nom_avec_modification: Nom_sans_modification}.
    Les colonnes qui sont concernés ont déjà été détectés dans gestion_flux.

    Retour:
        dict
    """
    colonne_to_modify = dict()
    dict_path = "columns_modifications.pkl"
    if utils.USE_S3:
        utils.download_file(dict_path, dict_path)
    # On récupère les colonnes détectés dans gestion_flux
    with open(dict_path, "rb") as file_modif:
        columns_modification = pickle.load(file_modif)
    for column in columns_modification:
        if "Modification" in column:
            key = column
            value = column.replace("Modification", "")
        else:
            key = column + "Modification"
            value = column
        colonne_to_modify[key] = value

    colonne_to_modify["objetModification"] = "objetModification" # Cette colonne est un cas particulier.
    print('Colonne de modification', colonne_to_modify)
    # On va utiliser cette fonction pour faire un mapping des noms issus des modifications avec les noms habituels.
    # Le mapping suivra la forme "xxxModification" : "xxx".
    # Sauf que "objet" concerne l'objet d'un marché, or "objetModification" l'objet de la modification.
    # Donc "objetModification" ne doit pas remplacer l'objet du marché.
    return colonne_to_modify

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
    
    mask_multiples_modifications = df.modifications.apply(lambda x:len(x)>1)
    df.loc[mask_multiples_modifications, col_to_normalize] = df.loc[mask_multiples_modifications, col_to_normalize].apply(concat_modifications)
    df["HowManyModification"] = df[col_to_normalize].apply(lambda x:len(x))
    df["booleanModification"] = df["HowManyModification"].apply(lambda x:1 if x>0 else 0)
    
    to_normalize = df[col_to_normalize]
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


def fusion_source_modification_whole_dataset(df_source : pd.DataFrame, dict_modification : dict):
    """
    Cette fonction met à jour les colonnes originales 

    """
    # Maintenant toutes les modifications sont uniques.
    for column_modif in dict_modification.keys():
        column_to_change = dict_modification[column_modif]# Les colonnes auxquelles il y a des modifications à apporter 
        # ont été construites ainsi nomcolonne+"Modification". 
        # Donc on retire Modificaiton pour pointer vers la bonne colonne                            
        mask_raw_to_change = df_source[column_modif].apply(lambda x:x!='').fillna(False)
        # Les valeurs None ne répondent pas au boolean. On les mets à False pour ne pas y toucher
        df_source.loc[mask_raw_to_change, column_to_change] = df_source.loc[mask_raw_to_change, column_modif]
    return df_source

def regroupement_marche(df: pd.DataFrame, dict_modification: dict) -> pd.DataFrame:
    """
    Permet de recoder la variable identifiant.
    Actuellement: 1 identifiant par déclaration (marché avec ou sans modification)
    Un marché peut être déclaré plusieurs fois en fonction du nombre d'entreprise. Si 2 entreprises sur
    En sortie: id correspondra à un identifiant unique pour toutes les lignes composants un marché SI il a eu une
    modification.
    Modification inplace du df source
    La variable idtech permet ici d'identifier les marchés qui sont en réalités des doublons causés par la gestion des modifications
    Ce n'est pas forcément une erreur de traitement, cela peut aussi être directement lié au fournisseur qui met une ligne pour chacune des nouvelles modifs.


    Retour:
        pd.DataFrame
    """
    df["idtech"] = ""
    subdata_modif = df[df.booleanModification == 1]  # Tout les marchés avec les modifications
    liste_objet = list(subdata_modif.objet.unique())
    marches_init = []
    for objet_marche in liste_objet: # C'est du dedoublonnage en fait ça
        # Récupération du dataframe modification et du dataframe source
        # On créée la colonne "idtech"
        marche, marche_init = split_dataframe(df, subdata_modif, objet_marche)
        marche_init["idtech"] = marche.iloc[-1].id_technique
        marches_init.append(marche_init)
    if marches_init :  # Si il y a des modifications on les gère, sinon on retourne le df tel qu'il est entré dans la fonction
        df_to_concatene = pd.concat([x for x in marches_init], copy=False)
        df.update(df_to_concatene)
        df["idMarche"] = np.where(df.idtech != "", df.idtech, df.id_technique)
        df = fusion_source_modification_whole_dataset(df, dict_modification)
    else: # Pour que la colonne idMarche existe quand même.
        df["idMarche"] = np.where(df.idtech != "", df.idtech, df.id_technique) 
    return df


def manage_modifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion du json en pandas et incorporation des modifications

    Retour:
        pd.DataFrame
    """
    logger.info(f"Taille dataframe avant manage_modifications {df.shape}")
    dict_modification = recuperation_colonne_a_modifier()

    df = df.astype(conf_glob["nettoyage"]['type_col_nettoyage'], copy=False, errors='ignore')
    # Création d'un id technique qui existait dans les versions précédentes. Pour que chaque marché ait un id unique.
    df["id_technique"] = df.index
    prise_en_compte_modifications(df)
    # Safe dict_modification 
    cols_df = df.columns.tolist()
    cols_to_del = []
    for col in dict_modification.keys():
        if col not in cols_df:
            print(col)
            cols_to_del.append(col)
    for col in cols_to_del:
        dict_modification.pop(col, "None")
    df = regroupement_marche(df, dict_modification)
    return df


if __name__ == "__main__":
    if conf_debug["debug"]:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        with open('df_nettoye_new_regroupement_marche', 'rb') as df_nettoye: # Forcément du local, pas besoin de gérer ça sur S3
            df = pickle.load(df_nettoye)
            init_len = len(df)
        with open(f"profilingSnettoyage_opti_size{init_len}.txt", "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats('ncalls')
            ps.sort_stats('cumulative')
            ps.print_stats()
    else:
        main()
