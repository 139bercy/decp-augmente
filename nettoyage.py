import json
import os
import pickle
import logging
import logging.handlers
import numpy as np
import pandas as pd
from pandas import json_normalize

logger = logging.getLogger("main.nettoyage")
logger.setLevel(logging.DEBUG)


def main():
    with open("config.json") as f:
        conf = json.load(f)
    check_reference_files(conf)
    path_to_data = conf["path_to_data"]
    decp_file_name = conf["decp_file_name"]
    # error_siret_file_name = conf["error_siret_file_name"]
    logger.info("Ouverture du fichier decp.json")
    with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
        data = json.load(json_data)

    # Applatit les données json en tableau
    # pd.read_json(data['marches'])
    # df = pd.read_json(json.dumps(data['marches']))
    df = json_normalize(data['marches'])

    df = df.astype({
        'id': 'string',
        'source': 'string',
        'uid': 'string',
        'uuid': 'string',
        '_type': 'string',
        'objet': 'string',
        'codeCPV': 'string',
        'lieuExecution.code': 'string',
        'lieuExecution.typeCode': 'string',
        'lieuExecution.nom': 'string',
        'dureeMois': 'int64',
        # TODO définition d'un format de date "généraliste" qui sera capable de gérer les différents formats présents
        # 'dateNotification': 'datetime64[ns]',
        # 'datePublicationDonnees': 'datetime64[ns]',
        'montant': 'float64',
        'formePrix': 'string',
        'titulaires': 'object',
        # TODO doit-on normaliser aussi ce champ (qui ne l'est pas automatiquement)
        # 'titulaires.typeIdentifiant': 'string',
        # 'titulaires.id': 'string',
        # 'titulaires.denominationSociale': 'string',
        'modifications': 'object',
        # TODO doit-on normaliser aussi ce champ (qui ne l'est pas automatiquement)
        # Risque de duplication car c'est un tableau
        # 'modifications.objetModification': 'string',
        # 'modifications.dateNotificationModification': 'string',
        # 'modifications.datePublicationDonneesModification': 'string',
        # 'modifications.montant': 'float64',
        'nature': 'string',
        'autoriteConcedante.id': 'string',
        'autoriteConcedante.nom': 'string',
        'acheteur.id': 'string',
        'acheteur.nom': 'string',
        'donneesExecution': 'string'
        # TODO doit-on normaliser aussi ce champ (qui ne l'est pas automatiquement)
        # Risque de duplication car c'est un tableau
        # 'donneesExecution.datePublicationDonneesExecution': 'datetime64[ns]',
        # 'donneesExecution.depensesInvestissement': 'datetime64[ns]',
        # 'donneesExecution.depensesInvestissement.tarifs': 'object',
        # 'donneesExecution.depensesInvestissement.tarifs.intituleTarif': 'string',
        # 'donneesExecution.depensesInvestissement.tarifs.tarif': 'float64',
    }, copy=False)
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
        pickle.dump(df, df_nettoye)
    df.to_csv("decp_nettoye.csv")
    logger.info("Ecriture du csv terminé")


def check_reference_files(conf):
    """Vérifie la présence des fichiers datas nécessaires, dans le dossier data.
        StockEtablissement_utf8.csv, cpv_2008_ver_2013.xlsx, "geoflar-communes-2015.csv", departement2020.csv, region2020.csv, StockUniteLegale_utf8.csv"""
    path_to_data = conf["path_to_data"]
    L_key_useless = ["path_to_project", "path_to_data"]
    path = os.path.join(os.getcwd(), path_to_data)
    for key in list(conf.keys()):
        if key not in L_key_useless:
            logger.info('Test du fichier {}'.format(conf[key]))
            mask = os.path.exists(os.path.join(path, conf[key]))
            if not mask:
                logger.error("Le fichier {} n'existe pas".format(conf[key]))
                raise ValueError("Le fichier data: {} n'a pas été trouvé".format(conf[key]))


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
                       'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom']
    df.drop(columns=donneesInutiles, inplace=True)

    # Récupération des données titulaires
    df["titulaires"].fillna('0', inplace=True)
    df = df[df['titulaires'] != '0']

    # Création d'une colonne nbTitulairesSurCeMarche
    df.loc[:, "nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x: len(x))

    df_titulaires = pd.concat([pd.DataFrame.from_records(x) for x in df['titulaires']],
                              keys=df.index).reset_index(level=1, drop=True)
    df_titulaires.rename(columns={"id": "idTitulaires"}, inplace=True)
    df = df.drop('titulaires', axis=1).join(df_titulaires).reset_index(drop=True)

    return df


def manage_duplicates(df):
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
    logger.info("Nombre de ligne doublons supprimées: {}".format(nb_ligne_avant_suppression - nb_ligne_apres_suppresion))

    # Correction afin que ces variables soient représentées pareil
    df['formePrix'] = np.where(df['formePrix'].isna(), np.nan, df['formePrix'])
    df['formePrix'] = np.where('Ferme, actualisable' == df['formePrix'], 'Ferme et actualisable', df['formePrix'])
    df['procedure'] = np.where('Appel d’offres restreint' == df['procedure'], "Appel d'offres restreint", df['procedure'])

    # détection des accords cadres avec de multiples titulaires

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
    df['montant'] = np.where(df['montant'] <= borne_inf, 0, df['montant'])
    logger.info("{} montant(s) étaient inférieurs à la borne inf {}".format(df.montant.value_counts()[0] - nb_montant_egal_zero, borne_inf))
    nb_montant_egal_zero = df.montant.value_counts()[0]
    df['montant'] = np.where(df['montant'] >= borne_sup, 0, df['montant'])
    logger.info("{} montant(s) étaient supérieurs à la borne sup: {}".format(df.montant.value_counts()[0] - nb_montant_egal_zero, borne_sup))

    # Colonne supplémentaire pour indiquer si la valeur est estimée ou non
    df['montantEstime'] = np.where(df['montant'] == 0, 'True', 'False')
    # Ecriture dans la log
    logger.info("Au total, {} montant(s) ont été corrigé (on compte aussi les montants vides).".format(df.montant.value_counts()[0]))
    return df


def manage_missing_code(df):
    # Ecriture dans les logs
    logger.info("Nombre d'identifiant manquants et remplacés: {}".format(sum(df["id"].isnull())))
    logger.info("Nombre de code CPV manquants et remplacés: {}".format(sum(df["codeCPV"].isnull())))

    df.id = np.where(df["id"].isnull(), '0000000000000000', df.id)
    df.codeCPV = np.where(df["codeCPV"].isnull(), '00000000', df.codeCPV)

    # Nettoyage des codes idTitulaires
    logger.info("Nettoyage des idTitualires")
    caracteres_speciaux_dict = {
        "\\t": "",
        "-": "",
        " ": "",
        ".": "",
        "?": "",
        "    ": ""}
    mask = (df.typeIdentifiant == 'SIRET') | \
           (df.typeIdentifiant.isnull()) | \
           (df.typeIdentifiant == 'nan')
    df.idTitulaires[mask].replace(caracteres_speciaux_dict, inplace=True)
    df.idTitulaires = np.where(df.idTitulaires == '', np.NaN, df.idTitulaires)
    # Ecriture dans les logs
    logger.info("Nombre d'identifiant titualire ou un traitement sur les caractères spéciaux a été fait: {}".format(sum(mask)))

    # Récupération code NIC
    logger.info("Récupération du code NIC")
    df.idTitulaires = df.idTitulaires.astype(str)
    df['nic'] = df["idTitulaires"].str[-5:]

    # df.loc[~df["nic"].str.isdigit()] = np.NaN
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
    # Régions / Départements #
    # Création de la colonne pour distinguer les départements
    logger.info("Création de la colonne département Execution")
    df['codeDepartementExecution'] = df['lieuExecution.code'].str[:3]
    listCorrespondance = {
        '976': 'YT',
        '974': 'RE',
        '972': 'MQ',
        '971': 'GP',
        '973': 'GF'}
    df['codeDepartementExecution'].replace(listCorrespondance, inplace=True)

    df['codeDepartementExecution'] = df['codeDepartementExecution'].str[:2]

    listCorrespondance2 = {
        'YT': '976',
        'RE': '974',
        'MQ': '972',
        'GP': '971',
        'GF': '973',
        'TF': '98',
        'NC': '988',
        'PF': '987',
        'WF': '986',
        'MF': '978',
        'PM': '975',
        'BL': '977'}
    df['codeDepartementExecution'].replace(listCorrespondance2, inplace=True)

    # Vérification si c'est bien un code postal
    listeCP = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '2A', '2B', '98', '976', '974', '972', '971', '973',
               '97', '988', '987', '984', '978', '975', '977', '986'] \
        + [str(i) for i in list(np.arange(10, 96, 1))]

    df['codeDepartementExecution'] = np.where(~df['codeDepartementExecution'].isin(listeCP), np.NaN, df['codeDepartementExecution'])

    # Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)

    df['lieuExecution.typeCode'] = np.where(df['lieuExecution.typeCode'].isna(), np.NaN, df['lieuExecution.typeCode'])
    df['codeDepartementExecution'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN, df['codeDepartementExecution'])

    ###############################################################################
    # Création de la colonne pour distinguer les régions
    logger.info("Création de la colonne contenant les codes des regions d'execution")
    df['codeRegionExecution'] = df['codeDepartementExecution'].astype(str)
    # Définition des codes des régions en fonctions des codes de départements
    listCorrespondance = {
        '84': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
        '27': ['21', '25', '39', '58', '70', '71', '89', '90'],
        '53': ['35', '22', '56', '29'],
        '24': ['18', '28', '36', '37', '41', '45'],
        '94': ['2A', '2B', '20'],
        '44': ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
        '32': ['02', '59', '60', '62', '80'],
        '11': ['75', '77', '78', '91', '92', '93', '94', '95'],
        '28': ['14', '27', '50', '61', '76'],
        '75': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
        '76': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
        '52': ['44', '49', '53', '72', '85'],
        '93': ['04', '05', '06', '13', '83', '84'],
        '06': ['976'],
        '04': ['974'],
        '02': ['972'],
        '01': ['971'],
        '03': ['973'],
        '98': ['97', '98', '988', '986', '984', '987', '975', '977', '978']
    }

    # Inversion du dict
    listCorrespondanceI = {value: str(key) for key, values in listCorrespondance.items() for value in values}

    df['codeRegionExecution'].replace(listCorrespondanceI, inplace=True)

    # Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
    df['codeRegionExecution'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'], df['codeRegionExecution'])
    df['codeRegionExecution'] = df['codeRegionExecution'].astype(str)
    # Vérification des codes région
    listeReg = ['84', '27', '53', '24', '94', '44', '32', '11', '28', '75', '76',
                '52', '93', '01', '02', '03', '04', '06', '98']  # 98 = collectivité d'outre mer

    df['codeRegionExecution'] = np.where(~df['codeRegionExecution'].isin(listeReg), np.NaN, df['codeRegionExecution'])

    # Identification du nom des régions
    logger.info("Creation de la colonne contenant le libbelé des regions d'execution")
    df['libelleRegionExecution'] = df['codeRegionExecution'].astype(str)
    correspondance_dict = {
        '84': 'Auvergne-Rhône-Alpes',
        '27': 'Bourgogne-Franche-Comté',
        '53': 'Bretagne',
        '24': 'Centre-Val de Loire',
        '94': 'Corse',
        '44': 'Grand Est',
        '32': 'Hauts-de-France',
        '11': 'Île-de-France',
        '28': 'Normandie',
        '75': 'Nouvelle-Aquitaine',
        '76': 'Occitanie',
        '52': 'Pays de la Loire',
        '93': 'Provence-Alpes-Côte d\'Azur',
        '01': 'Guadeloupe',
        '02': 'Martinique',
        '03': 'Guyane',
        '04': 'La Réunion',
        '06': 'Mayotte',
        '98': 'Collectivité d\'outre mer'}

    df['libelleRegionExecution'].replace(correspondance_dict, inplace=True)
    # del chemin, chemindata, dfO, initial, key, listCorrespondance, listCorrespondanceI, string, value, word
    df['codeDepartementExecution'] = df['codeDepartementExecution'].astype(str)
    df['codeRegionExecution'] = df['codeRegionExecution'].astype(str)

    return df


def manage_date(df):
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
    logger.info("Au total, {} marchés n'ont pas de date de publication des données connue".format(sum(df["datePublicationDonnees"].isna())))

    return df


def correct_date(df):
    # On cherche les éventuelles erreurs mois -> jours
    mask = ((df['montant'] == df['dureeMois'])
            | (df['montant'] / df['dureeMois'] < 100)
            | (df['montant'] / df['dureeMois'] < 1000) & (df['dureeMois'] >= 12)
            | ((df['dureeMois'] == 30) & (df['montant'] < 200000))
            | ((df['dureeMois'] == 31) & (df['montant'] < 200000))
            | ((df['dureeMois'] == 360) & (df['montant'] < 10000000))
            | ((df['dureeMois'] == 365) & (df['montant'] < 10000000))
            | ((df['dureeMois'] == 366) & (df['montant'] < 10000000))
            | ((df['dureeMois'] > 120) & (df['montant'] < 2000000)))

    df['dureeMoisEstimee'] = np.where(mask, "True", "False")

    # On corrige pour les colonnes considérées comme aberrantes, on divise par 30 (nombre de jours par mois)
    df['dureeMoisCalculee'] = np.where(mask, round(df['dureeMois'] / 30, 0), df['dureeMois'])
    # Comme certaines valeurs atteignent zero, on remplace par un mois
    df['dureeMoisCalculee'] = np.where(df['dureeMoisCalculee'] <= 0, 1, df['dureeMoisCalculee'])  # Il y a une valeur négative
    logger.info("Au total, {} duree de marché en mois ont été jugées rentrées en jour et non en mois.".format(sum(mask)))

    return df


def data_inputation(df):
    """Permet une estimation de la dureeMois (pour les durees évidemment fausses) grace au contenu de la commande (codeCPV)"""
    df_intermediaire = df[["objet", "dureeMois", "dureeMoisEstimee", "dureeMoisCalculee", "CPV_min", "montant"]]
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
    """Fonction pour mettre en qualité le champ objetMarche"""
    # Remplacement brutal du caractère (?) par un espace
    df['objet'] = df['objet'].str.replace('�', 'XXXXX')

    return df


if __name__ == "__main__":
    main()
