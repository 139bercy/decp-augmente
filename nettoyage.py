import json
import os
import numpy as np
import pandas as pd
import pickle

from pandas import json_normalize


def main():
    with open("config.json") as f:
        conf = json.load(f)
    path_to_data = conf["path_to_data"]
    decp_file_name = conf["decp_file_name"]
    error_siret_file_name = conf["error_siret_file_name"]

    with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
        data = json.load(json_data)

    # Applatit les données json en tableau
    df = json_normalize(data['marches'])

    df = manage_titulaires(df)

    df = drop_duplicates(df)

    df = manage_montant(df)

    df = manage_missing_code(df)

    df = manage_region(df)

    df = manage_date(df)

    #df = data_inputation(df)

    df = correct_date(df)

    with open('df_nettoye', 'wb') as df_nettoye:
        pickle.dump(df, df_nettoye)

    df.to_csv("decp_nettoye.csv")
    #df = apply_luhn(df)


def manage_titulaires(df):

    # Gestion différences concessionnaires / titulaires
    df.titulaires = np.where(df["titulaires"].isnull(), df.concessionnaires, df.titulaires)
    df.montant = np.where(df["montant"].isnull(), df.valeurGlobale, df.montant)
    df['acheteur.id'] = np.where(df['acheteur.id'].isnull(), df['autoriteConcedante.id'], df['acheteur.id'])
    df['acheteur.nom'] = np.where(df['acheteur.nom'].isnull(), df['autoriteConcedante.nom'], df['acheteur.nom'])
    donneesInutiles = ['dateSignature', 'dateDebutExecution',  'valeurGlobale', 'donneesExecution', 'concessionnaires',
                       'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom']
    df.drop(columns=donneesInutiles, inplace=True)

    # Récupération des données titulaires
    df["titulaires"].fillna('0', inplace=True)
    dfO = df[df['titulaires'] == '0']
    df = df[df['titulaires'] != '0']

    # Création d'une colonne nbTitulairesSurCeMarche
    df.loc[:, "nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x : len(x))

    df_titulaires = pd.concat([pd.DataFrame.from_records(x) for x in df['titulaires']],
                              keys=df.index).reset_index(level=1, drop=True)
    df_titulaires.rename(columns={"id":  "idTitulaires"}, inplace=True)
    df = df.drop('titulaires', axis=1).join(df_titulaires).reset_index(drop=True)

    return df


def drop_duplicates(df):
    df.drop_duplicates(subset=['source', '_type', 'nature', 'procedure', 'dureeMois',
                               'datePublicationDonnees', 'lieuExecution.code', 'lieuExecution.typeCode',
                               'lieuExecution.nom', 'id', 'objet', 'codeCPV', 'dateNotification', 'montant',
                               'formePrix', 'acheteur.id', 'acheteur.nom', 'typeIdentifiant', 'idTitulaires',
                               'denominationSociale'],
                       keep='first',
                       inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Correction afin que ces variables soient représentées pareil
    df['formePrix'] = np.where(df['formePrix'] == 'Ferme, actualisable', 'Ferme et actualisable', df['formePrix'])
    df['procedure'] = np.where(df['procedure'] == 'Appel d’offres restreint', "Appel d'offres restreint", df['procedure'])

    return df


def manage_montant(df):
    ################### Identifier les outliers - travail sur les montants
    df["montant"] = pd.to_numeric(df["montant"])
    df['montantOriginal'] = df["montant"]

    montant_borne_inf = 200.0
    montant_borne_sup = 9.99e8
    df['montant'] = np.where(df['montant'] <= montant_borne_inf, 0, df['montant'])
    df['montant'] = np.where(df['montant'] >= montant_borne_sup, 0, df['montant'])

    # On applique au df la division
    df["montant"] = df["montant"] / df["nbTitulairesSurCeMarche"]

    # Nettoyage colonnes
    df['montant'] = np.where(df['montant'] == 0, np.NaN, df['montant'])

    # Colonne supplémentaire pour indiquer si la valeur est estimée ou non
    df['montantEstime'] = np.where(df['montant'].isnull(), 'Oui', 'Non')

    return df


def manage_missing_code(df):
    df.id = np.where(df["id"].isnull(), '0000000000000000', df.id)
    df.codeCPV = np.where(df["codeCPV"].isnull(), '00000000', df.codeCPV)

    # Nettoyage des codes idTitulaires
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

    ########  Récupération code NIC
    df.idTitulaires = df.idTitulaires.astype(str)
    df['nic'] = df["idTitulaires"].str[-5:]

    #df.loc[~df["nic"].str.isdigit()] = np.NaN
    df.nic = np.where(~df["nic"].str.isdigit(), np.NaN, df.nic)
    df['nic'] = df.nic.astype(str)

    ######## Gestion code CPV
    df.codeCPV = df.codeCPV.astype(str)
    df["CPV_min"] = df["codeCPV"].str[:2]
    df["CPV_min_label"] = "Fourniture"
    if ( df["CPV_min"] == '45') : 
        df["CPV_min_label"] = "Travaux"
    else if ( df["CPV_min"] > '45') : 
        df["CPV_min_label"] = "Service"

    # Mise en forme des données vides
    df.denominationSociale = np.where(
        (df.denominationSociale == 'N/A') | (df.denominationSociale == 'null'),
        np.NaN, df.denominationSociale)

    return df


def manage_region(df):
    ################### Régions / Départements ##################
    # Création de la colonne pour distinguer les départements
    df['codePostal'] = df['lieuExecution.code'].str[:3]
    listCorrespondance = {
        '976': 'YT',
        '974': 'RE',
        '972': 'MQ',
        '971': 'GP',
        '973': 'GF'}
    df['codePostal'].replace(listCorrespondance, inplace=True)

    df['codePostal'] = df['codePostal'].str[:2]

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
    df['codePostal'].replace(listCorrespondance2, inplace=True)

    # Vérification si c'est bien un code postal
    listeCP = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '2A', '2B', '98', '976', '974', '972', '971', '973',
               '97', '988', '987', '984', '978', '975', '977', '986'] \
              + [str(i) for i in list(np.arange(10, 96, 1))]

    df['codePostal'] = np.where(~df['codePostal'].isin(listeCP), np.NaN, df['codePostal'])

    # Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)
    df['codePostal'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN, df['codePostal'])

    ###############################################################################
    # Création de la colonne pour distinguer les régions
    df['codeRegion'] = df['codePostal'].astype(str)
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

    df['codeRegion'].replace(listCorrespondanceI, inplace=True)

    # Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
    df['codeRegion'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'], df['codeRegion'])
    df['codeRegion'] = df['codeRegion'].astype(str)
    # Vérification des codes région
    listeReg = ['84', '27', '53', '24', '94', '44', '32', '11', '28', '75', '76',
                '52', '93', '01', '02', '03', '04', '06', '98']  # 98 = collectivité d'outre mer

    df['codeRegion'] = np.where(~df['codeRegion'].isin(listeReg), np.NaN, df['codeRegion'])

    # Identification du nom des régions
    df['Region'] = df['codeRegion'].astype(str)
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

    df['Region'].replace(correspondance_dict, inplace=True)
    # del chemin, chemindata, dfO, initial, key, listCorrespondance, listCorrespondanceI, string, value, word
    df['codePostal'] = df['codePostal'].astype(str)
    df['codeRegion'] = df['codeRegion'].astype(str)

    return df


def manage_date(df):
    ################### Date / Temps ##################
    # ..............Travail sur les variables de type date
    df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
    df.dateNotification = df.dateNotification.str[0:10]
    # On récupère l'année de notification
    df['anneeNotification'] = df.dateNotification.str[0:4]
    df['anneeNotification'] = df['anneeNotification'].astype(float)
    # On supprime les erreurs (0021 ou 2100 par exemple)
    df['dateNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['dateNotification'])
    df['dateNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['dateNotification'])
    df['anneeNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['anneeNotification'])
    df['anneeNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['anneeNotification'])
    df['anneeNotification'] = df.anneeNotification.astype(str).str[:4]

    # On récupère le mois de notification
    df['moisNotification'] = df.dateNotification.str[5:7]
    df.datePublicationDonnees = np.where(df.datePublicationDonnees == '', np.NaN, df.datePublicationDonnees)

    return df

def data_inputation(df):
    # Utilisation de la méthode 5 pour estimer les valeurs manquantes
    df['Region'] = df['Region'].astype(str)
    df['formePrix'] = df['formePrix'].astype(str)
    df['codeCPV'] = df['codeCPV'].astype(str)

    df['moisNotification'] = df['moisNotification'].astype(str)
    df['anneeNotification'] = df['anneeNotification'].astype(str)
    df['conca'] = df['formePrix'] + df['Region'] + df['codeCPV']

    # Calcul de la médiane par stratification
    medianeRegFP = pd.DataFrame(df.groupby('conca')['montant'].median())
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['conca', 'montantEstimation']
    df = pd.merge(df, medianeRegFP, on='conca')
    # Remplacement des valeurs manquantes par la médiane du groupe
    df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])
    del df['conca'], df['montantEstimation']

    # On recommence avec une plus petite stratification
    df['conca'] = df['formePrix'] + df['Region']
    df.reset_index(level=0, inplace=True)
    # Calcul de la médiane par stratification
    medianeRegFP = pd.DataFrame(df.groupby('conca')['montant'].median())
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['conca', 'montantEstimation']
    df = pd.merge(df, medianeRegFP, on='conca')
    # Remplacement des valeurs manquantes par la médiane du groupe
    df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])
    # S'il reste encore des valeurs nulles...
    df['montant'] = np.where(df['montant'].isnull(), df['montant'].median(), df['montant'])
    del df['conca'], df['montantEstimation'], df['index']
    del medianeRegFP

    # Colonne par marché
    df['montantTotalMarché'] = df["montant"] * df["nbTitulairesSurCeMarche"]

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

    df['dureeMoisEstime'] = np.where(mask, "Oui", "Non")

    # On corrige pour les colonnes considérées comme aberrantes, on divise par 30 (nombre de jours par mois)
    df['dureeMoisCalculee'] = np.where(mask, round(df['dureeMois'] / 30, 0), df['dureeMois'])
    # Comme certaines valeurs atteignent zero, on remplace par un mois
    df['dureeMoisCalculee'] = np.where(df['dureeMoisCalculee'] == 0, 1, df['dureeMoisCalculee'])

    # Au cas ils restent encore des données aberrantes, on les remplace par un mois -> A CHALLENGER
    df['dureeMoisCalculee'] = np.where( mask, 1, df.dureeMoisCalculee)

    return df

##### Algorithme de Luhn
def luhn(codeSIREN):
    try:
        chiffres = pd.DataFrame(map(int, list(str(codeSIREN))), columns=['siren'])
        chiffres['parite'] = [1, 2, 1, 2, 1, 2, 1, 2, 1]
        chiffres['multiplication'] = chiffres.siren * chiffres.parite
        for i in range(len(chiffres)):
            chiffres.multiplication[i] = sum([int(c) for c in str(chiffres.multiplication[i])])
        resultat = chiffres.multiplication.sum()
        if (resultat % 10) == 0:
            resultat = 0  # code BON
        else:
            resultat = 1  # code FAUX
    except:
        resultat = 1  # code FAUX
        pass
    return resultat


def apply_luhn(df):
    # Application sur les siren des acheteurs
    df['siren1Acheteur'] = df["acheteur.id"].str[:9]
    df_SA = pd.DataFrame(df['siren1Acheteur'])
    df_SA = df_SA.drop_duplicates(subset=['siren1Acheteur'], keep='first')
    df_SA['verifSirenAcheteur'] = df_SA['siren1Acheteur'].apply(luhn)

    # Application sur les siren des établissements
    df['siren2Etablissement'] = df.sirenEtablissement.str[:9]
    df_SE = pd.DataFrame(df['siren2Etablissement'])
    df_SE = df_SE.drop_duplicates(subset=['siren2Etablissement'], keep='first')
    df_SE['verifSirenEtablissement'] = df_SE['siren2Etablissement'].apply(luhn)

    # Merge avec le df principal
    df = pd.merge(df, df_SA, how='left', on='siren1Acheteur')
    df = pd.merge(df, df_SE, how='left', on='siren2Etablissement')
    del df['siren1Acheteur'], df['siren2Etablissement']

    # On rectifie pour les codes non-siret
    df.verifSirenEtablissement = np.where(
        (df.typeIdentifiantEtablissement != 'SIRET') | (df.typeIdentifiantEtablissement.isnull()), 0,
        df.verifSirenEtablissement)

    return df


if __name__ == "__main__":
    main()