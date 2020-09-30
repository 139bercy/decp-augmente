# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 
@author: Lucas GEFFARD
Spyder : 3.3.6
Python : 3.7
"""

######################################################################
######################################################################
# I. IMPORTATION ET MISE EN FORME DES DONNEES
######################################################################
######################################################################

######################### Importation des librairies ##########################
import pandas as pd
from pandas import json_normalize
import numpy as np
import json
import os
import time
import pickle
from tqdm import tqdm 

from lxml import html
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
import urllib

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
from folium import plugins

######################################################################

#import warnings
#warnings.filterwarnings("ignore")
######################################################################

siren_len = 9

#Chargement des données
with open("config.json") as f:
    conf = json.load(f)

path_to_data = conf["path_to_data"]
decp_file_name = conf["decp_file_name"]
error_siret_file_name = conf["error_siret_file_name"]


with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
    data = json.load(json_data)
df = json_normalize(data['marches']) #Aplatir les données Json imbriquées

#Gestion différences concessionnaires / titulaires
df.titulaires = np.where(df.titulaires.isnull(), df.concessionnaires, df.titulaires)
df.montant = np.where(df.montant.isnull(), df.valeurGlobale, df.montant)
df['acheteur.id'] = np.where(df['acheteur.id'].isnull(), df['autoriteConcedante.id'], df['acheteur.id'])
df['acheteur.nom'] = np.where(df['acheteur.nom'].isnull(), df['autoriteConcedante.nom'], df['acheteur.nom'])
donneesInutiles = ['dateSignature', 'dateDebutExecution',  'valeurGlobale', 'donneesExecution', 'concessionnaires', 
                   'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom']
df = df.drop(columns=donneesInutiles)

#Récupération des données titulaires    
df.titulaires.fillna('0', inplace=True)
dfO = df[df['titulaires'] == '0']
df = df[df['titulaires'] != '0']

# Création d'une colonne nbTitulairesSurCeMarche
df["nbTitulairesSurCeMarche"] = df['titulaires'].apply(lambda x : len(x))


df_titulaires = pd.concat([pd.DataFrame.from_records(x) for x in df['titulaires']], keys=df.index).reset_index(level=1,drop=True)
df_titulaires.rename(columns={"id":  "idTitulaires"}, inplace=True)
df = df.drop('titulaires', axis=1).join(df_titulaires).reset_index(drop=True)

del df_titulaires, donneesInutiles

######################################################################
#...............    Nettoyage/formatage des données
################### Identifier et supprimer les doublons
df = df.drop_duplicates(subset=['source', '_type', 'nature', 'procedure', 'dureeMois',
                           'datePublicationDonnees', 'lieuExecution.code', 'lieuExecution.typeCode',
                           'lieuExecution.nom', 'id', 'objet', 'codeCPV', 'dateNotification', 'montant', 
                           'formePrix', 'acheteur.id', 'acheteur.nom', 'typeIdentifiant', 'idTitulaires',
                           'denominationSociale'], keep='first')
df.reset_index(inplace=True, drop = True)

# Correction afin que ces variables soient représentées pareil    
df['formePrix'] = np.where(df['formePrix'] == 'Ferme, actualisable', 'Ferme et actualisable', df['formePrix'])
df['procedure'] = np.where(df['procedure'] == 'Appel d’offres restreint', "Appel d'offres restreint", df['procedure'])


######################################################################
######################################################################
# II. VALORISATION
######################################################################
######################################################################


################### Identifier les outliers - travail sur les montants
df["montant"] = pd.to_numeric(df["montant"])
df['montantOriginal'] = df["montant"]

montant_borne_inf = 200.0
montant_borne_sup = 9.99e8
df['montant'] = np.where(df['montant'] <= montant_borne_inf, 0, df['montant']) 
df['montant'] = np.where(df['montant'] >= montant_borne_sup, 0, df['montant'])


######################################################################
#################### Gestion des id/code manquants
df.id = np.where(df.id.isnull(), '0000000000000000', df.id)
df.codeCPV = np.where(df.codeCPV.isnull(), '00000000', df.codeCPV)

# On applique au df la division
df["montant"] = pd.to_numeric(df["montant"])
df["montant"] = df["montant"]/df["nbTitulairesSurCeMarche"]

# Nettoyage colonnes
df['montant'] = np.where(df['montant'] == 0, np.NaN, df['montant'])


###############################################################################
##################### Nettoyage de ces nouvelles colonnes #####################
# Nettoyage des codes idTitulaires
caracteres_speciaux_dict = {"\\t" : "",
                            "-": "",
                            " ": "",
                            ".": "",
                            "?": "",
                            "    ":""}
mask =  (df.typeIdentifiant=='SIRET') | \
    (df.typeIdentifiant.isnull()) | \
    (df.typeIdentifiant=='nan')
df.idTitulaires[mask].replace(caracteres_speciaux_dict, inplace=True)


######## Gestion code CPV
df.codeCPV = df.codeCPV.astype(str)
df["CPV_min"] = df.codeCPV.str[:2]

########  Récupération code NIC 
df.idTitulaires = df.idTitulaires.astype(str)
df['nic'] = df.idTitulaires.str[-5:]

df.loc[~df["nic"].str.isdigit()] = np.NaN

################### Régions / Départements ##################
# Création de la colonne pour distinguer les départements
df['codePostal'] = df['lieuExecution.code'].str[:3]
listCorrespondance = {'976': 'YT', '974': 'RE', '972': 'MQ', '971': 'GP', '973': 'GF'}
df['codePostal'].replace(listCorrespondance, inplace=True)

df['codePostal'] = df['codePostal'].str[:2]

listCorrespondance2 = {'YT': '976', 'RE': '974', 'MQ': '972', 'GP': '971', 'GF': '973', 'TF': '98', 'NC' : '988','PF' : '987','WF' : '986','MF' : '978','PM' : '975','BL' : '977'}
df['codePostal'].replace(listCorrespondance2, inplace=True)

# Vérification si c'est bien un code postal
listeCP = ['01','02','03','04','05','06','07','08','09','2A','2B','98','976','974','972','971','973','97','988','987','984','978','975','977', '986'] \
    + [str(i) for i in list(np.arange(10,96,1))]
    

df['codePostal'] = np.where(~df['codePostal'].isin(listeCP), np.NaN, df['codePostal'])

#Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)
df['codePostal'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN, df['codePostal'])

###############################################################################
# Création de la colonne pour distinguer les régions
df['codeRegion'] = df['codePostal'].astype(str)
# Définition des codes des régions en fonctions des codes de départements
listCorrespondance = {
    '84' : ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
    '27' : ['21', '25', '39', '58', '70', '71', '89', '90'], 
    '53' : ['35', '22', '56', '29'],
    '24' : ['18', '28', '36', '37', '41', '45'], 
    '94' : ['2A', '2B', '20'],
    '44' : ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'], 
    '32' : ['02', '59', '60', '62', '80'],
    '11' : ['75', '77', '78', '91', '92', '93', '94', '95'], 
    '28' : ['14', '27', '50', '61', '76'],
    '75' : ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
    '76' : ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
    '52' : ['44', '49', '53', '72', '85'], '93' : ['04', '05', '06', '13', '83', '84'],
    '06': ['976'], 
    '04': ['974'], 
    '02': ['972'], 
    '01': ['971'], 
    '03': ['973'], 
    '98': ['97','98','988','986','984','987','975','977','978']
    }

#Inversion du dict
listCorrespondanceI = {value : str(key) for key, values in listCorrespondance.items() for value in values}

df['codeRegion'].replace(listCorrespondanceI, inplace=True)

# Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
df['codeRegion'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'], df['codeRegion'])
df['codeRegion'] = df['codeRegion'].astype(str)
# Vérification des codes région 
listeReg = ['84', '27', '53', '24', '94', '44', '32', '11', '28', '75', '76', 
            '52', '93', '01', '02', '03', '04', '06', '98'] #98 = collectivité d'outre mer


df['codeRegion'] = np.where(~df['codeRegion'].isin(listeReg), np.NaN, df['codeRegion'])

# Identification du nom des régions
df['Region'] = df['codeRegion'].astype(str)
correspondance_dict = {'84' : 'Auvergne-Rhône-Alpes','27' : 'Bourgogne-Franche-Comté','53' : 'Bretagne','24' : 'Centre-Val de Loire',
                      '94' : 'Corse','44' : 'Grand Est','32' : 'Hauts-de-France','11' : 'Île-de-France',
                      '28' : 'Normandie','75' : 'Nouvelle-Aquitaine','76' : 'Occitanie','52' : 'Pays de la Loire',
                      '93' : 'Provence-Alpes-Côte d\'Azur','01' : 'Guadeloupe', '02' : 'Martinique',
                      '03' : 'Guyane','04' : 'La Réunion','06' : 'Mayotte','98' : 'Collectivité d\'outre mer'}

df['Region'].replace(correspondance_dict, inplace=True)
# del chemin, chemindata, dfO, initial, key, listCorrespondance, listCorrespondanceI, string, value, word

###############################################################################
################### Date / Temps ##################    
#..............Travail sur les variables de type date           
df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
df.dateNotification = df.dateNotification.str[0:10] 
#On récupère l'année de notification
df['anneeNotification'] = df.dateNotification.str[0:4] 
df['anneeNotification'] = df['anneeNotification'].astype(float)
#On supprime les erreurs (0021 ou 2100 par exemple)
df['dateNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['dateNotification'])
df['dateNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['dateNotification'])
df['anneeNotification'] = np.where(df['anneeNotification'] < 1980, np.NaN, df['anneeNotification'])
df['anneeNotification'] = np.where(df['anneeNotification'] > 2100, np.NaN, df['anneeNotification'])
df['anneeNotification'] = df.anneeNotification.astype(str).str[:4]

#On récupère le mois de notification
df['moisNotification'] = df.dateNotification.str[5:7] 

######################################################################
# Mise en forme de la colonne montant
df["montant"] = pd.to_numeric(df["montant"])
df["montantOriginal"] = pd.to_numeric(df["montant"])

df['codePostal'] = df['codePostal'].astype(str)
df['codeRegion'] = df['codeRegion'].astype(str)
df['nic'] = df['nic'].astype(str)

# Mise en forme des données vides
df.datePublicationDonnees = np.where(df.datePublicationDonnees == '', np.NaN, df.datePublicationDonnees)
df.idTitulaires = np.where(df.idTitulaires == '', np.NaN, df.idTitulaires)
df.denominationSociale = np.where(
    (df.denominationSociale == 'N/A') | (df.denominationSociale == 'null'),
    np.NaN, df.denominationSociale)

######################################################################
# Colonne supplémentaire pour indiquer si la valeur est estimée ou non
df['montantEstime'] = np.where(df['montant'].isnull(), 'Oui', 'Non')

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
medianeRegFP.columns = ['conca','montantEstimation']
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
medianeRegFP.columns = ['conca','montantEstimation']
df = pd.merge(df, medianeRegFP, on='conca')
# Remplacement des valeurs manquantes par la médiane du groupe
df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])
# S'il reste encore des valeurs nulles...
df['montant'] = np.where(df['montant'].isnull(), df['montant'].median(), df['montant'])
del df['conca'], df['montantEstimation'], df['index']
del medianeRegFP

# Colonne par marché
df['montantTotalMarché'] = df["montant"] * df["nbTitulairesSurCeMarche"]

##############################################################################
##############################################################################
### Rectification des durées en mois aberrantes

# On cherche les éventuelles erreurs mois -> jours
df['dureeMoisEstime'] = np.where(
      (df['montant']==df['dureeMois'])
    | (df['montant']/df['dureeMois'] < 100)
    | (df['montant']/df['dureeMois'] < 1000) & (df['dureeMois']>=12)
    | ((df['dureeMois'] == 30) & (df['montant'] < 200000))
    | ((df['dureeMois'] == 31) & (df['montant'] < 200000))
    | ((df['dureeMois'] == 360) & (df['montant'] < 10000000))
    | ((df['dureeMois'] == 365) & (df['montant'] < 10000000))
    | ((df['dureeMois'] == 366) & (df['montant'] < 10000000))
    | ((df['dureeMois'] > 120) & (df['montant'] < 2000000)), 
    "Oui", "Non")

# On corrige pour les colonnes considérées comme aberrantes, on divise par 30 (nombre de jours par mois)
df['dureeMoisCalculee'] = np.where(df['dureeMoisEstime'] == "Oui", round(df['dureeMois']/30,0), df['dureeMois'])
# Comme certaines valeurs atteignent zero, on remplace par un mois
df['dureeMoisCalculee'] = np.where(df['dureeMoisCalculee'] == 0, 1, df['dureeMoisCalculee'])

# Au cas ils restent encore des données aberrantes, on les remplace par un mois -> A CHALLENGER
df['dureeMoisCalculee'] = np.where(
      (df['montant']/df['dureeMois'] < 100)
    | (df['montant']/df['dureeMois'] < 1000) & (df['dureeMois']>=12)
    | ((df['dureeMois'] == 30) & (df['montant'] < 200000))
    | ((df['dureeMois'] == 31) & (df['montant'] < 200000))
    | ((df['dureeMois'] == 360) & (df['montant'] < 10000000))
    | ((df['dureeMois'] == 365) & (df['montant'] < 10000000))
    | ((df['dureeMois'] == 366) & (df['montant'] < 10000000))
    | ((df['dureeMois'] > 120) & (df['montant'] < 2000000)), 
    1, df.dureeMoisCalculee)

######################################################################
######## Enrichissement des données via les codes siret/siren ########
### Utilisation d'un autre data frame pour traiter les Siret unique

dfSIRET = pd.DataFrame.copy(df[['idTitulaires', 'typeIdentifiant', 'denominationSociale']])
dfSIRET = dfSIRET.drop_duplicates(subset=['idTitulaires'], keep='first')
dfSIRET.reset_index(inplace=True) 
dfSIRET.idTitulaires = dfSIRET.idTitulaires.astype(str)

dfSIRET["idTitulaires"] = np.where(~dfSIRET["idTitulaires"].str.isdigit(), '00000000000000', dfSIRET.idTitulaires)

del dfSIRET['index']
dfSIRET.reset_index(inplace=True, drop=True) 


dfSIRET.rename(columns={"idTitulaires" : "siret", "typeIdentifiant" : "siren"}, inplace=True)
dfSIRET.siren = dfSIRET.siret.str[:siren_len]
dfSIRET.denominationSociale = dfSIRET.denominationSociale.astype(str)

######################################################################
### On supprime les siret déjà identifié comme faux 
path = os.path.join(path_to_data, error_siret_file_name)

try :
    archiveErrorSIRET = pd.read_csv(path, sep=';', encoding='utf-8', 
                                    dtype={'siren' : str, 'siret' : str, 'denominationSociale' : str})
    dfSIRET = pd.merge(dfSIRET, archiveErrorSIRET, how='outer', indicator='source')
    dfSIRET = dfSIRET[dfSIRET.source.eq('left_only')].drop('source', axis=1)
    dfSIRET.reset_index(inplace=True, drop=True)
    print('Erreurs archivées supprimées')
except:
    archiveErrorSIRET = pd.DataFrame(columns=['siret', 'siren', 'denominationSociale'])
    print('Aucune archive d\'erreur')
    pass

######################################################################
#StockEtablissement_utf8
path = os.path.join(path_to_data, conf["stock_etablissement"])
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
    'nomenclatureActivitePrincipaleEtablissement']    

result = pd.DataFrame(columns)
chunksize = 1000000
for gm_chunk in pd.read_csv(path, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns):
    gm_chunk['siret'] = gm_chunk['siret'].astype(str)
    resultTemp = pd.merge(dfSIRET, gm_chunk, on=['siret'])
    result = pd.concat([result, resultTemp], axis=0)
result = result.drop_duplicates(subset=['siret'], keep='first')
del resultTemp, gm_chunk

del result['siren_x'], result['siren_y']
dfSIRET = pd.merge(dfSIRET, result, how='outer', on=['siret'])
nanSiret = dfSIRET[dfSIRET.activitePrincipaleEtablissement.isnull()]
dfSIRET = dfSIRET[dfSIRET.activitePrincipaleEtablissement.notnull()]
nanSiret = nanSiret.loc[:,["siret", "siren", "denominationSociale_x"]]

# %%
print("hello")

#%%
result2 = pd.DataFrame(columns)
for gm_chunk in pd.read_csv(path, chunksize=chunksize, sep=',', encoding='utf-8', usecols=columns):
    gm_chunk['siren'] = gm_chunk['siren'].astype(str)
    resultTemp = pd.merge(nanSiret, gm_chunk, on=['siren'])
    result2 = pd.concat([result2, resultTemp], axis=0)
    
result2 = result2.drop_duplicates(subset=['siren'], keep='first')
del result2['siret_x'], result2['siret_y'], result2['denominationSociale_x']
del resultTemp, gm_chunk

result2 = pd.merge(nanSiret, result2, how='inner', on='siren')
myList = list(result2.columns); myList[2] = 'denominationSociale'; result2.columns = myList
del dfSIRET['denominationSociale_y']
dfSIRET.columns = myList

######## Merge des deux resultats
enrichissementInsee = pd.concat([dfSIRET, result2])

####### Récupération des données tjrs pas enrichies
nanSiren = pd.merge(nanSiret, result2, indicator=True, how='outer', on='siren')
nanSiren = nanSiren[nanSiren['activitePrincipaleEtablissement'].isnull()]
nanSiren = nanSiren.iloc[:,:3]
nanSiren.columns = ['siret', 'siren', 'denominationSociale'] 
nanSiren.reset_index(inplace=True, drop=True)
del dfSIRET, nanSiret, result, result2, myList

#....... Solution complémentaire pour ceux non-identifié dans la BDD
columns = [
    'index', 
    'rue', 
    'siret', 
    'ville', 
    'typeEntreprise', 
    'codeType', 
    'detailsType', 
    'SIRETisMatched']

def get_scrap_dataframe(index, code):
    url = 'https://www.infogreffe.fr/entreprise-societe/' + code
        
    page = requests.get(url)
    tree = html.fromstring(page.content)
    
    rueSiret = tree.xpath('//div[@class="identTitreValeur"]/text()')
    infos = tree.xpath('//p/text()')
    details = tree.xpath('//a/text()')

    rue = rueSiret[1]
    siret = rueSiret[5].replace(" ","")
    ville = infos[7]
    typeEntreprise = infos[15]
    codeType = infos[16].replace(" : ","")
    detailsType1 = details[28]
    detailsType2 = details[29]
    
    if len(code) == 9:
        SIRETisMatched = siret[:9] == code
    else:
        SIRETisMatched = siret == code
    
    if (detailsType1 ==' '):
        detailsType = detailsType2
    else:
        detailsType = detailsType1
    
    if not SIRETisMatched:
        codeSiret = tree.xpath('//span[@class="data ficheEtablissementIdentifiantSiret"]/text()')
        infos = tree.xpath('//span[@class="data"]/text()')
        
        rue = infos[8]
        siret = codeSiret[0].replace(" ", "")
        ville = infos[9].replace(",\xa0","")
        typeEntreprise = infos[4]
        detailsType = infos[11]
        SIRETisMatched = (siret == code)
        
    scrap = pd.DataFrame([index, rue, siret, ville, typeEntreprise, codeType, detailsType, SIRETisMatched]).T
    scrap.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched']
    return scrap    
    
df_scrap = pd.DataFrame(columns = columns)    
for i in tqdm(range(len(nanSiren))):
    try:
        scrap = get_scrap_dataframe(i, nanSiren.siret[i])
        df_scrap = pd.concat([df_scrap, scrap], axis=0)       
    except:
        try :
            scrap = get_scrap_dataframe(i, nanSiren.siren[i])
            df_scrap = pd.concat([df_scrap, scrap], axis=0)        
        except:
            scrap = pd.DataFrame([i, ' ', ' ', ' ', ' ', ' ', ' ', False]).T; 
            scrap.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched']
            df_scrap = pd.concat([df_scrap, scrap], axis=0)
            pass

# Récupération des résultats
nanSiren.reset_index(inplace=True)
resultat = pd.merge(nanSiren, df_scrap, on='index')
resultatScrap1 = resultat[resultat.rue != ' ']

# Données encore manquantes
dfDS = resultat[resultat.rue == ' ']
dfDS = dfDS.iloc[:,1:4]
dfDS.columns = ['siret', 'siren', 'denominationSociale'] 
dfDS.reset_index(inplace=True, drop=True)
#del codeSiret, codeType, detailType, details, detailsType, detailsType1, detailsType2, i, index, infos, rue, rueSiret, scrap, siret, typeEntreprise, url, SIRETisMatched, ville, df_scrap, nanSiren, resultat

######################################################################
def requete(nom):
    pager.get('https://www.infogreffe.fr/recherche-siret-entreprise/chercher-siret-entreprise.html')
    pager.find_element_by_xpath('//*[@id="p1_deno"]').send_keys(nom, Keys.ENTER)
    time.sleep(2)
    url = pager.current_url
    return url
options = Options()
options.add_argument('--headless')
pager = webdriver.Firefox(options=options)
#pager = webdriver.PhantomJS('webdriver/phantomjs.exe')

df_scrap2 = pd.DataFrame(columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched'])    
for i in tqdm(range(len(dfDS))):
    try:
        url = requete(dfDS.denominationSociale[i])
        
        page = requests.get(url)
        tree = html.fromstring(page.content)
        
        rueSiret = tree.xpath('//div[@class="identTitreValeur"]/text()')
        infos = tree.xpath('//p/text()')
        details = tree.xpath('//a/text()')
        
        index = i
        rue = rueSiret[1]
        siret = rueSiret[5].replace(" ","")
        ville = infos[7]
        typeEntreprise = infos[15]
        codeType = infos[16].replace(" : ","")
        detailsType1 = details[28]
        detailsType2 = details[29]
        SIRETisMatched = (siret == dfDS.siret[i])
        if (detailsType1 ==' '):
            detailType = detailsType2
        else:
            detailsType = detailsType1
        
        scrap2 = pd.DataFrame([index, rue, siret, ville, typeEntreprise, codeType, detailsType, SIRETisMatched]).T; scrap2.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched']
        df_scrap2 = pd.concat([df_scrap2, scrap2], axis=0)
    except:
        index = i
        scrap2 = pd.DataFrame([index, ' ', ' ', ' ', ' ', ' ', ' ', False]).T; scrap2.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'SIRETisMatched']
        df_scrap2 = pd.concat([df_scrap2, scrap2], axis=0)
        pass
pager.quit()

# Récupération des résultats
dfDS.reset_index(inplace=True)
resultat = pd.merge(dfDS, df_scrap2, on='index')
resultatScrap2 = resultat[resultat.rue != ' ']

###############################################################################
### Enregistrement des entreprises n'ayant aucune correspondance
errorSIRET = resultat[(resultat.siret_y=='')|(resultat.siret_y=='')|(resultat.siret_y==' ')|(resultat.siret_y.isnull())]
errorSIRET = errorSIRET[['siret_x', 'siren', 'denominationSociale']]; errorSIRET.columns = ['siret', 'siren', 'denominationSociale']; errorSIRET.reset_index(inplace=True, drop=True)
errorSIRET = pd.concat([errorSIRET, archiveErrorSIRET], axis=0)
errorSIRET = errorSIRET.drop_duplicates(subset=['siret', 'siren', 'denominationSociale'], keep='first')
errorSIRET.to_csv('errorSIRET.csv', sep=';',index = False, header=True, encoding='utf-8')
###############################################################################

# On réuni les résultats du scraping
enrichissementScrap = pd.concat([resultatScrap1, resultatScrap2])
del enrichissementScrap['index'], enrichissementScrap['siret_y'], enrichissementScrap['SIRETisMatched']

############ Arrangement des colonnes 
# Gestion bdd insee
enrichissementInsee.reset_index(inplace=True, drop=True)
listCorrespondance = {'ALL': 'Allée', 'AV': 'Avenue', 'BD': 'Boulevard', 'CAR': 'Carrefour',
                      'CHE': 'Chemin', 'CHS': 'Chaussée', 'CITE': 'Cité', 'COR': 'Corniche',
                      'CRS': 'Cours', 'DOM': 'Domaine', 'DSC': 'Descente', 'ECA': 'Ecart',
                      'ESP': 'Esplanade', 'FG': 'Faubourg', 'GR': 'Grande Rue', 'HAM': 'Hameau',
                      'HLE': 'Halle', 'IMP': 'Impasse', 'LD': 'Lieu dit', 'LOT': 'Lotissement',
                      'MAR': 'Marché', 'MTE': 'Montée', 'PAS': 'Passage', 'PL': 'Place', 
                      'PLN': 'Plaine', 'PLT': 'Plateau', 'PRO': 'Promenade', 'PRV': 'Parvis',
                      'QUA': 'Quartier', 'QUAI': 'Quai', 'RES': 'Résidence', 'RLE': 'Ruelle',
                      'ROC': 'Rocade', 'RPT': 'Rond Point', 'RTE': 'Route', 'RUE': 'Rue', 
                      'SEN': 'Sentier', 'SQ': 'Square', 'TPL': 'Terre-plein', 'TRA': 'Traverse',
                      'VLA': 'Villa', 'VLGE': 'Village'}

enrichissementInsee['typeVoieEtablissement'].replace(listCorrespondance, inplace=True)
enrichissementInsee['rue'] = enrichissementInsee.typeVoieEtablissement + ' ' + enrichissementInsee.libelleVoieEtablissement
enrichissementInsee['activitePrincipaleEtablissement'] = enrichissementInsee['activitePrincipaleEtablissement'].str.replace(".", "")
del enrichissementInsee['typeVoieEtablissement'], enrichissementInsee['libelleVoieEtablissement'], enrichissementInsee['nic'], enrichissementInsee['nomenclatureActivitePrincipaleEtablissement']

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
del enrichissementScrap['ville'], enrichissementScrap['typeEntreprise'], enrichissementScrap['detailsType']

# Renomme les colonnes
enrichissementScrap.columns = [
    'siret', 
    'siren', 
    'denominationSociale', 
    'adresseEtablissement', 
    'codeTypeEtablissement', 
    'codePostalEtablissement', 
    'communeEtablissement']

enrichissementInsee = enrichissementInsee[['siret',
                        'siren',
                        'denominationSociale',
                        'activitePrincipaleEtablissement',
                        'codeCommuneEtablissement',
                        'codePostalEtablissement',
                        'libelleCommuneEtablissement',
                        'rue']]
enrichissementInsee.columns = [
    'siret', 
    'siren', 
    'denominationSociale', 
    'codePostalEtablissement', 
    'communeEtablissement', 
    'codeCommuneEtablissement', 
    'codeTypeEtablissement', 
    'adresseEtablissement']

# df final pour enrichir les données des entreprises
dfenrichissement = pd.concat([enrichissementInsee, enrichissementScrap])
dfenrichissement = dfenrichissement.astype(str)
# On s'assure qu'il n'y ai pas de doublons
dfenrichissement = dfenrichissement.drop_duplicates(subset=['siret'], keep=False)

########### Ajout au df principal !
del df['denominationSociale']
# Concaténation
df =  pd.merge(df, dfenrichissement, how='outer', left_on="idTitulaires", right_on="siret")
#df =  pd.merge(df, dfenrichissement, how='left', left_on="idTitulaires", right_on="siret")

del df['CPV_min'], df['uid'], df['uuid']

######################################################################
################### Enrichissement avec le code CPV ##################
######################################################################
# Importation et mise en forme des codes/ref CPV
refCPV = pd.read_excel("dataEnrichissement/cpv_2008_ver_2013.xlsx", usecols=['CODE', 'FR'])
refCPV.columns = ['CODE', 'refCodeCPV']
refCPV_min = pd.DataFrame.copy(refCPV, deep = True)
refCPV_min["CODE"] = refCPV_min.CODE.str[0:8]
refCPV_min = refCPV_min.drop_duplicates(subset=['CODE'], keep='first')
refCPV_min.columns = ['CODEmin', 'FR2']
# Merge avec le df principal
df = pd.merge(df, refCPV, how='left', left_on="codeCPV", right_on="CODE")
df = pd.merge(df, refCPV_min, how='left', left_on="codeCPV", right_on="CODEmin")
# Garde uniquement la colonne utile / qui regroupe les nouvelles infos
df.refCodeCPV = np.where(df.refCodeCPV.isnull(), df.FR2, df.refCodeCPV)
del df['CODE'], df['CODEmin'], df['FR2'], refCPV, refCPV_min, i

######################################################################
######################################################################
with open('config.dictionary', 'wb') as df_backup1:
  pickle.dump(df, df_backup1)
with open('config.dictionary', 'rb') as df_backup1:
    df_backup1 = pickle.load(df_backup1)
df = pd.DataFrame.copy(df_backup1, deep = True)
######################################################################
######################################################################


######################################################################
############## Enrichissement des données des acheteurs ##############
######################################################################
######## Enrichissement des données via les codes siret/siren ########
### Utilisation d'un autre data frame pour traiter les Siret unique : acheteur.id
dfAcheteurId = df['acheteurId']; dfAcheteurId.columns = ['siret']
dfAcheteurId = dfAcheteurId.drop_duplicates(subset=['siret'], keep='first')
dfAcheteurId.reset_index(inplace=True, drop=True) 
dfAcheteurId.siret = dfAcheteurId.siret.astype(str)

#StockEtablissement_utf8
chemin = 'dataEnrichissement/StockEtablissement_utf8.csv'
result = pd.DataFrame(columns = ['siret', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement'])    
for gm_chunk in pd.read_csv(chemin, chunksize=1000000, sep=',', encoding='utf-8', usecols=['siret', 'codePostalEtablissement', 
                                                                                           'libelleCommuneEtablissement', 
                                                                                           'codeCommuneEtablissement']):
    gm_chunk['siret'] = gm_chunk['siret'].astype(str)
    resultTemp = pd.merge(dfAcheteurId, gm_chunk, on=['siret'])
    result = pd.concat([result, resultTemp], axis=0)
result = result.drop_duplicates(subset=['siret'], keep='first')

dfAcheteurId["siren"] = np.nan
dfAcheteurId.siren = dfAcheteurId.siret.str[:siren_len]
chemin = 'dataEnrichissement/StockEtablissement_utf8.csv'
result2 = pd.DataFrame(columns = ['siren', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement'])    
for gm_chunk in pd.read_csv(chemin, chunksize=1000000, sep=',', encoding='utf-8', usecols=['siren', 'codePostalEtablissement', 
                                                                                           'libelleCommuneEtablissement', 
                                                                                           'codeCommuneEtablissement']):
    gm_chunk['siren'] = gm_chunk['siren'].astype(str)
    resultTemp = pd.merge(dfAcheteurId, gm_chunk, on="siren")
    result2 = pd.concat([result2, resultTemp], axis=0)
result2 = result2.drop_duplicates(subset=['siren'], keep='first')
siret = pd.DataFrame(result['siret']); siret.columns=['s']
result2 = pd.merge(result2, siret, how='outer',  left_on='siret', right_on='s')
result2 = result2[result2.s.isnull()]; del result2['s']

dfManquant = pd.merge(dfAcheteurId, result, how='outer', on='siret')
dfManquant = dfManquant[dfManquant['codeCommuneEtablissement'].isnull()]
dfManquant  = dfManquant .iloc[:,:2]
result2 = pd.merge(dfManquant, result2, how='inner', on='siren')
del result2['siret_y'], result2['siren']
result2.columns = ['siret', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement']


enrichissementAcheteur = pd.concat([result, result2])
enrichissementAcheteur.columns = ['acheteur.id', 'codePostalAcheteur', 'libelleCommuneAcheteur', 'codeCommuneAcheteur']
enrichissementAcheteur = enrichissementAcheteur.drop_duplicates(subset=['acheteur.id'], keep='first')


df = pd.merge(df, enrichissementAcheteur, how='left', on='acheteur.id')
#dfEnregistrement = pd.merge(df, enrichissementAcheteur, how='left', on='acheteur.id')

del chemin, dfAcheteurId, dfManquant, enrichissementAcheteur, gm_chunk, result, result2, resultTemp, siret

######################################################################
######################################################################
# Ajustement de certaines colonnes
df.codePostalEtablissement = df.codePostalEtablissement.astype(str).str[:5]
df.codePostalAcheteur = df.codePostalAcheteur.astype(str).str[:5]
df.codeCommuneEtablissement = df.codeCommuneEtablissement.astype(str).str[:5]
df.codeCommuneAcheteur = df.codeCommuneAcheteur.astype(str).str[:5]

df.anneeNotification = df.anneeNotification.astype(str)
df.codePostal = df.codePostal.astype(str)

# Réorganisation des colonnes et de leur nom
df.columns = ['source', 'type', 'nature', 'procedure', 'dureeMois',
       'datePublicationDonnees', 'lieuExecutionCode',
       'lieuExecutionTypeCode', 'lieuExecutionNom', 'identifiantMarche', 'objetMarche', 'codeCPV',
       'dateNotification', 'montant', 'formePrix', 'acheteurId',
       'acheteurNom', 'typeIdentifiantEtablissement', 'idEtablissement', 'montantOriginal', 'nbTitulairesSurCeMarche',
       'nicEtablissement', 'codeDepartementAcheteur', 'codeRegionAcheteur', 'regionAcheteur',
       'anneeNotification', 'moisNotification', 'montantEstEstime', 'montantTotalMarche',
       'dureeMoisEstEstime', 'dureeMoisCalculee', 'siretEtablissement', 'sirenEtablissement', 
       'denominationSocialeEtablissement', 'codePostalEtablissement',
       'communeEtablissement', 'codeCommuneEtablissement',
       'codeTypeEtablissement', 'adresseEtablissement', 'referenceCPV',
       'codePostalAcheteur', 'libelleCommuneAcheteur', 'codeCommuneAcheteur']

df = df[['source', 'type', 'nature', 'procedure', 'datePublicationDonnees', 'dateNotification',  
         'anneeNotification', 'moisNotification', 'formePrix', 'identifiantMarche', 'objetMarche' , 'codeCPV',
         'referenceCPV', 'montantOriginal', 'montant', 'montantEstEstime', 'montantTotalMarche', 'nbTitulairesSurCeMarche',
         'dureeMois', 'dureeMoisEstEstime', 'dureeMoisCalculee', 'acheteurId', 'acheteurNom',
         'lieuExecutionCode', 'lieuExecutionTypeCode', 'lieuExecutionNom', 'codeCommuneAcheteur',
         'codePostalAcheteur', 'libelleCommuneAcheteur', 'codeDepartementAcheteur', 'codeRegionAcheteur', 'regionAcheteur',
         'typeIdentifiantEtablissement', 'idEtablissement', 'nicEtablissement', 'adresseEtablissement',
         'codeCommuneEtablissement', 'codePostalEtablissement', 'codeTypeEtablissement', 'communeEtablissement',
         'denominationSocialeEtablissement','sirenEtablissement', 'siretEtablissement']]

# Rectification codePostalAcheteur et codeCommuneAcheteur
df.codePostalAcheteur = df.codePostalAcheteur.str.replace(".", "")
df.codePostalAcheteur = df.codePostalAcheteur.str.replace(" ", "")
if len(df.codePostalAcheteur)<5:
    df.codePostalAcheteur = '0'+df.codePostalAcheteur

df.codeCommuneAcheteur = df.codeCommuneAcheteur.str.replace(".", "")
df.codeCommuneAcheteur = df.codeCommuneAcheteur.str.replace(" ", "")
if len(df.codeCommuneAcheteur)<5:
    df.codeCommuneAcheteur = '0'+df.codeCommuneAcheteur

######################################################################
# Petites corrections
df['lieuExecutionTypeCode'] = df['lieuExecutionTypeCode'].str.upper()
df['lieuExecutionTypeCode'] = np.where(df['lieuExecutionTypeCode'] == 'CODE DÉPARTEMENT', 'CODE DEPARTEMENT', df['lieuExecutionTypeCode'])
df['lieuExecutionTypeCode'] = np.where(df['lieuExecutionTypeCode'] == 'CODE RÉGION', 'CODE REGION', df['lieuExecutionTypeCode'])

df['nature'] = np.where(df['nature'] == 'Délégation de service public', 'DELEGATION DE SERVICE PUBLIC', df['nature'])
df['nature'] = df['nature'].str.upper()

######################################################################
######################################################################
with open('config.dictionary', 'wb') as df_backup2:
  pickle.dump(df, df_backup2)
with open('config.dictionary', 'rb') as df_backup2:
    df_backup2 = pickle.load(df_backup2)
df_decp = pd.DataFrame.copy(df_backup2, deep = True)
######################################################################
######################################################################
df_decp = pd.DataFrame.copy(df, deep = True)

compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df_decp.to_csv('out.zip', index=False,
          compression=compression_opts)  

del [archiveErrorSIRET, codeType, detailType, details, detailsType, detailsType1, 
     detailsType2, dfDS, df_scrap2, dfenrichissement, enrichissementInsee, 
     enrichissementScrap, errorSIRET, index, infos, listCorrespondance, 
     listeCP, listeReg, resultat, resultatScrap1, resultatScrap2, rue, rueSiret, 
     scrap2, typeEntreprise, url, SIRETisMatched, ville, df]





######################################################################
######## Enrichissement latitude & longitude avec adresse la ville 
df_villes = pd.read_csv('dataEnrichissement/code-insee-postaux-geoflar.csv', 
                        sep=';', header = 0, error_bad_lines=False,
                        usecols=['CODE INSEE', 'geom_x_y', 'Superficie', 'Population'])
df_villes['ordre']=0
df_villes2 = pd.read_csv('dataEnrichissement/code-insee-postaux-geoflar.csv', 
                        sep=';', header = 0, error_bad_lines=False,
                        usecols=['Code commune complet', 'geom_x_y', 'Superficie', 'Population'])
df_villes2['ordre']=1
df_villes2.columns = ['geom_x_y', 'Superficie', 'Population', 'CODE INSEE', 'ordre']
df_villes = pd.concat([df_villes2, df_villes])
del df_villes2
#Suppression des doublons
df_villes = df_villes.sort_values(by = 'ordre', ascending = False)
df_villes.reset_index(inplace=True, drop=True)
df_villes = df_villes.drop_duplicates(subset=['CODE INSEE', 'geom_x_y', 'Superficie', 'Population'], keep='last')
df_villes = df_villes.drop_duplicates(subset=['CODE INSEE'], keep='last')
df_villes = df_villes[(df_villes['CODE INSEE'].notnull()) & (df_villes.geom_x_y.notnull())]
del df_villes['ordre']
df_villes.reset_index(inplace=True, drop=True)
#Multiplier population par 1000
df_villes.Population = df_villes.Population.astype(float)
df_villes.Population = round(df_villes.Population*1000,0)
# Divise la colonne geom_x_y pour obtenir la latitude et la longitude séparemment
# Latitude avant longitude
df_villes.geom_x_y = df_villes.geom_x_y.astype(str)
df_sep = pd.DataFrame(df_villes.geom_x_y.str.split(',',1, expand=True))
df_sep.columns = ['latitude','longitude']

df_villes = df_villes.join(df_sep)
del df_villes['geom_x_y'], df_sep
df_villes.latitude = df_villes.latitude.astype(float)
df_villes.longitude = df_villes.longitude.astype(float)

################################# Ajout au dataframe principal
# Ajout pour les acheteurs
df_villes.columns = ['codeCommuneAcheteur', 'populationAcheteur', 'superficieAcheteur', 'latitudeAcheteur','longitudeAcheteur']
df_decp.codeCommuneAcheteur = df_decp.codeCommuneAcheteur.astype(object)
df_villes.codeCommuneAcheteur = df_villes.codeCommuneAcheteur.astype(object)
df_decp = pd.merge(df_decp, df_villes, how='left', on='codeCommuneAcheteur')
# Ajout pour les etablissement
df_villes.columns = ['codeCommuneEtablissement', 'populationEtablissement', 'superficieEtablissement', 'latitudeEtablissement','longitudeEtablissement']
df_decp.codeCommuneEtablissement = df_decp.codeCommuneEtablissement.astype(object)
df_villes.codeCommuneEtablissement = df_villes.codeCommuneEtablissement.astype(object)
df_decp = pd.merge(df_decp, df_villes, how='left', on='codeCommuneEtablissement')
del df_villes
########### Calcul de la distance entre l'acheteur et l'etablissement
# Utilisation de la formule de Vincenty avec le rayon moyen de la Terre
#df_decp['distanceAcheteurEtablissement'] = round((((2*6378137+6356752)/3)*np.arctan2(np.sqrt((np.cos(np.radians(df_decp.latitudeEtablissement))*np.sin(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))*(np.cos(np.radians(df_decp.latitudeEtablissement))*np.sin(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur)))) + (np.cos(np.radians(df_decp.latitudeAcheteur))*np.sin(np.radians(df_decp.latitudeEtablissement)) - np.sin(np.radians(df_decp.latitudeAcheteur))*np.cos(np.radians(df_decp.latitudeEtablissement))*np.cos(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))*(np.cos(np.radians(df_decp.latitudeAcheteur))*np.sin(np.radians(df_decp.latitudeEtablissement)) - np.sin(np.radians(df_decp.latitudeAcheteur))*np.cos(np.radians(df_decp.latitudeEtablissement))*np.cos(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))), (np.sin(np.radians(df_decp.latitudeAcheteur)))*(np.sin(np.radians(df_decp.latitudeEtablissement))) + (np.cos(np.radians(df_decp.latitudeAcheteur)))*(np.cos(np.radians(df_decp.latitudeEtablissement)))*(np.cos(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))))/1000,0)
df_decp['distanceAcheteurEtablissement'] = round((((2*6378137+6356752)/3)*np.arctan2(
        np.sqrt((np.cos(np.radians(df_decp.latitudeEtablissement))*np.sin(
        np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))*(
        np.cos(np.radians(df_decp.latitudeEtablissement))*np.sin(np.radians(np.fabs(
        df_decp.longitudeEtablissement-df_decp.longitudeAcheteur)))) + (np.cos(np.radians(
        df_decp.latitudeAcheteur))*np.sin(np.radians(df_decp.latitudeEtablissement)) - np.sin(
        np.radians(df_decp.latitudeAcheteur))*np.cos(np.radians(df_decp.latitudeEtablissement))*np.cos(
        np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))*(
        np.cos(np.radians(df_decp.latitudeAcheteur))*np.sin(np.radians(df_decp.latitudeEtablissement)) - np.sin(
        np.radians(df_decp.latitudeAcheteur))*np.cos(np.radians(df_decp.latitudeEtablissement))*np.cos(
        np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))), (np.sin(
        np.radians(df_decp.latitudeAcheteur)))*(np.sin(np.radians(df_decp.latitudeEtablissement))) + (
        np.cos(np.radians(df_decp.latitudeAcheteur)))*(np.cos(np.radians(df_decp.latitudeEtablissement)))*(
        np.cos(np.radians(np.fabs(df_decp.longitudeEtablissement-df_decp.longitudeAcheteur))))))/1000,0)

# Taux d'enrichissement
round(100-df_decp.distanceAcheteurEtablissement.isnull().sum()/len(df_decp)*100,2)

# Remise en forme des colonnes géo-spatiales
df_decp.latitudeAcheteur = df_decp.latitudeAcheteur.astype(str)
df_decp.longitudeAcheteur = df_decp.longitudeAcheteur.astype(str)
df_decp['geomAcheteur'] = df_decp.latitudeAcheteur + ',' + df_decp.longitudeAcheteur
df_decp.latitudeEtablissement = df_decp.latitudeEtablissement.astype(str)
df_decp.longitudeEtablissement = df_decp.longitudeEtablissement.astype(str)
df_decp['geomEtablissement'] = df_decp.latitudeEtablissement + ',' + df_decp.longitudeEtablissement

df_decp['geomAcheteur'] = np.where(df_decp['geomAcheteur'] == 'nan,nan', np.NaN, df_decp['geomAcheteur'])
df_decp['geomEtablissement'] = np.where(df_decp['geomEtablissement'] == 'nan,nan', np.NaN, df_decp['geomEtablissement'])
df_decp.reset_index(inplace=True, drop=True)

###############################################################################
############################ Segmentation de marché ###########################
###############################################################################
#... Créer une bdd par villes (acheteur/client)
dfBIN = df_decp[['type', 'nature', 'procedure', 'lieuExecutionTypeCode']]
# Arrangement du code du lieu d'exécution
dfBIN['lieuExecutionTypeCode'] = np.where(dfBIN['lieuExecutionTypeCode'] == 'CODE ARRONDISSEMENT', 'CODE DEPARTEMENT', dfBIN['lieuExecutionTypeCode'])
dfBIN['lieuExecutionTypeCode'] = np.where((dfBIN['lieuExecutionTypeCode'] == 'CODE COMMUNE') | (dfBIN['lieuExecutionTypeCode'] == 'CODE POSTAL'), 'CODE COMMUNE/POSTAL', dfBIN['lieuExecutionTypeCode'])

#... On binarise les variables qualitatives
def binateur(data, to_bin):
    data = data.copy()
    X = data[to_bin]
    X = pd.get_dummies(X)
    data = data.drop(columns=to_bin)
    X = X.fillna(0)
    return pd.concat([data, X], axis=1)
dfBIN = binateur(dfBIN, dfBIN.columns) 

#... Selection des variables quantitatives + nom de la commune
dfNoBin = df_decp[['libelleCommuneAcheteur', 'montant', 'dureeMois', 'dureeMoisCalculee', 'distanceAcheteurEtablissement']]
# Création d'une seule colonne pour la durée du marché
dfNoBin['duree'] = round(dfNoBin.dureeMoisCalculee, 0)
del dfNoBin['dureeMois'], dfNoBin['dureeMoisCalculee']
# On modifie les valeurs manquantes pour la distance en appliquant la médiane
dfNoBin.distanceAcheteurEtablissement = np.where(dfNoBin['distanceAcheteurEtablissement'].isnull(), dfNoBin['distanceAcheteurEtablissement'].median(), dfNoBin['distanceAcheteurEtablissement'])

# On obtient alors notre df prêt sans variables qualitatives (sauf libellé)
df = dfNoBin.join(dfBIN)
del dfNoBin, dfBIN
df = df[df['libelleCommuneAcheteur'].notnull()]
df['nbContrats'] = 1 # Trouver autre solution
df = df.groupby(['libelleCommuneAcheteur']).sum().reset_index()

#... Fréquence 
ensemble = ['type_Contrat de concession', 'type_Marché', 'nature_ACCORD-CADRE', 'nature_CONCESSION DE SERVICE',
       'nature_CONCESSION DE SERVICE PUBLIC', 'nature_CONCESSION DE TRAVAUX', 'nature_DELEGATION DE SERVICE PUBLIC', 'nature_MARCHÉ',
       'nature_MARCHÉ DE PARTENARIAT', 'nature_MARCHÉ HORS ACCORD CADRE', 'nature_MARCHÉ SUBSÉQUENT', "procedure_Appel d'offres ouvert",
       "procedure_Appel d'offres restreint", 'procedure_Dialogue compétitif', 'procedure_Marché négocié sans publicité ni mise en concurrence préalable',
       'procedure_Marché public négocié sans publicité ni mise en concurrence préalable', 'procedure_Procédure adaptée', 'procedure_Procédure avec négociation',
       'procedure_Procédure non négociée ouverte', 'procedure_Procédure non négociée restreinte', 'procedure_Procédure négociée ouverte',
       'procedure_Procédure négociée restreinte', 'lieuExecutionTypeCode_CODE CANTON', 'lieuExecutionTypeCode_CODE COMMUNE/POSTAL',
       'lieuExecutionTypeCode_CODE DEPARTEMENT', 'lieuExecutionTypeCode_CODE PAYS', 'lieuExecutionTypeCode_CODE REGION']
for x in ensemble:
    df[x] = df[x]/df['nbContrats']
del ensemble, x

#... Duree, montant et distance moyenne par ville (par rapport au nb de contrats)
df.distanceAcheteurEtablissement = round(df.distanceAcheteurEtablissement/df['nbContrats'],0)
df.duree = round(df.duree/df['nbContrats'],0)
df['montantMoyen'] = round(df.montant/df['nbContrats'],0)

# Renomme des colonnes
df=df.rename(columns = {'montant': 'montantTotal', 'distanceAcheteurEtablissement': 'distanceMoyenne', 'duree': 'dureeMoyenne', 
                     'type_Contrat de concession': 'nbContratDeConcession', 'type_Marché': 'nbMarché'})

#... Mettre les valeurs sur une même unité de mesure
df_nom = pd.DataFrame(df.libelleCommuneAcheteur)
del df['libelleCommuneAcheteur']
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

#... On réassemble le df
df = df_nom.join(df)
del df_nom

###############################################################################
### Application de l'algorithme de classification ascendante hiérarchique - CAH
############ Avec les données normalisée
# Générer la matrice des liens
Z = linkage(scaled_df, method='ward', metric='euclidean')
# Dendrogramme
plt.title('CAH avec matérialisation des X classes')
dendrogram(Z,labels=df.index,orientation='left',color_threshold=65)
plt.show()
# Récupération des classes
groupes_cah = pd.DataFrame(fcluster(Z,t=65,criterion='distance'), columns = ['segmentation_CAH'])
### Ajout au df 
df = df.join(groupes_cah)
del Z, groupes_cah, scaled_df

# On créé une 4e catégorie avec toutes les valeurs seules
df.reset_index(inplace=True)
a=pd.DataFrame(df.groupby('segmentation_CAH')['index'].nunique())
a.reset_index(inplace=True); a.columns=['cluster', 'nb']
a=a.sort_values(by = 'nb', axis = 0, ascending = False)
a.reset_index(inplace=True, drop=True)
a=a.drop([0,1,2]); a = list(a.cluster)
# On remplace
df['segmentation_CAH']=df['segmentation_CAH'].replace(a, 0); del a
df.segmentation_CAH = df.segmentation_CAH.astype(int)

# Changement / TRI des clusters
cahORDER =pd.DataFrame(df.groupby('segmentation_CAH')[['montantTotal', 'segmentation_CAH']].mean())
cahORDER = cahORDER.sort_values(by = 'montantTotal')
cahORDER = cahORDER[cahORDER.segmentation_CAH!=0]
l = ['0'] + list(cahORDER.segmentation_CAH.unique())
k = [0, 1, 2, 3]; listCorrespondance = {x:y for x,y in zip(k, l)}
for word, initial in listCorrespondance.items():
    df['segmentation_CAH'] = np.where(df['segmentation_CAH'] == initial, word, df['segmentation_CAH'])
del l,k,listCorrespondance

# On ajoute au dataframe principal
df=df[['libelleCommuneAcheteur','segmentation_CAH']]
df_decp = pd.merge(df_decp, df, how='left', on='libelleCommuneAcheteur')
df_decp.segmentation_CAH = np.where(df_decp.segmentation_CAH.isnull(), 0, df_decp.segmentation_CAH)
df_decp.segmentation_CAH = df_decp.segmentation_CAH.astype(int)

###############################################################################
############........ CARTE DES MARCHES PAR VILLE
df_carte = df_decp[['latitudeAcheteur', 'longitudeAcheteur', 'libelleCommuneAcheteur']]
df_carte=df_carte[df_carte['latitudeAcheteur'] != 'nan']
df_carte=df_carte[df_carte['longitudeAcheteur'] != 'nan']
df_carte = df_carte.drop_duplicates(subset=['latitudeAcheteur', 'longitudeAcheteur'], keep='first')
df_carte.reset_index(inplace=True, drop=True)

dfMT = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).montant.sum().to_frame('montantTotal').reset_index()
dfMM = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).montant.mean().to_frame('montantMoyen').reset_index()
dfIN = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).identifiantMarche.nunique().to_frame('nbMarches').reset_index()
dfSN = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).siretEtablissement.nunique().to_frame('nbEntreprises').reset_index()
dfDM = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).distanceAcheteurEtablissement.median().to_frame('distanceMediane').reset_index()

df_carte = pd.merge(df_carte, dfMT, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfMM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfIN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfSN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfDM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, df, how='left', on=['libelleCommuneAcheteur'])
del dfMM, dfMT, dfIN, dfSN, dfDM

df_carte.montantTotal = round(df_carte.montantTotal, 0)
df_carte.montantMoyen = round(df_carte.montantMoyen, 0)
df_carte.nbMarches = round(df_carte.nbMarches, 0)
df_carte.nbEntreprises = round(df_carte.nbEntreprises, 0)
df_carte.distanceMediane = round(df_carte.distanceMediane, 0)
df_carte.distanceMediane = np.where(df_carte.distanceMediane.isnull(), 0, df_carte.distanceMediane)

###############################################################################
### Carte des DECP
geojson = json.loads(urllib.request.urlopen('https://france-geojson.gregoiredavid.fr/repo/regions.geojson').read())
df_Reg = df_decp.groupby(['codeRegionAcheteur']).montant.sum().to_frame('montantMoyen').reset_index()
df_Reg.columns = ['code','montant']; df_Reg = df_Reg[(df_Reg.code!='nan') & (df_Reg.code!='98')]
df_Reg.montant=round(df_Reg.montant/1000000,0).astype(int)
df_Reg.montant = np.where(df_Reg.montant>10000, 10000, df_Reg.montant)

depPop = pd.read_csv("dataEnrichissement/departements-francais.csv", sep='\t', encoding='utf-8', usecols=['NUMÉRO', 'POPULATION'])
depPop.columns= ['code','population']; depPop.code = depPop.code.astype(str); depPop=depPop[depPop.population.notnull()]; depPop.population = depPop.population.astype(int)
for i in range(len(depPop)):
    if len(depPop['code'][i])<2:
        depPop['code'][i] = '0' + depPop['code'][i]

geojson2 = json.loads(urllib.request.urlopen('https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-avec-outre-mer.geojson').read())
df_Dep = df_decp.groupby(['codeDepartementAcheteur']).montant.sum().to_frame('montantMoyen').reset_index()
df_Dep.columns = ['code','montant']; df_Dep = df_Dep[(df_Dep.code!='nan')]
df_Dep = pd.merge(df_Dep, depPop, how='left', on='code'); df_Dep = df_Dep[df_Dep.population.notnull()]
df_Dep.montant = round(df_Dep.montant/df_Dep.population,0).astype(int)
del df_Dep['population']
df_Dep.montant = np.where(df_Dep.montant>2000, 2000, df_Dep.montant)

dfHM = df_decp[['latitudeAcheteur', 'longitudeAcheteur']]
dfHM = dfHM[(dfHM.latitudeAcheteur!='nan') | (dfHM.longitudeAcheteur!='nan')]

### Mise en forme
c= folium.Map(location=[47, 2.0],zoom_start=6, control_scale = True)
plugins.MiniMap(toggle_display=True).add_to(c)

mapMarker = folium.Marker([44,-4],icon=folium.features.CustomIcon('https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Information_icon.svg/1000px-Information_icon.svg.png', icon_size=(20,20)),
              popup = folium.Popup('<b>Indicateur de distance</b></br></br>' +
                                   '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=8 height=8/>'+ ' ' +
                                   '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=14 height=14/>'+ ' ' +
                                   '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=20 height=20/> : Distance moyenne</br>entre acheteurs et entreprises' + '</br></br>' + 

                                   '<b>Ségmentation des acheteurs </b></br></br>'+
                                   '<img src="https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg"  width=20 height=20/> : Petit' + '</br>' + 
                                   '<img src="https://cdn1.iconfinder.com/data/icons/vibrancie-map/30/map_001-location-pin-marker-place-512.png"  width=20 height=20/> : Moyen' + '</br>' + 
                                   '<img src="https://cdn.cnt-tech.io/api/v1/tenants/dd1f88aa-e3e2-450c-9fa9-a03ea59a6bf0/domains/57a9d53a-fe30-4b6f-a4de-d624bd25134b/buckets/8f139e2f-9e74-4be3-9d30-d8f180f02fbb/statics/56/56d48498-d2bf-45f8-846e-6c9869919ced"  width=20 height=20/> : Grand' + '</br>' + 
                                   '<img src="https://svgsilh.com/svg/157354.svg"  width=20 height=20/> : Hors-segmentation',
                                   max_width=320, show=True), overlay=False).add_to(c)

marker_cluster = MarkerCluster(name='DECP par communes').add_to(c)
for i in range(len(df_carte)):
    if (df_carte.segmentation_CAH[i]==1):
        icon = folium.features.CustomIcon('https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg', icon_size=(max(20, min(40,df_carte.distanceMediane[i]/2)), max(20, min(40,df_carte.distanceMediane[i]/2))))
    elif (df_carte.segmentation_CAH[i]==2):
        icon = folium.features.CustomIcon('https://cdn1.iconfinder.com/data/icons/vibrancie-map/30/map_001-location-pin-marker-place-512.png', icon_size=(max(20, min(40,df_carte.distanceMediane[i]/2)), max(20, min(40,df_carte.distanceMediane[i]/2))))
    elif (df_carte.segmentation_CAH[i]==3):
        icon = folium.features.CustomIcon('https://cdn.cnt-tech.io/api/v1/tenants/dd1f88aa-e3e2-450c-9fa9-a03ea59a6bf0/domains/57a9d53a-fe30-4b6f-a4de-d624bd25134b/buckets/8f139e2f-9e74-4be3-9d30-d8f180f02fbb/statics/56/56d48498-d2bf-45f8-846e-6c9869919ced', icon_size=(max(20, min(40,df_carte.distanceMediane[i]/2)), max(20, min(40,df_carte.distanceMediane[i]/2))))
    else :
        icon = folium.features.CustomIcon('https://svgsilh.com/svg/157354.svg', icon_size=(max(20, min(40,df_carte.distanceMediane[i]/2)), max(20, min(40,df_carte.distanceMediane[i]/2))))

    folium.Marker([df_carte.latitudeAcheteur[i],  df_carte.longitudeAcheteur[i]], 
                  icon=icon, popup = folium.Popup('<b><center>' + df_carte.libelleCommuneAcheteur[i] + '</center></b></br>'
                  + '<b>' + df_carte.nbMarches[i].astype(str) + '</b> marchés '
                  + 'pour un montant moyen de <b>' + round(df_carte.montantMoyen[i]/1000,0).astype(int).astype(str) + ' mille euros</b> '
                  + "</br>avec <b>" + df_carte.nbEntreprises[i].astype(str) + ' entreprises</b> '
                  + "à une distance médiane de <b>" + df_carte.distanceMediane[i].astype(int).astype(str) + ' km</b> ',
                  max_width = 320, min_width = 200), 
                  tooltip=folium.Tooltip(df_carte.libelleCommuneAcheteur[i]), clustered_marker = True).add_to(marker_cluster)

HeatMap(data=dfHM[['latitudeAcheteur', 'longitudeAcheteur']], radius=10, name="HeatMap des marchés", show=False,overlay=False).add_to(c)

choropleth = folium.Choropleth(geo_data=geojson, name='Régions', data=df_Reg, columns=['code', 'montant'],
    key_on='feature.properties.code', fill_color= 'YlGnBu', fill_opacity=0.7, line_opacity=0.2, nan_fill_color='#8c8c8c',
    highlight=True, line_color='black', show=False,overlay=False, legend_name= "Montant total des marchés (en millions €)").add_to(c)
choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['nom'],labels=False))

choropleth = folium.Choropleth(geo_data=geojson2, name='Départements', data=df_Dep, columns=['code', 'montant'],
    key_on='feature.properties.code', fill_color= 'YlOrRd', fill_opacity=0.7, line_opacity=0.2, nan_fill_color='#8c8c8c',
    highlight=False, line_color='black', show=False,overlay=False, legend_name= "Montant total par habitants (en milliers €)").add_to(c)
choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['nom'],labels=False))

folium.TileLayer('OpenStreetMap', overlay=True, show=True, control =False).add_to(c)
folium.LayerControl(collapsed=False).add_to(c)
c.save('carte/carteDECP.html')

###############################################################################
##### Algorithme de Luhn 
def luhn(codeSIREN):
    try:
        chiffres = pd.DataFrame(map(int, list(str(codeSIREN))), columns=['siren'])        
        chiffres['parite']=[1,2,1,2,1,2,1,2,1]
        chiffres['multiplication']=chiffres.siren*chiffres.parite
        for i in range(len(chiffres)):
            chiffres.multiplication[i]=sum([int(c) for c in str(chiffres.multiplication[i])])
        resultat= chiffres.multiplication.sum()
        if (resultat % 10)==0:
            resultat=0 # code BON
        else:
            resultat=1 # code FAUX
    except:
        resultat=1 # code FAUX
        pass
    return resultat

# Application sur les siren des acheteurs
df_decp['siren1Acheteur'] = df_decp.acheteurId.str[:9]
df_SA = pd.DataFrame(df_decp['siren1Acheteur'])
df_SA = df_SA.drop_duplicates(subset=['siren1Acheteur'], keep='first')
df_SA['verifSirenAcheteur'] = df_SA['siren1Acheteur'].apply(luhn)
# Application sur les siren des établissements
df_decp['siren2Etablissement'] = df_decp.sirenEtablissement.str[:9]
df_SE = pd.DataFrame(df_decp['siren2Etablissement'])
df_SE = df_SE.drop_duplicates(subset=['siren2Etablissement'], keep='first')
df_SE['verifSirenEtablissement'] = df_SE['siren2Etablissement'].apply(luhn)
# Merge avec le df principal
df_decp = pd.merge(df_decp, df_SA, how='left', on = 'siren1Acheteur')
df_decp = pd.merge(df_decp, df_SE, how='left', on = 'siren2Etablissement')
del df_decp['siren1Acheteur'], df_decp['siren2Etablissement']
# On rectifie pour les codes non-siret
df_decp.verifSirenEtablissement = np.where((df_decp.typeIdentifiantEtablissement!='SIRET') | (df_decp.typeIdentifiantEtablissement.isnull()), 0, df_decp.verifSirenEtablissement)

###############################################################################
###############################################################################
### Ratio nb entreprises / nb marchés
df_carte['ratioEntreprisesMarchés']=df_carte['nbEntreprises']/df_carte['nbMarches']
df_bar = df_carte[['libelleCommuneAcheteur', 'nbMarches', 'ratioEntreprisesMarchés']]
df_bar = df_bar[(df_bar.nbMarches>100) & (df_bar.ratioEntreprisesMarchés>0)]
df_bar = df_bar.sort_values(by = 'ratioEntreprisesMarchés').sort_values(by = 'ratioEntreprisesMarchés', ascending = True)
# Graphique des résultats : top 10
df_barGraph = df_bar.head(10)
df_barGraph.ratioEntreprisesMarchés.plot(kind='barh', title='Top 10 des communes avec le plus petit ratio NBentreprise/NBmarchés')
plt.yticks(range(0,len(df_barGraph.libelleCommuneAcheteur)), df_barGraph.libelleCommuneAcheteur)
del df_barGraph
round(df_bar.ratioEntreprisesMarchés.mean(),2)

### Ratio nb entreprises / nb marchés > 40K
df_ratio = pd.DataFrame.copy(df_decp, deep = True); 
df_ratio = df_ratio[df_ratio.montantTotalMarche>=40000]
df_ratio_marche = df_ratio.groupby(['libelleCommuneAcheteur']).identifiantMarche.nunique().to_frame('nbMarches').reset_index()
df_ratio_entreprises = df_ratio.groupby(['libelleCommuneAcheteur']).siretEtablissement.nunique().to_frame('nbEntreprises').reset_index()
df_ratio = pd.merge(df_ratio_marche, df_ratio_entreprises, how='inner', on='libelleCommuneAcheteur')
df_ratio['ratioEntreprisesMarchés']=df_carte['nbEntreprises']/df_carte['nbMarches']
df_ratio = df_ratio[(df_ratio.nbMarches>100) & (df_ratio.ratioEntreprisesMarchés>0)]
df_ratio = df_ratio.sort_values(by = 'ratioEntreprisesMarchés')
# Graphique des résultats : top 10
df_barGraph = df_ratio.head(20)
df_barGraph.ratioEntreprisesMarchés.plot(kind='barh', title='Top 10 des communes avec le plus petit ratio NBentreprise/NBmarchés')
plt.yticks(range(0,len(df_barGraph.libelleCommuneAcheteur)), df_barGraph.libelleCommuneAcheteur)
del df_barGraph
round(df_ratio.ratioEntreprisesMarchés.mean(),2) # Moyenne à 0.9

df_bar.to_csv(r'resultatsCSV/df_Ratio.csv', sep=';',index = False, header=True, encoding='utf-8')
df_bar.to_csv(r'resultatsCSV/df_Ratio40K.csv', sep=';',index = False, header=True, encoding='utf-8')

###############################################################################
### Récap des erreurs
df_ERROR = df_decp[(df_decp.montantEstEstime=='Oui') | (df_decp.dureeMoisEstEstime=='Oui') 
                    | ((df_decp.verifSirenAcheteur==1) | (df_decp.acheteurId=='00000000000000'))  
                    | ((df_decp.verifSirenEtablissement==1) & (df_decp.typeIdentifiantEtablissement=='SIRET'))
                    | ((df_decp.verifSirenEtablissement==1) & (df_decp.typeIdentifiantEtablissement.isnull()))
                    | ((df_decp.idEtablissement=='00000000000000') & (df_decp.typeIdentifiantEtablissement=='SIRET'))
                    | ((df_decp.idEtablissement=='00000000000000') & (df_decp.typeIdentifiantEtablissement.isnull()))]

df_ERROR = df_ERROR[['source', 'identifiantMarche','objetMarche', 'acheteurId','acheteurNom', 
                     'idEtablissement', 'montantOriginal',  'dureeMois','typeIdentifiantEtablissement',
                     'montantEstEstime', 'dureeMoisEstEstime', 'verifSirenAcheteur', 'verifSirenEtablissement']] 
(df_ERROR.montantEstEstime=='Oui').sum()
(df_ERROR.dureeMoisEstEstime=='Oui').sum()
((df_ERROR.verifSirenAcheteur==1) |(df_ERROR.acheteurId=='00000000000000')).sum()
((df_ERROR.verifSirenEtablissement==1) |(df_ERROR.idEtablissement=='00000000000000')).sum()


df_ERROR.columns = ['source', 'identifiantMarche','objetMarche', 'acheteurId','acheteurNom', 'EtablissementID',
                     'montantOriginal', 'dureeMoisOriginal', 'montantAberrant', 'dureeMoisAberrant',
                     'siretAcheteur', 'siretEtablissement']
df_ERROR.source.value_counts(normalize=True).plot(kind='pie') # Aucune corrélation entre les erreurs et les sources

df_Classement = pd.DataFrame.copy(df_ERROR, deep = True)
df_Classement = df_Classement[['acheteurNom', 'montantAberrant', 'dureeMoisAberrant', 'siretAcheteur', 'siretEtablissement']]
df_Classement.columns = ['acheteurNom', 'montantEstEstime', 'dureeMoisEstEstime', 'siretAcheteur', 'siretEtablissement']
df_Classement.montantEstEstime = np.where(df_Classement.montantEstEstime=='Oui',1,0)
df_Classement.dureeMoisEstEstime = np.where(df_Classement.dureeMoisEstEstime=='Oui',1,0)

df_Classement = df_Classement.groupby(['acheteurNom']).sum().reset_index()
df_50 = pd.DataFrame(df_Classement[(df_Classement.montantEstEstime >= 50) |
        (df_Classement.dureeMoisEstEstime >= 300) |
        (df_Classement.siretAcheteur >= 180) |
        (df_Classement.siretEtablissement >= 50)])
df_50['Note']=df_50.montantEstEstime*4+df_50.dureeMoisEstEstime*1+df_50.siretAcheteur*1+df_50.siretEtablissement*2
df_50=df_50.sort_values(by = 'Note', ascending = False)
del df_50['Note']

#siretEtablissement
Bilan=pd.DataFrame(df_Classement.sum()[1:5]).T; Bilan.columns=['Montant aberrant ','Durée en mois aberrante ','Siret acheteur mauvais ','Siret entreprise mauvais ']

# Les pires lignes (4 erreurs): 
F = df_ERROR[(df_ERROR.montantAberrant=='Oui') & (df_ERROR.dureeMoisAberrant=='Oui') & (df_ERROR.siretAcheteur=='MAUVAIS') & (df_ERROR.siretEtablissement=='MAUVAIS')]

# Liste de tous les acheteurs ayant fait au moins 10 supposées erreurs :
df_Classement['Total'] = df_Classement.montantEstEstime + df_Classement.dureeMoisEstEstime + df_Classement.siretAcheteur + df_Classement.siretEtablissement
ListeMauvaixAcheteurs = pd.DataFrame(np.array([df_Classement.acheteurNom[df_Classement['Total']>10].unique()]).T, columns=['Acheteur'])

###############################################################################
### Rapide aperçu des données principales
# Aperçu répartition des sources
round(df_decp.source.value_counts(normalize=True)*100,2) # pourcentage des sources
df_decp.source.value_counts(normalize=True).plot(kind='pie')

# Recapitulatif quantitatif
df_RECAP = pd.concat([df_decp.montantOriginal.describe(),
                      df_decp.montant.describe(),
                      df_decp.dureeMois.describe(),
                      df_decp.dureeMoisCalculee.describe(),
                      df_decp.distanceAcheteurEtablissement.describe()], axis=1)
df_RECAP.columns=['Montant original (€)', 'Montant calculé (€)', 'Durée en mois originale', 'Durée en mois calculée','Distance acheteur - établissement (km)']
df_RECAP = df_RECAP[1:8]

# Récupération sous format CSV
df_ERROR.to_csv(r'resultatsCSV/df_ERROR.csv', sep=';',index = False, header=True, encoding='utf-8')
ListeMauvaixAcheteurs.to_csv(r'resultatsCSV/ListeMauvaixAcheteurs.csv', sep=';',index = False, header=True, encoding='utf-8')
df_50.columns = ['acheteurNom', 'montantAberrant', 'dureeMoisAberrant', 'siretAcheteurFAUX', 'siretEtablissementFAUX']
df_50.to_csv(r'resultatsCSV/df_50.csv', sep=';',index = False, header=True, encoding='utf-8')

###############################################################################
del F, ListeMauvaixAcheteurs, df_ERROR, df_RECAP, df_50, df_Classement, Bilan
del depPop, df, dfHM, df_Dep, df_Reg, df_carte, geojson, geojson2, i

'''
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Exportation des données 
#df.dtypes
df_decp.to_csv(r'decp.csv', sep=';',index = False, header=True, encoding='utf-8')
 
# Réimportation des données
df_decp = pd.read_csv('H:/Desktop/Data/decp.csv', sep=';', encoding='utf-8', 
                      dtype={'acheteurId' : str, 'nicEtablissement' : str, 'codeRegionAcheteur' : str, 'denominationSocialeEtablissement' : str,
                             'moisNotification' : str,  'idEtablissement' : str, 'montantOriginal' : float, 'montant' : float, 'montantTotalMarche' : float, 'codeDepartementAcheteur' : str,
                             'anneeNotification' : str, 'codeCommuneEtablissement' : str, 'codePostalEtablissement' : str,  'identifiantMarche' : str,
                             'codeTypeEtablissement' : str, 'sirenEtablissement' : str, 'siretEtablissement' : str, 'codeCPV' : str,
                             'nbTitulairesSurCeMarche' : int, 'dureeMois': int, 'dureeMoisCalculee': int, 'codeCommuneAcheteur': str, 'codePostalAcheteur': str})
#### Comparaison de toutes les autres colonnes
dftest = df.drop(columns=['formePrix', 'denominationSocialeEtablissement'])
dftest_copy = df.drop(columns=['formePrix' , 'denominationSocialeEtablissement'])
try:
    assert_frame_equal(dftest, dftest_copy)
    print(True)
except:
    print(False)
'''