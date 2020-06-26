# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08
@author: Lucas GEFFARD
"""
######################################################################
import pandas as pd
import numpy as np
#import math
import json
from pandas.io.json import json_normalize #Pour corriger les données Json imbriquées
#%matplotlib inline
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import seaborn as sns
import scipy.stats as st
from lxml import html
import requests
######################################################################
#Chargement des données
chemin = "H:/Desktop/Data/Json/fichierPrincipal/decp.json"
with open(chemin, encoding='utf-8') as json_data:
    data = json.load(json_data)
#Aplatir les données Json imbriquées
df = json_normalize(data['marches'])
#test = json_normalize(data, record_path = ['marches', 'modifications'])

#Autre solution
"""
lien = "H:/Desktop/Data/Json/fichierPrincipal/decp.json"
test = pd.read_json(path_or_buf=lien, orient='index', typ='series', dtype=True,
                 convert_axes=False, convert_dates=True, keep_default_dates=True, 
                 numpy=False, precise_float=False, date_unit=None, encoding="utf-8", 
                 lines=False, chunksize=None, compression=None) #lines tester True et chunk
test = json_normalize(test['marches'])
"""
######################################################################

#Vision rapide df
df.head(5)
#Quelques informations sur les variables quanti
df.describe()
#Nombre de données par variable
df.info()

######################################################################
#............... Quelques stats descriptives AVANT nettoyages des données
dfStat = pd.DataFrame.copy(df, deep = True)
#Différentes sources des datas
dfStat["source"].value_counts(normalize=True).plot(kind='pie') #data.gouv.fr_aife - Marches-public.info
plt.xlabel('')
plt.ylabel('')
plt.title("Source des marchés\n", fontsize=18, color='#3742fa')
#Type de contrat
dfStat["_type"].value_counts(normalize=True).plot(kind='pie') #Marché : 127 574
plt.xlabel('')
plt.ylabel('')
plt.title("Type de marchés\n", fontsize=18, color='#3742fa')
          
#Nature des contrats
dfNature = pd.DataFrame(dfStat.groupby('nature')['uid'].nunique())
dfNature.reset_index(level=0, inplace=True) 
dfNature = dfNature.sort_values(by = 'uid')
sns.barplot(x=dfNature['nature'], y=dfNature['uid'], palette="Blues_r")
plt.xlabel('\nNature', fontsize=15, color='#2980b9')
plt.ylabel('Nombre de contrats\n', fontsize=15, color='#2980b9')
plt.title("Nature des contrats\n", fontsize=18, color='#3742fa')
plt.xticks(rotation= 45)
plt.tight_layout()
#On enlève les 3 plus importants afin de voir le reste
dfNature.drop(dfNature.tail(3).index,inplace=True)
sns.barplot(x=dfNature['nature'], y=dfNature['uid'], palette="Blues_r")
plt.xlabel('\nNature', fontsize=15, color='#2980b9')
plt.ylabel('Nombre de contrats\n', fontsize=15, color='#2980b9')
plt.title("Nature des contrats\n", fontsize=18, color='#3742fa')
plt.xticks(rotation= 45)
plt.tight_layout()

#Montants en fonction des sources - Moyenne ! Pas somme, si sum -> Aife  
sns.barplot(x=dfStat['source'], y=dfStat['montant'], palette="Reds_r")
plt.xlabel('\nSources', fontsize=15, color='#c0392b')
plt.ylabel("Montant\n", fontsize=15, color='#c0392b')
plt.title("Montant moyen en fonction des sources\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()
#Montants en fonction des jours (date)
dfStat['montant'].fillna(0, inplace=True)
GraphDate = pd.DataFrame(dfStat.groupby('datePublicationDonnees')['montant'].sum())
GraphDate = GraphDate[GraphDate['montant'].between(10, 1.0e+8)] #Suppression des montants exhorbitants
GraphDate.plot()
plt.xlabel('\nJour', fontsize=15, color='#3742fa')
plt.ylabel("Montant\n", fontsize=15, color='#3742fa')
plt.title("Montants par jour\n", fontsize=18, color='#3742fa')

#df.boxplot(column=df['dureeMois'])
#df.boxplot(column=df['montant'], by='source')
plt.boxplot(GraphDate['montant']) #Boite à moustache des montants
plt.title("Représentation des montants\n", fontsize=18, color='#3742fa')
#plt.show()
sum(GraphDate['montant'])/len(GraphDate['montant']) #Moyenne par jour : 273M à 15M
sum(df['montant'])/len(df['montant']) #Moyenne par achat : 3 143 354
df['montant'].describe() #Médiane : 71 560 !!!
df['dureeMois'].describe() #Durée des contrats
del [dfStat, dfNature, GraphDate]

######################################################################
######################################################################
#...............    Nettoyage/formatage des données

################### Identifier et supprimer les doublons -> environ 6000
df = df.drop_duplicates(subset=['source', '_type', 'nature', 'dureeMois',
                           'dateSignature', 'datePublicationDonnees',
                           'dateDebutExecution', 'valeurGlobale',
                           'lieuExecution.code', 'lieuExecution.typeCode',
                           'lieuExecution.nom', 'objet','dateNotification',
                           'montant', 'acheteur.id'], keep='first')
# Impossible de tester avec ces colonnes (car listes) :
# 'titulaires', 'concessionnaires', 'donneesExecution',
    
#Il faudrait peut-être réduire le nombre de variable pour définir les doublons
#ce qui permettrait d'en supprimer d'avantage (mais risque de perte de données)

# Correction afin que ces variables soient représentées pareil    
df['formePrix'] = np.where(df['formePrix'] == 'Ferme, actualisable', 'Ferme et actualisable', df['formePrix'])
    
################### Identifier les outliers - travail sur les montants
### Valeur aberrantes ou valeurs atypiques ?
#Suppression des variables qui n'auraient pas du être présentes
df['montant'] = np.where(df['montant'] <= 1, np.NaN, df['montant']) #Identification des erreurs avant
df = df[(df['montant'] >= 40000) | (df['montant'].isnull())] #Suppression des montants < 40 000
#Après avoir analysé cas par cas les données extrêmes, la barre des données
#que l'on peut considérées comme aberrantes est placée au milliard d'€
df['montant'] = np.where(df['montant'] >= 9.99e8, np.NaN, df['montant']) #Suppression des valeurs aberrantes
#Il reste toujours des valeurs extrêmes mais elles sont vraisemblables
#(surtout lorsque l'on sait que le budget annuel d'une ville comme Paris atteint les 10 milliards)
GraphDate = pd.DataFrame(df.groupby('datePublicationDonnees')['montant'].sum())
plt.boxplot(GraphDate['montant'])
#Vision des montants après néttoyage
df.montant.describe()

################### Régions / Départements ##################
# Création de la colonne pour distinguer les départements
df['lieuExecution.code'] = df['lieuExecution.code'].astype(str)
df['codePostal'] = df['lieuExecution.code'].str[:3]
df['codePostal'] = np.where(df['codePostal'] == '976', 'YT', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == '974', 'RE', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == '972', 'MQ', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == '971', 'GP', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == '973', 'GF', df['codePostal'])

df['codePostal'] = df['codePostal'].str[:2]
# Remplacement des régions outre-mer YT - RE - ... par 97 ou 98
df['codePostal'] = np.where(df['codePostal'] == 'YT', '976', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'RE', '974', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'PM', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'MQ', '972', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'MF', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'GP', '971', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'GF', '973', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'BL', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'WF', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'TF', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'PF', '98', df['codePostal'])
df['codePostal'] = np.where(df['codePostal'] == 'NC', '98', df['codePostal'])

# Vérification si c'est bien un code postal
listeCP = ['01','02','03','04','05','06','07','08','09','10',
           '11','12','13','14','15','16','17','18','19','20',
           '21','22','23','24','25','26','27','28','29','30',
           '31','32','33','34','35','36','37','38','39','40',
           '41','42','43','44','45','46','47','48','49','50',
           '51','52','53','54','55','56','57','58','59','60',
           '61','62','63','64','65','66','67','68','69','70',
           '71','72','73','74','75','76','77','78','79','80',
           '81','82','83','84','85','86','87','88','89','90',
           '91','92','93','94','95','2A','2B','98',
           '976','974','972','971','973','97']
def check_cp(codePostal):
    if codePostal not in listeCP:
        return np.NaN
    return codePostal
df['codePostal'] = df['codePostal'].apply(check_cp)
#Suppression des codes régions (qui sont retenues jusque là comme des codes postaux)
df['codePostal'] = np.where(df['lieuExecution.typeCode'] == 'Code région', np.NaN, df['codePostal'])

# Création de la colonne pour distinguer les régions
df['codeRegion'] = df['codePostal']
df['codeRegion'] = df['codeRegion'].astype(str)
# Définition des codes des régions en fonctions des codes de départements
liste84 = ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74']
liste27 = ['21', '25', '39', '58', '70', '71', '89', '90']
liste53 = ['35', '22', '56', '29']
liste24 = ['18', '28', '36', '37', '41', '45']
liste94 = ['2A', '2B', '20']
liste44 = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
liste32 = ['02', '59', '60', '62', '80']
liste11 = ['75', '77', '78', '91', '92', '93', '94', '95']
liste28 = ['14', '27', '50', '61', '76']
liste75 = ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87']
liste76 = ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82']
liste52 = ['44', '49', '53', '72', '85']
liste93  =['04', '05', '06', '13', '83', '84']
# Fonction pour attribuer les codes régions
def code_region(codeRegion): 
    if codeRegion in liste84: return '84'
    if codeRegion in liste27: return '27'
    if codeRegion in liste53: return '53'
    if codeRegion in liste24: return '24'
    if codeRegion in liste94: return '94'
    if codeRegion in liste44: return '44'
    if codeRegion in liste32: return '32'
    if codeRegion in liste11: return '11'
    if codeRegion in liste28: return '28'
    if codeRegion in liste75: return '75'
    if codeRegion in liste76: return '76'
    if codeRegion in liste52: return '52'
    if codeRegion in liste93: return '93'
    return codeRegion
df['codeRegion'] = df['codeRegion'].apply(code_region)

# Mise à jour des codes des régions outres mer
df['codeRegion'] = np.where(df['codeRegion'] == "976", "06", df['codeRegion'])
df['codeRegion'] = np.where(df['codeRegion'] == "974", "04", df['codeRegion'])
df['codeRegion'] = np.where(df['codeRegion'] == "972", "02", df['codeRegion'])
df['codeRegion'] = np.where(df['codeRegion'] == "971", "01", df['codeRegion'])
df['codeRegion'] = np.where(df['codeRegion'] == "973", "03", df['codeRegion'])
df['codeRegion'] = np.where(df['codeRegion'] == "97", "98", df['codeRegion'])
# Ajout des codes régions qui existaient déjà dans la colonne lieuExecution.code
df['codeRegion'] = np.where(df['lieuExecution.typeCode'] == "Code région", df['lieuExecution.code'], df['codeRegion'])
df['codeRegion'] = df['codeRegion'].astype(str)
# Vérification des codes région 
listeReg = ['84', '27', '53', '24', '94', '44', '32',
            '11', '28', '75', '76', '52', '93', 
            '01', '02', '03', '04', '06', '98'] #98 = collectivité d'outre mer
def check_reg(codeRegion):
    if codeRegion not in listeReg:
        return np.NaN
    return codeRegion
df['codeRegion'] = df['codeRegion'].apply(check_reg)
#df['codeRegion'].describe()

# Création de la colonne pour le nom des régions
df['Region'] = df['codeRegion']
df['Region'] = df['Region'].astype(str)
# Lien entre codeRegion et nomRegion 
def nom_region(Region): 
    if Region == '84' : return 'Auvergne-Rhône-Alpes'
    if Region == '27' : return 'Bourgogne-Franche-Comté'
    if Region == '53' : return 'Bretagne'
    if Region == '24' : return 'Centre-Val de Loire'
    if Region == '94' : return 'Corse'
    if Region == '44' : return 'Grand Est'
    if Region == '32' : return 'Hauts-de-France'
    if Region == '11' : return 'Île-de-France'
    if Region == '28' : return 'Normandie'
    if Region == '75' : return 'Nouvelle-Aquitaine'
    if Region == '76' : return 'Occitanie'
    if Region == '52' : return 'Pays de la Loire'
    if Region == '93' : return 'Provence-Alpes-Côte d\'Azur'
    if Region == '01' : return 'Guadeloupe'
    if Region == '02' : return 'Martinique'
    if Region == '03' : return 'Guyane'
    if Region == '04' : return 'La Réunion'
    if Region == '06' : return 'Mayotte'
    if Region == '98' : return 'Collectivité d\'outre mer'
    return Region;
df['Region'] = df['Region'].apply(nom_region)
#df['Region'].describe()
#del [liste11, liste24, liste27, liste28, liste32, liste44, liste52, liste53, liste75, liste76, liste84, liste93, liste94]

################### Date / Temps ##################
#..............Les différents types 
#Duree
df['dureeMois'].describe() # Duree également en jours...
#Dates utiles
df['datePublicationDonnees'].describe() # Date à laquelle les données sont saisies
df['dateNotification'].describe() # Réel intérêt, plus que datePublicationDonnees
#Dates utiles que pour les données concessionnaires 
df['dateDebutExecution'].describe() # 137
df['dateSignature'].describe() # 137

#..............Travail sur la variable de la durée des marchés
# Graph avec la médiane, très utile !
GraphDate = pd.DataFrame(df.groupby('dureeMois')['montant'].median())
GraphDate.reset_index(level=0, inplace=True)
GraphDate = GraphDate[GraphDate['dureeMois'].between(1, 38)] 
del GraphDate['dureeMois']
GraphDate.plot()
plt.plot([24, 24], [0, 250000], 'g-', lw=1) # 2 ans
plt.plot([30, 30], [0, 250000], 'r-', lw=1) # 30 jours
plt.plot([36, 36], [0, 250000], 'g-', lw=1) # 3 ans
#plt.plot([48, 48], [0, 400000], 'g-', lw=2) # 4 ans
plt.title("Montant médian par mois\n", fontsize=18, color='#000000')
plt.xlabel('\nDurée en mois', fontsize=15, color='#000000')
plt.ylabel("Montant\n", fontsize=15, color='#000000')
          
# Pourquoi le graph avec la moyenne ne donne rien d'utile
# --> très certainement car il y a des données extrêmes qui fausses tout...           
GraphDate = pd.DataFrame(df.groupby('dureeMois')['montant'].mean())
GraphDate.reset_index(level=0, inplace=True)
GraphDate = GraphDate[GraphDate['dureeMois'].between(1, 60)] 
GraphDate = GraphDate[GraphDate['montant'].between(10, 1.0e+6)] 
del GraphDate['dureeMois']
GraphDate.plot()
plt.plot([24, 24], [0, 2500000], 'g-', lw=1) # 2 ans
plt.plot([30, 30], [0, 2500000], 'r-', lw=1) # 30 jours
plt.plot([36, 36], [0, 2500000], 'g-', lw=1) # 3 ans
plt.title("Montant moyen par mois\n", fontsize=18, color='#000000')
plt.xlabel('\Durée en mois', fontsize=15, color='#000000')
plt.ylabel("Montant\n", fontsize=15, color='#000000')

           
#..............Travail sur les variables de type date
df.datePublicationDonnees.describe() 
df.dateNotification.describe()
           
df.datePublicationDonnees = df.datePublicationDonnees.str[0:10]
df.dateNotification = df.dateNotification.str[0:10] 
#On récupère l'année de notification
df['anneeNotification'] = df.dateNotification.str[0:4] 
df['anneeNotification'] = df['anneeNotification'].astype(float)
#On supprime les erreurs (0021 par exemple)
df['dateNotification'] = np.where(df['anneeNotification'] < 2000, np.NaN, df['dateNotification'])
df['anneeNotification'] = np.where(df['anneeNotification'] < 2000, np.NaN, df['anneeNotification'])
#On récupère le mois de notification
df['moisNotification'] = df.dateNotification.str[5:7] 

#Graphique pour voir les résultats
#... Médiane par année
GraphDate = pd.DataFrame(df.groupby('anneeNotification')['montant'].median())
GraphDate.plot()
plt.xlabel('\nAnnée', fontsize=15, color='#000000')
plt.ylabel("Montant\n", fontsize=15, color='#000000')
plt.title("Médiane des montants par année\n", fontsize=18, color='#3742fa')
#... Somme
GraphDate = pd.DataFrame(df.groupby('moisNotification')['montant'].sum())
GraphDate.plot()
plt.xlabel('\nMois', fontsize=15, color='#000000')
plt.ylabel("Montant\n", fontsize=15, color='#000000')
plt.title("Somme des montants par mois\n", fontsize=18, color='#3742fa')
#... Médiane
GraphDate = pd.DataFrame(df.groupby('moisNotification')['montant'].median())
GraphDate.plot()
plt.xlabel('\nMois', fontsize=15, color='#000000')
plt.ylabel("Montant\n", fontsize=15, color='#000000')
plt.title("Médiane des montants par mois\n", fontsize=18, color='#3742fa')

######################################################################
######################################################################          

# Vérification du nombre de nan dans les colonnes annee, mois, région, departement
df['moisNotification'].isnull().sum() #1601
df['anneeNotification'].isnull().sum() #1601
df['codePostal'].isnull().sum() #2876
(df['Region']=='nan').sum() #2495

######################################################################
#On décide de supprimer les lignes ou la variable région est manquante
#car ceux sont des données au niveau du pays ou internationales
df = df[df.Region != 'nan']

######################################################################

# Check du nombre de nan dans les montants          
df['montant'].isnull().sum() #1188
(df['montant']=='nan').sum() #0

########## Analysons les liens entre les variables et le montant
dfM = pd.DataFrame.copy(df, deep = True)
dfM = dfM[dfM['montant'].notnull()]
dfM = dfM[dfM['Region'].notnull()]
X = "Region"
Y = "montant"
sous_echantillon = dfM
def eta_squared(x, y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT;
eta_squared(sous_echantillon[X], sous_echantillon[Y])

dfM = pd.DataFrame.copy(df, deep = True)
dfM = dfM[dfM['montant'].notnull()]
dfM = dfM[dfM['moisNotification'].notnull()]
df['montant'] = df['montant'].astype(float)
df['moisNotification'] = df['moisNotification'].astype(float)
st.pearsonr(dfM['moisNotification'], dfM['montant'])[0]

del [X, Y, sous_echantillon, dfM]

'''
Résultats : 
source : 0.0026 (Rapport de corrélation - SCE/SCT)
_type : 0 (Rapport de corrélation - SCE/SCT)
nature : 0.0032 (Rapport de corrélation - SCE/SCT)
procedure : 0.0076 (Rapport de corrélation - SCE/SCT)
formePrix : 0.0017 (Rapport de corrélation - SCE/SCT)
codePostal : 0.0267 (Rapport de corrélation - SCE/SCT)
codeRegion : 0.0046 (Rapport de corrélation - SCE/SCT)
Region : 0.0046 (Rapport de corrélation - SCE/SCT)
anneeNotification : -0.0056 (Coefficient de corrélation - Pearson)
moisNotification : 0.0080 (Coefficient de corrélation - Pearson)

Conclusion : 
    A première vue aucune variable n'influe sur le montant
    Néanmoins, ces tests ne sont pas robustes face aux outliers
    Or, nos données en contiennent énormément.. 
    Donc, nous allons tenter de trouver des liens via d'autres moyens 
'''
#Recherchons des liens via des graphiques :
dfM = pd.DataFrame.copy(df, deep = True)
# On sélectionne la variable formePrix
dfM = dfM[dfM['formePrix'].notnull()]
plt.scatter(dfM['formePrix'], dfM['montant'])
dfM.groupby('formePrix')['montant'].median()
# Et la variable codeRegion 
plt.scatter(dfM['Region'], dfM['montant'])
dfM.groupby('Region')['montant'].median()

#dfM = dfM[dfM['montant'].between(0, 5000000)] #Suppression des montants exhorbitants
#plt.scatter(dfM['Region'], dfM['montant'])
#dfM.groupby('Region')['montant'].median()
## D'après les graph ces deux variables ont une influence sur le montant 
#..... Regardons le résultat de ces variables combinés
dfM = pd.DataFrame.copy(df, deep = True)
dfMnull = dfM[dfM['montant'].isnull()]
(dfMnull['formePrix']=='nan').sum() # 137, donc c'est normal !
# Quelques graphes :
dfM['Region'] = dfM['Region'].astype(str)
dfM['formePrix'] = dfM['formePrix'].astype(str)
dfM['reg_FP'] = dfM['Region'] + dfM['formePrix']
(dfM['reg_FP']=='nannan').sum() # 0 = tout est bon !
plt.scatter(dfM['reg_FP'], dfM['montant'])
medianeRegFP = dfM.groupby('reg_FP')['montant'].median()

# On retient : les variables Region et formePrix

##############################################################################
##############################################################################
#......... Méthode 1 : Suppression des valeurs manquantes
dfM1 = pd.DataFrame.copy(df, deep = True)
dfM1 = dfM1[dfM1['montant'].notnull()]
(1 - len(dfM1) / len(df)) * 100
# On perd 1.5% de données, donc cette méthode est à éviter

# Testons d'autres méthodes
listeM2 = []; listeM3 = []; listeM4 = []; listeM5 = []; listeM6 = []
for i in range(100):
    dfM = pd.DataFrame.copy(df, deep = True)
    dfM = dfM[dfM['montant'].notnull()]
    dfM['Region'] = dfM['Region'].astype(str)
    dfM['formePrix'] = dfM['formePrix'].astype(str)
    dfM['reg_FP'] = dfM['Region'] + dfM['formePrix']
    dfM['montantTest'] = dfM['montant']
    nb = (dfM['montantTest'].notnull()).sum()
    dfM.reset_index(level=0, inplace=True)
    dfM.reset_index(level=0, inplace=True)
    del dfM['index']
      
    for i in range(1200): 
        dfM['montantTest'] = np.where(dfM['level_0'] == np.random.randint(0,nb), np.NaN, dfM['montantTest'])
    
    dfM2 = pd.DataFrame.copy(dfM, deep = True)
    dfM3 = pd.DataFrame.copy(dfM, deep = True)
    dfM4 = pd.DataFrame.copy(dfM, deep = True)
    dfM5 = pd.DataFrame.copy(dfM, deep = True)
    
    # Methode 2
    dfM2['montantTest'] = np.where(dfM2['montantTest'].isnull(), dfM2['montantTest'].mean(), dfM2['montantTest'])
    dfM2['diffMontant'] = dfM2['montant'] - dfM2['montantTest']
    
    dfM2['diffMontant'] = dfM2['diffMontant'].abs()
    dfM2['diffMontant'] = np.where(dfM2['diffMontant'] == 0, np.NaN, dfM2['diffMontant'])
    listeM2 = listeM2 + [dfM2['diffMontant'].mean()]
    
    # Methode 3
    dfM3['montantTest'] = np.where(dfM3['montantTest'].isnull(), dfM3['montantTest'].median(), dfM3['montantTest'])
    dfM3['diffMontant'] = dfM3['montant'] - dfM3['montantTest']    
    
    dfM3['diffMontant'] = dfM3['diffMontant'].abs()
    dfM3['diffMontant'] = np.where(dfM3['diffMontant'] == 0, np.NaN, dfM3['diffMontant'])
    listeM3 = listeM3 + [dfM3['diffMontant'].mean()]
    
    # Methode 4
    moyenneRegFP = dfM4.groupby('reg_FP')['montantTest'].mean()
    moyenneRegFP = pd.DataFrame(moyenneRegFP)
    moyenneRegFP.reset_index(level=0, inplace=True)
    moyenneRegFP.columns = ['reg_FP','montantEstimation']
    dfM4 = pd.merge(dfM4, moyenneRegFP, on='reg_FP')    
    
    dfM4['montantTest'] = np.where(dfM4['montantTest'].isnull(), dfM4['montantEstimation'], dfM4['montantTest'])
    dfM4['diffMontant'] = dfM4['montant'] - dfM4['montantTest']
    dfM4['diffMontant'] = dfM4['diffMontant'].abs()
    dfM4['diffMontant'] = np.where(dfM4['diffMontant'] == 0, np.NaN, dfM4['diffMontant'])
    listeM4 = listeM4 + [dfM4['diffMontant'].mean()]
    
    # Methode 5
    medianeRegFP = dfM5.groupby('reg_FP')['montantTest'].median()
    medianeRegFP = pd.DataFrame(medianeRegFP)
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['reg_FP','montantEstimation']
    dfM5 = pd.merge(dfM5, medianeRegFP, on='reg_FP')

    dfM5['montantTest'] = np.where(dfM5['montantTest'].isnull(), dfM5['montantEstimation'], dfM5['montantTest'])
    dfM5['diffMontant'] = dfM5['montant'] - dfM5['montantTest']
    dfM5['diffMontant'] = dfM5['diffMontant'].abs()
    dfM5['diffMontant'] = np.where(dfM5['diffMontant'] == 0, np.NaN, dfM5['diffMontant'])
    listeM5 = listeM5 + [dfM5['diffMontant'].mean()]
    
    # Methode 6
    dfM6 = pd.DataFrame.copy(dfM, deep = True)
    dfM6['codePostal'] = dfM6['codePostal'].astype(str)
    dfM6['source'] = dfM6['source'].astype(str)
    dfM6['nature'] = dfM6['nature'].astype(str)
    dfM6['procedure'] = dfM6['procedure'].astype(str)
    dfM6['moisNotification'] = dfM6['moisNotification'].astype(str)
    dfM6['anneeNotification'] = dfM6['anneeNotification'].astype(str)
    dfM6['conca'] = dfM6['codePostal'] + dfM6['formePrix'] + dfM6['source'] + dfM6['nature'] + dfM6['procedure'] + dfM6['moisNotification'] + dfM6['anneeNotification']
    
    medianeRegFP = dfM6.groupby('conca')['montantTest'].median()
    medianeRegFP = pd.DataFrame(medianeRegFP)
    medianeRegFP.reset_index(level=0, inplace=True)
    medianeRegFP.columns = ['conca','montantEstimation']
    dfM6 = pd.merge(dfM6, medianeRegFP, on='conca')

    dfM6['montantTest'] = np.where(dfM6['montantTest'].isnull(), dfM6['montantEstimation'], dfM6['montantTest'])
    dfM6['diffMontant'] = dfM6['montant'] - dfM6['montantTest']
    dfM6['diffMontant'] = dfM6['diffMontant'].abs()
    dfM6['diffMontant'] = np.where(dfM6['diffMontant'] == 0, np.NaN, dfM6['diffMontant'])
    listeM6 = listeM6 + [dfM6['diffMontant'].mean()]

Minimum = []; Moyenne = []; Mediane = []; Ecart_type = []; Maximum = [];
for dfM in [listeM2, listeM3, listeM4, listeM5, listeM6]:
    Minimum += [pd.DataFrame(dfM).abs().min()] 
    Moyenne += [pd.DataFrame(dfM).abs().mean()]
    Mediane += [pd.DataFrame(dfM).abs().median()]
    Ecart_type += [pd.DataFrame(dfM).abs().std()]
    Maximum += [pd.DataFrame(dfM).abs().max()]
    
Minimum = pd.DataFrame(Minimum, index = ['M2', 'M3', 'M4', 'M5', 'M6']); Minimum.columns = ['Minimum']   
Moyenne = pd.DataFrame(Moyenne, index = ['M2', 'M3', 'M4', 'M5', 'M6']); Moyenne.columns = ['Moyenne']
Mediane = pd.DataFrame(Mediane, index = ['M2', 'M3', 'M4', 'M5', 'M6']); Mediane.columns = ['Mediane']
Ecart_type = pd.DataFrame(Ecart_type, index = ['M2', 'M3', 'M4', 'M5', 'M6']); Ecart_type.columns = ['Ecart_type']
Maximum = pd.DataFrame(Maximum, index = ['M2', 'M3', 'M4', 'M5', 'M6']); Maximum.columns = ['Maximum']
dfResultats = Minimum.join(Moyenne).join(Mediane).join(Ecart_type).join(Maximum)    

del [Minimum, Moyenne, Mediane, Ecart_type, dfM, dfM1, dfM2, dfM3, dfM4, dfM5, dfM6,
     listeM2, listeM3, listeM4, listeM5, medianeRegFP, moyenneRegFP, i, nb, listeM6]
##############################################################################
### Autre méthode à tester
# Random forest 




##############################################################################
##############################################################################

##### Conclusion - Quelle méthode on sélectionne :
dfResultats.idxmin()  

# Colonne supplémentaire pour indiquer si la valeur est estimée ou non
df['montantEstime'] = np.where(df['montant'].isnull(), 'Oui', 'Non')

# Utilisation de la méthode 5 pour estimer les valeurs manquantes
df['Region'] = df['Region'].astype(str)
df['formePrix'] = df['formePrix'].astype(str)
df['reg_FP'] = df['Region'] + df['formePrix']
df.reset_index(level=0, inplace=True)
df.reset_index(level=0, inplace=True)
del df['index']
# Calcul de la médiane par classe
medianeRegFP = pd.DataFrame(df.groupby('reg_FP')['montant'].median())
medianeRegFP.reset_index(level=0, inplace=True)
medianeRegFP.columns = ['reg_FP','montantEstimation']
df = pd.merge(df, medianeRegFP, on='reg_FP')
# Remplacement des valeurs manquantes par la médiane du groupe
df['montant'] = np.where(df['montant'].isnull(), df['montantEstimation'], df['montant'])


######################################################################
#..............Travail sur la variable de la durée des marchés
### Analyse de la composition de la duree par rapport au montant
dfDuree = pd.DataFrame.copy(df, deep = True)
dfDuree['rationMoisMontant'] = dfDuree['montant'] / dfDuree['dureeMois']
dfDuree['rationMoisMontant'].describe()
dfDuree = dfDuree[dfDuree['rationMoisMontant'].notnull()]
l = [i for i in np.arange(0,0.3,0.01)]
Rq = pd.DataFrame(np.quantile(dfDuree['rationMoisMontant'], l )); Rq.columns = ['Resultats']  
q = pd.DataFrame(l); q.columns = ['Quantiles']  
Quantiles = Rq.join(q)
plt.plot(Quantiles['Resultats'])

### Application sur le jeu de données principal df
df['dureeMoisEstime'] = np.where((df['montant']/df['dureeMois'] < 1000)
  | ((df['dureeMois'] > 12) & (df['montant']/df['dureeMois'] < 5000))
  | (df['dureeMois'] > 120), "Oui", "Non")
df['dureeMois'] = np.where(df['montant']/df['dureeMois'] < 1000, round(df['dureeMois']/30,0), df['dureeMois'])
df['dureeMois'] = np.where((df['dureeMois'] > 12) & (df['montant']/df['dureeMois'] < 5000), round(df['dureeMois']/30,0), df['dureeMois'])
df['dureeMois'] = np.where(df['dureeMois'] > 120, round(df['dureeMois']/30,0), df['dureeMois'])
df['dureeMois'] = np.where(df['dureeMois'] == 0, 1, df['dureeMois'])
#df = df(math.ceil(df['dureeMois']))

##### Check du nombre de données estimées
# Nombre de données estimées pour la durée 
(df['dureeMoisEstime'] == "Oui").sum()
# Nombre de données estimées pour le montant et la durée 
((df['dureeMoisEstime'] == "Oui") & (df['montantEstime'] == "Oui")).sum()
# Nombre de données estimées pour le montant &/OU la durée 
((df['dureeMoisEstime'] == "Oui") | (df['montantEstime'] == "Oui")).sum()


#del [GraphDate, Quantiles, Rq, chemin, data, dfDuree, l, listeCP, listeReg, medianeRegFP, q]
######################################################################



######################################################################
# Début de récupération des données des titulaires "à la main"
df.uid.value_counts() #Vérifie que les uid sont uniques
#df avec titulaires et uid
mycolumns = ['uid', 'titulaires'] 
dfTitulaires = df[mycolumns]
#Format json appliqué même pour les variables manquantes
dfDegroupe = np.where(dfTitulaires[['titulaires']].isnull(), "[{'typeIdentifiant': 'nan', 'id': 'nan', 'denominationSociale': 'nan'}]", df[['titulaires']])
dfDegroupe = pd.DataFrame(dfDegroupe, columns = ['ensemble'])
dfDegroupe = dfDegroupe.applymap(str)

dfEnsemble = dfDegroupe['ensemble'].str.split("\'", 12, expand=True)

dfEnsemble['typeIdentifiant'] = np.where((dfEnsemble[1] == 'typeIdentifiant') &  (dfEnsemble[2] == ': '), dfEnsemble[3], np.NaN)
dfEnsemble['id'] = np.where((dfEnsemble[5] == 'id') &  (dfEnsemble[6] == ': '), dfEnsemble[7], np.NaN)
dfEnsemble['denominationSociale'] = np.where((dfEnsemble[9] == 'denominationSociale') &  (dfEnsemble[10] == ': '), dfEnsemble[11], np.NaN)

dfEnsemble['OK'] = np.where((dfEnsemble['typeIdentifiant'].notnull()) & (dfEnsemble['id'].notnull()) & (dfEnsemble['denominationSociale'].notnull()) , 'OK!', np.NaN)
dfEnsemble['DS'] = np.where(dfEnsemble['OK'] == 'nan', dfEnsemble[10] +  "'" + dfEnsemble[11], np.NaN)
dfEnsemble.reset_index(level=0, inplace=True)
dfEnsembleDS = dfEnsemble['DS'].str.split('\"', 2, expand=True)
dfEnsembleDS.columns = ['t1', 'DS1', 't2']
dfEnsembleDS.reset_index(level=0, inplace=True)
dfEnsemble = pd.merge(dfEnsemble, dfEnsembleDS, on='index')
dfEnsemble['denominationSociale'] = np.where(dfEnsemble['DS1'].notnull(), dfEnsemble['DS1'], dfEnsemble['denominationSociale'])

dfEnsemble['typeIdentifiant'] = np.where((dfEnsemble[9] == 'typeIdentifiant') &  (dfEnsemble[10] == ': '), dfEnsemble[11], dfEnsemble['typeIdentifiant'])
dfEnsemble['denominationSociale'] = np.where((dfEnsemble[1] == 'denominationSociale') &  (dfEnsemble[2] == ': '), dfEnsemble[3], dfEnsemble['denominationSociale'])

dfEnsemble['OK'] = np.where((dfEnsemble['typeIdentifiant'].notnull()) & (dfEnsemble['id'].notnull()) & (dfEnsemble['denominationSociale'].notnull()) , 'OK!', np.NaN)
dfEnsemble['DS'] = np.where((dfEnsemble['OK'] == 'nan') & (dfEnsemble[0] == '[{') & (dfEnsemble[1] == 'denominationSociale'), dfEnsemble[2] +  "'" + dfEnsemble[3] + "'" + dfEnsemble[4], np.NaN)
dfEnsemble.reset_index(level=0, inplace=True)
dfEnsembleDS = dfEnsemble['DS'].str.split('\"', 2, expand=True)
dfEnsembleDS.columns = ['t1', 'DS2', 't2']
dfEnsembleDS.reset_index(level=0, inplace=True)
dfEnsemble = pd.merge(dfEnsemble, dfEnsembleDS, on='index')
dfEnsemble['denominationSociale'] = np.where(dfEnsemble['DS2'].notnull(), dfEnsemble['DS2'], dfEnsemble['denominationSociale'])

dfEnsemble['denominationSociale'] = np.where((dfEnsemble['denominationSociale'].isnull()) & (dfEnsemble[5] == "denominationSociale") & (dfEnsemble[6] == ": "), dfEnsemble[7], dfEnsemble['denominationSociale'])
dfEnsemble['id'] = np.where((dfEnsemble['id'].isnull()) & (dfEnsemble[9] == "id") & (dfEnsemble[10] == ": "), dfEnsemble[11], dfEnsemble['id'])

dfEnsemble['typeIdentifiant'] = np.where((dfEnsemble['typeIdentifiant'].isnull()) & (dfEnsemble['id'].notnull()) & (dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble[5] == 'typeIdentifiant') & (dfEnsemble[6] == ": "), dfEnsemble[7], dfEnsemble['typeIdentifiant']) 

dfEnsemble['typeIdentifiant'] = np.where(((dfEnsemble['typeIdentifiant'].isnull()) & (dfEnsemble['id'].isnull()) & (dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble[8] == 'typeIdentifiant') & (dfEnsemble[9] == ': ')), dfEnsemble[10], dfEnsemble['typeIdentifiant'])
dfEnsemble['id'] = np.where(((dfEnsemble['typeIdentifiant'].notnull()) & (dfEnsemble['id'].isnull()) & (dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble[4] == 'id') & (dfEnsemble[5] == ': ')), dfEnsemble[6], dfEnsemble['id'])

dfEnsemble['id'] = np.where(((dfEnsemble['typeIdentifiant'].isnull()) & (dfEnsemble['id'].isnull()) & (dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble[1] == 'id') & (dfEnsemble[2] == ': ')), dfEnsemble[3], dfEnsemble['id'])
dfEnsemble['id'] = np.where(((dfEnsemble['typeIdentifiant'].isnull()) & (dfEnsemble['id'].isnull()) & (dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble[6] == 'id') & (dfEnsemble[7] == ': ')), dfEnsemble[8], dfEnsemble['id'])

dfEnsemble = dfEnsemble[(dfEnsemble['denominationSociale'].notnull()) & (dfEnsemble['id'].notnull())]
dfTitulaires = dfEnsemble[['index', 'typeIdentifiant', 'id', 'denominationSociale']]
dfTitulaires.columns = ['index', 'typeIdentifiant', 'idTitulaires', 'denominationSociale']
myList = list(df.columns); myList[0] = 'index'; df.columns = myList
df = df.sort_values(['index'], ascending=[True])
df = pd.merge(df, dfTitulaires, how='left', on='index')

df['denominationSociale'].describe()
df['id'].describe()
df['typeIdentifiant'].describe()
del [dfDegroupe, dfEnsemble, dfEnsembleDS, dfTitulaires, myList, mycolumns]

##################### Nettoyage de ces nouvelles colonnes #####################
dfSIRET = df[['idTitulaires', 'typeIdentifiant', 'denominationSociale']]
dfSIRET = dfSIRET[dfSIRET['typeIdentifiant'] == 'SIRET']
dfSIRET.idTitulaires.astype(str)
dfSIRET = dfSIRET.drop_duplicates(subset=['idTitulaires'], keep='first')
dfSIRET.reset_index(inplace=True) 
for i in range(len(dfSIRET)):
    dfSIRET.idTitulaires[i] = dfSIRET.idTitulaires[i].replace("\\t", "")
    dfSIRET.idTitulaires[i] = dfSIRET.idTitulaires[i].replace("-", "")
    dfSIRET.idTitulaires[i] = dfSIRET.idTitulaires[i].replace(" ", "")
del dfSIRET['index']

dfSIRET = dfSIRET.drop_duplicates(subset=['idTitulaires'], keep='first')
dfSIRET.reset_index(inplace=True) 
for i in range (len(dfSIRET)):
    if (dfSIRET.idTitulaires[i].isdigit() == True):
        dfSIRET.typeIdentifiant[i] = 'Oui'
    else:
        dfSIRET.typeIdentifiant[i] = 'Non'
dfSIRET = dfSIRET[dfSIRET['typeIdentifiant'] == 'Oui']
del dfSIRET['typeIdentifiant'], dfSIRET['index']


######################################################################
########## Test gestion différentes colonnes titulaire ###############
######################################################################
#df[['titulaires']] = np.where(df[['titulaires']].isnull(), "[{'typeIdentifiant': 'nan', 'id': 'nan', 'denominationSociale': 'nan'}]", df[['titulaires']])
#df.groupby(level=0).apply(lambda df: df.xs(df.name)[colname].to_dict()).to_dict() 
'''
dfTitulaires = pd.DataFrame(df['titulaires'])
dfTitulaires['titulaires'] = np.where(dfTitulaires['titulaires'].isnull(), '[]', dfTitulaires['titulaires'])
dfTitulaires['titulaires'] = dfTitulaires['titulaires'].astype(str)
dfTitulaires = pd.DataFrame(dfTitulaires.titulaires.str.split('[',1).tolist(),
                                   columns = ['txt1','titulaires'])
dfTitulaires = pd.DataFrame(dfTitulaires.titulaires.str.split(']',1).tolist(),
                                   columns = ['titulaires','txt2'])
del dfTitulaires['txt2']
dfTitulaires['titulaires'] = np.where(dfTitulaires['titulaires'] == '', '{}', dfTitulaires['titulaires'])
#dfTitulaires = pd.DataFrame(dfTitulaires.values.tolist())
dfTitulaires = pd.DataFrame(dfTitulaires.groupby(level=0).apply(lambda dfTitulaires: dfTitulaires.xs(dfTitulaires.name).to_dict()).to_dict()).T 

######################################################################

dfTest = pd.DataFrame.copy(df, deep = True)
dfTest.titulaires.dropna(inplace=True)
dfTest.titulaires[1].values
pd.DataFrame.from_dict(dfTest.titulaires[1],orient='index').T

def reorga(x):
    return pd.DataFrame.from_dict(x,orient='index').T

liste_col = []
for index, liste in enumerate(dfTitulaires[:100]) :
    for i in liste :
        if i is not np.nan:
            col = reorga(i)
            col["index"] = index
            liste_col.append(col)
            
data = pd.concat(liste_col)

#df.reset_index(level=0, inplace=True) 
df['index'] = list(range(df.shape[0]))
pd.merge(df,data,"inner", on='index')
'''
######################################################################
######## Enrichissement des données via les codes siret/siren ########

#StockEtablissement_utf8
chemin = 'H:/Desktop/Data/Json/fichierPrincipal/StockEtablissement_utf8.csv'
result = pd.DataFrame(columns = ['siren', 'nic', 'siret', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement', 'activitePrincipaleEtablissement', 'nomenclatureActivitePrincipaleEtablissement'])    
dfSIRET.columns = ['siret', 'denominationSociale']
dfSIRET['siret'] = dfSIRET['siret'].astype(str)
for gm_chunk in pd.read_csv(chemin, chunksize=1000000, sep=',', encoding='utf-8', usecols=['siren', 'nic',
                                                               'siret', 'typeVoieEtablissement', 
                                                               'libelleVoieEtablissement',
                                                               'codePostalEtablissement',
                                                               'libelleCommuneEtablissement',
                                                               'codeCommuneEtablissement',
                                                               'activitePrincipaleEtablissement',
                                                               'nomenclatureActivitePrincipaleEtablissement']):
    gm_chunk['siret'] = gm_chunk['siret'].astype(str)
    resultTemp = pd.merge(dfSIRET, gm_chunk, on=['siret'])
    result = pd.concat([result, resultTemp], axis=0)
result = result.drop_duplicates(subset=['siret'], keep='first')
del [resultTemp, gm_chunk, chemin]

nanSiret = dfSIRET.merge(result, indicator=True, how='outer')
nanSiret = nanSiret[nanSiret['_merge'] == 'left_only']
nanSiret = pd.DataFrame(nanSiret[['siret', 'denominationSociale']])
nanSiret.reset_index(inplace=True)
nanSiret.columns = ['siren', 'siret', 'denominationSociale'] 
nanSiret['siren'] = nanSiret['siren'].astype(str)
for i in range(len(nanSiret)):
    nanSiret.siren[i] = nanSiret.siret[i][0:9]

chemin = 'H:/Desktop/Data/Json/fichierPrincipal/StockEtablissement_utf8.csv'
result2 = pd.DataFrame(columns = ['siren', 'nic', 'siret', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement', 'libelleCommuneEtablissement', 'codeCommuneEtablissement', 'activitePrincipaleEtablissement', 'nomenclatureActivitePrincipaleEtablissement'])    
for gm_chunk in pd.read_csv(chemin, chunksize=1000000, sep=',', encoding='utf-8', usecols=['siren', 'nic',
                                                               'siret', 'typeVoieEtablissement', 
                                                               'libelleVoieEtablissement',
                                                               'codePostalEtablissement',
                                                               'libelleCommuneEtablissement',
                                                               'codeCommuneEtablissement',
                                                               'activitePrincipaleEtablissement',
                                                               'nomenclatureActivitePrincipaleEtablissement']):
    gm_chunk['siren'] = gm_chunk['siren'].astype(str)
    resultTemp = pd.merge(nanSiret, gm_chunk, on=['siren'])
    result2 = pd.concat([result2, resultTemp], axis=0)
result2 = result2.drop_duplicates(subset=['siren'], keep='first')
del [resultTemp, gm_chunk, chemin]
   
nanSiren = nanSiret.merge(result2, indicator=True, how='outer', on='siren')
nanSiren = nanSiren[nanSiren['_merge'] == 'left_only']
nanSiren = pd.DataFrame(nanSiren[['siren', 'siret_x', 'denominationSociale_x']])
nanSiren.columns = ["siren", "siret", "a" ,"denominationSociale"]
nanSiren.reset_index(inplace=True)
del nanSiren['index']; del nanSiren['a']

######################################################################
''' #....... Solution complémentaire pour ceux non-identifié : selenium pour click button
df_scrap = pd.DataFrame(columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'verification'])    
for i in range(len(nanSiren)):
    try:
        url = 'https://www.infogreffe.fr/entreprise-societe/' + nanSiren.idTitulaires[i]
        
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
        verification = (siret == nanSiren.idTitulaires[i])
        if (detailsType1 ==' '):
            detailType = detailsType2
        else:
            detailsType = detailsType1
        
        if (verification == False):
            codeSiret = tree.xpath('//span[@class="data ficheEtablissementIdentifiantSiret"]/text()')
            infos = tree.xpath('//span[@class="data"]/text()')
            
            index = i
            rue = infos[8]
            siret = codeSiret[0].replace(" ", "")
            ville = infos[9].replace(",\xa0","")
            typeEntreprise = infos[4]
            #codeType = infos[12].replace(" : ","")
            detailsType = infos[11]
            #detailsType2 = infos[29]
            verification = (siret == nanSiren.idTitulaires[i])
    
        scrap = pd.DataFrame([index, rue, siret, ville, typeEntreprise, codeType, detailsType, verification]).T; scrap.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'verification']
        df_scrap = pd.concat([df_scrap, scrap], axis=0)
    except:
        index = i
        scrap = pd.DataFrame([index, ' ', ' ', ' ', ' ', ' ', ' ', False]).T; scrap.columns = ['index', 'rue', 'siret', 'ville', 'typeEntreprise', 'codeType', 'detailsType', 'verification']
        df_scrap = pd.concat([df_scrap, scrap], axis=0)
        pass

# Résultat : 97% des données sont enrichies
df_codeEntreprise = df_scrap[['codeType', 'detailsType']]
df_codeEntreprise = df_codeEntreprise.drop_duplicates(subset=['detailsType'], keep='first')

test = pd.DataFrame(df[df['idTitulaires'] == '79020892000023'])
'''
######################################################################
# Analyse géographique - carte 



######################################################################
# régression - random forest pour les modèles du montant
# REGEX pour les communes
# Revenir sur region avec les dict
(df['lieuExecution.typeCode']=='CODE COMMUNE').sum()
(df['lieuExecution.typeCode']=='Code commune').sum()
df['lieuExecution.typeCode'].unique()
df["lieuExecution.typeCode"].value_counts(normalize=True).plot(kind='pie')

# Gérer les concessionnaires à part
d = df.iloc[0]['concessionnaires']
d[0].describe()
# Division du dataframe en 2 
#dfMT = df[(df['montant'].notnull()) & (df['valeurGlobale'].isnull())] # données titulaires
#dfMC = df[df['valeurGlobale'].notnull()] # données concessionnaires

# Interroger Api pour code SIRET (voir INSEE)
# mettre en cache 
#   7 juillet web insee

#installer git sur pc
#mettre une nouvelle branche sur github
######################################################################