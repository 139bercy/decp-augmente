# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08
@author: Lucas GEFFARD
"""
######################################################################
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize #Pour corriger les données Json imbriquées
#%matplotlib inline
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import seaborn as sns
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

# Vérification du nombre de nan dans les colonnes annee, mois, région, departement
df['moisNotification'].isnull().sum() #1601
df['anneeNotification'].isnull().sum() #1601
df['codePostal'].isnull().sum() #2876
(df['Region']=='nan').sum() #2495

#On décide de supprimer les lignes ou la variable région est manquante
#car ceux sont des données au niveau du pays ou internationales
df = df[df.Region != 'nan']

# Check du nombre de nan dans les montants          
df['montant'].isnull().sum() #1188
(df['montant']=='nan').sum() #0

###################### Méthodes simples ########################
#On va utiliser différentes méthodes pour gérer les valeurs manquantes   
#......... Méthode 1 : Suppression des valeurs manquantes
dfM1 = pd.DataFrame.copy(df, deep = True)
dfM1 = dfM1[dfM1['montant'].notnull()]
       
#......... Méthode 2 : Remplacement des nan par la moyenne
dfM2 = pd.DataFrame.copy(df, deep = True)
dfM2['montant'] = np.where(dfM2['montant'].isnull(), dfM2['montant'].mean(), dfM2['montant'])

#......... Méthode 3 : Remplacement des nan par la médiane
dfM3 = pd.DataFrame.copy(df, deep = True)
dfM3['montant'] = np.where(dfM3['montant'].isnull(), dfM3['montant'].median(), dfM3['montant'])


###################### Méthodes plus complexes ########################
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

import scipy.stats as st
dfM = pd.DataFrame.copy(df, deep = True)
dfM = dfM[dfM['montant'].notnull()]
dfM = dfM[dfM['moisNotification'].notnull()]
df['montant'] = df['montant'].astype(float)
df['moisNotification'] = df['moisNotification'].astype(float)
st.pearsonr(dfM['moisNotification'], dfM['montant'])[0]

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
# Quelques graphes :
plt.scatter(dfM['moisNotification'], dfM['montant'])
plt.scatter(dfM['anneeNotification'], dfM['montant'])
plt.scatter(dfM['codeRegion'], dfM['montant'])





'''
#......... Méthode 4 : Remplacement des nan par la moyenne des strates
dfM4 = pd.DataFrame.copy(df, deep = True)
impute_grps = pd.pivot_table(dfM4, values=["montant"], 
                               index=["Region", "moisNotification", "anneeNotification"], 
                               aggfunc=[np.mean])
impute_grps.describe()
# source - _type - nature - procedure - formePrix - codePostal
# codeRegion - Region - anneeNotification - moisNotification
dfM4 = dfM4[dfM4[['montant']].fillna(df.groupby('Region').transform('mean'))]
dfM4['montant'].isnull().sum() #1188 ou 1209 ?..
dfM4.montant.describe()
dfM4["montant"] = dfM4.groupby("Region").transform(dfM4.montant.isnull(dfM4.montant.mean()))
dfM4 = pd.DataFrame(dfM4.groupby('Region')['montant'].isnull().transform('mean'))
dfM4.groupby('Region')["montant"].median()
grouped = dfM4.groupby('montant').mean()

#......... Méthode 5 : Remplacement des nan par la médiane des strates
dfM5 = pd.DataFrame.copy(df, deep = True)



##### Conclusion - Quelle méthode on sélectionne :





######################################################################
# Vérification des données durée via le montant          
'''         
          
          
          
          
          
          
          
          
          
          
          
          
          
          
'''          
######################################################################
#Check pour comprendre les différences de data entre les sources
df['source'].describe() #5 sources différentes
df['source'].unique() 
#Regardons les 5 sources indépendemment :
dfGouvAife = df.loc[df['source'] == 'data.gouv.fr_aife'] #Normal !
dfMpInfo = df.loc[df['source'] == 'marches-publics.info'] #denomination en premier...
dfGouvPes = df.loc[df['source'] == 'data.gouv.fr_pes'] #Normal !
dfMpE = df.loc[df['source'] == 'e-marchespublics'] #Normal !
dfGrandLyon = df.loc[df['source'] == 'grandlyon'] #id en premier ...

######################################################################
# Début de récupération des données des titulaires "à la main"
#............... Résultats partiels, non finis

df.uid.value_counts() #Vérifie que les uid sont uniques
#df avec titulaires et uid
mycolumns = ['uid', 'titulaires'] 
dfTitulaires = df[mycolumns]
#Format json appliqué même pour les variables manquantes
dfDegroupe = np.where(dfTitulaires[['titulaires']].isnull(), "[{'typeIdentifiant': 'nan', 'id': 'nan', 'denominationSociale': 'nan'}]", df[['titulaires']])
dfDegroupe = pd.DataFrame(dfDegroupe, columns = ['ensemble'])
dfDegroupe = dfDegroupe.applymap(str)
#Découpage de la colonne titualire pour récupérer la première variable
dfEnsemble1 = dfDegroupe['ensemble'].str.split("\': \'", 1, expand=True)
dfEnsemble1 = dfEnsemble1[1].str.split("\', \'", 1, expand=True)
dfEnsemble = pd.DataFrame(dfEnsemble1[0])
dfEnsemble.columns = ['texte1']
#Découpage de la colonne titualire pour récupérer la deuxième variable
dfEnsemble1 = dfEnsemble1[1].str.split("\': \'", 1, expand=True)
dfEnsemble1 = dfEnsemble1[1].str.split("\'", 1, expand=True)
dfEnsemble = pd.concat([dfEnsemble, dfEnsemble1[0]], axis = 1)
dfEnsemble.columns = ['texte1', 'texte2']
#Découpage de la colonne titualire pour récupérer la troisième variable
dfEnsemble1 = dfEnsemble1[1].str.split("\': \'", 1, expand=True)
dfEnsemble1 = dfEnsemble1[1].str.split("\'}", 1, expand=True)
#Groupage des 3 variables récupérées permettant d'identifier entièrement une entreprise
dfEnsemble = pd.concat([dfEnsemble, dfEnsemble1[0]], axis = 1)
dfEnsemble.columns = ['texte1', 'texte2', 'texte3']
#Tri des variables car elles sont dans le désordre 
dfEnsemble['texte1'] = np.where(dfEnsemble['texte1'] == 'SIRET', None, dfEnsemble['texte1'])
dfEnsemble['texte2'] = np.where(dfEnsemble['texte2'] == 'SIRET', None, dfEnsemble['texte2'])
dfEnsemble['texte3'] = np.where(dfEnsemble['texte3'] == 'SIRET', None, dfEnsemble['texte3'])
dfEnsemble['texte1'] = np.where(dfEnsemble['texte1'] == 'nan', None, dfEnsemble['texte1'])
dfEnsemble['texte2'] = np.where(dfEnsemble['texte2'] == 'nan', None, dfEnsemble['texte2'])
dfEnsemble['texte3'] = np.where(dfEnsemble['texte3'] == 'nan', None, dfEnsemble['texte3'])
#Regroupement des variables 
dfEnsemble['texte3'] = np.where(dfEnsemble['texte3'].isnull(), dfEnsemble['texte1'], dfEnsemble['texte3'])
dfEnsemble['texte3'] = np.where(dfEnsemble['texte3'] == dfEnsemble['texte1'], dfEnsemble['texte2'], dfEnsemble['texte3'])
dfEnsemble['texte2'] = np.where(dfEnsemble['texte3'] == dfEnsemble['texte2'], dfEnsemble['texte1'], dfEnsemble['texte2'])
dfEnsemble['texte1'] = None

###### Test pour différencier les deux colonnes, ne fonctionnent pas.. 
#isinstance(dfEnsemble['texte3'], int
#s.startswith('0') == True
#len("amelp7135sja2gf")==14
#"0" in ttestt

#df['test'] = df['test'].str.replace(',','-')

# Utile quand le reste fonctionnera, pour l'instant non
#del dfEnsemble['texte1']  
#dfEnsemble.columns = ['code.Siret', 'nom.Entreprise']
#........... A continuer pour les autres entreprises (si aucune autre solution)

######################################################################
#........................... Identifier les variables peu représentées
print(df.isnull().sum())
#Sur 127 711 lignes il y a pour 6 variables 127 574 valeurs manquantes 
#ce qui représente que 99.89% des valeurs
df['_type'].describe()
#Ces variables sont de _type contrat de concession, il faudra sans doute les traiter à part
dfCttConcession = pd.DataFrame(df.loc[df['_type'] == 'Contrat de concession'])
#Graphiques pour ces données
dfCttConcession['valeurGlobale'] = dfCttConcession['valeurGlobale'].astype('float')
plt.boxplot(dfCttConcession['valeurGlobale']) #Boite à moustache des montants
plt.title("Montant des contrats de concession\n", fontsize=18, color='#3742fa')
#Sans les plus grandes valeurs
dfCttConcession = dfCttConcession[dfCttConcession['valeurGlobale'].between(-1, 1.0e+7)]
plt.boxplot(dfCttConcession['valeurGlobale']) #Boite à moustache des montants
plt.title("Montant des contrats de concession\n", fontsize=18, color='#3742fa')
#........................... A noter
# Ces variables sont très différentes des autres, elles n'ont même pas de montant
# Il faut alors utiliser la colonne valeurGlobale        
'''