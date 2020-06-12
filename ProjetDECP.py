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
#............... Quelques stats descriptives
#Différentes sources des datas
df["source"].value_counts(normalize=True).plot(kind='pie') #data.gouv.fr_aife - Marches-public.info
plt.xlabel('')
plt.ylabel('')
plt.title("Source des marchés\n", fontsize=18, color='#3742fa')
#Type de contrat
df["_type"].value_counts(normalize=True).plot(kind='pie') #Marché : 127 574
plt.xlabel('')
plt.ylabel('')
plt.title("Type de marchés\n", fontsize=18, color='#3742fa')
          
#Nature des contrats
dfNature = pd.DataFrame(df.groupby('nature')['uid'].nunique())
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
sns.barplot(x=df['source'], y=df['montant'], palette="Reds_r")
plt.xlabel('\nSources', fontsize=15, color='#c0392b')
plt.ylabel("Montant\n", fontsize=15, color='#c0392b')
plt.title("Montant moyen en fonction des sources\n", fontsize=18, color='#e74c3c')
plt.xticks(rotation= 45)
plt.tight_layout()
#Montants en fonction des jours (date)
df['montant'].fillna(0, inplace=True)
GraphDate = pd.DataFrame(df.groupby('datePublicationDonnees')['montant'].sum())
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
######################################################################
######################################################################
#...............    Suppression des variables "inutiles"
#del df['donneesExecution']
#del df['concessionnaires']
#del df['uuid']       
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
######################################################################
#Identifier les montants non rempli ou incorrects
(df['montant'].isnull()).sum() # Le nombre de contrat de concession
(df['montant']<100).sum() # En dessous de 100, on peut penser que c'est des erreurs
((df['montant']>10) & (df['montant']<40000)).sum() # En dessous de 40 000, ils ne devraient pas être mentionnés...
((df['montant']>=40000) & (df['montant']<100000)).sum() # Si le projet de loi passe, ils ne seront plus mentionnés
((df['montant']>=5e5) & (df['montant']<5e6)).sum() # Vérifier si ces données sont correctes
(df['montant']>=5e6).sum() # Données très certainement fausses
#........................... On a donc :
((df['montant']<40000) | (df['montant']>=5e5)).sum()# Données supposées fausses
((df['montant']>=40000) & (df['montant']<1.5e5)).sum()# Données supposées bonnes
# A voir si la barre des 500 000 est bonne ou trop petite, peut-être que bcp de projets dépassent les millions
# A continuer...