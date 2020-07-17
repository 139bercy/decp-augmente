# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:14:51 2020

@author: Administrator
"""
# Librairies
import pandas as pd
import numpy as np
import folium

# Importation des données
df_decp = pd.read_csv('H:/Desktop/Data/decp.csv', sep=';', encoding='utf-8', 
                      dtype={'acheteurId' : str, 'nicEtablissement' : str, 'codeRegionAcheteur' : str, 'denominationSocialeEtablissement' : str,
                             'moisNotification' : str,  'idEtablissement' : str, 'montant' : float, 'montantTotalMarche' : float, 'codeDepartementAcheteur' : str,
                             'anneeNotification' : str, 'codeCommuneEtablissement' : str, 'codePostalEtablissement' : str,  'identifiantMarche' : str,
                             'codeTypeEtablissement' : str, 'sirenEtablissement' : str, 'siretEtablissement' : str, 'codeCPV' : str,
                             'nbTitulairesSurCeMarche' : int, 'dureeMois': int, 'dureeMoisCalculee': int, 'codeCommuneAcheteur': str, 'codePostalAcheteur': str})

'''
######################################################################
############### Stats descriptives ###############
import matplotlib.pyplot as plt

###..... Jeu de données
dfStat = pd.DataFrame.copy(df_decp, deep = True)
dfStat.columns
dfStat.info()
dfStat.isnull().sum()
dfStat.nunique()

###..... Variables quantitatives
# Montant des marchés
dfStat.montant.describe()
dfStat.montant[dfStat.montant < 1000000].plot(kind='box')
dfStat.montant[dfStat.montant < 173000].plot(kind='box')

# Duree des marchés
dfStat.dureeMois.describe()
dfStat.dureeMoisCalculee.describe()
dfStat.dureeMois[dfStat.dureeMois < 120].plot(kind='box')
dfStat.dureeMoisCalculee[dfStat.dureeMois < 120].plot(kind='box')


###..... Variables qualitatives
# Source
dfStat.source.value_counts(normalize=True).plot(kind='bar')
# Forme des prix / PROCEDURE
dfStat.formePrix.value_counts(normalize=True).plot(kind='bar', legend=True)
dfStat.formePrix.value_counts(normalize=True).plot(kind='bar', legend=True, logy =True)
# Nature
dfStat.nature[(dfStat.nature=='Marché')|(dfStat.nature=='Accord-cadre')|(dfStat.nature=='Marché subséquent')].value_counts(normalize=True).plot(kind='pie')
# _Type
dfStat.type.value_counts(normalize=True).plot(kind='pie')

# Region
dfStat.regionAcheteur.value_counts(normalize=True).plot(kind='bar')
# Code Postal
dfStat.codePostalAcheteur.describe()

# AnneeNotification
dfStat.anneeNotification.value_counts(normalize=True).sort_index().plot(kind='line')
# MoisNotification 
plt.plot(dfStat.moisNotification.value_counts(normalize=True).sort_index())
# Date de publication
dfStat.datePublicationDonnees.value_counts(normalize=True).sort_index().plot(kind='line', rot=45)
# Date de notification
dfStat.dateNotification.value_counts(normalize=True).sort_index().plot(kind='line', rot=45)

# Lieu d'exécution
dfStat['lieuExecutionNom'].describe()
# Nom acheteur
dfStat['acheteur.nom'].describe()
# codeCPV
dfStat.codeCPV.describe()
# pie chart top 6

# nic
dfStat.nicEtablissement.describe()
# codeTypeEtablissement
dfStat.codeTypeEtablissement.describe()
# siren
dfStat.sirenEtablissement.describe()
# siret
dfStat.siretEtablissement.describe()

# Acheteur - Etablissement 
(dfStat.codeCommuneEtablissement == dfStat.codeCommuneAcheteur).sum()
(dfStat.codePostalEtablissement == dfStat.codePostalAcheteur).sum()

######## Statistiques bivariées
# Duree | Montant
dfStat[(dfStat.dureeMoisCalculee < 120) & (dfStat.montant < 50000000)].plot.scatter("dureeMoisCalculee", "montant")
dfStat[(dfStat.dureeMois < 120) & (dfStat.montant < 50000000)].plot.scatter("dureeMoisCalculee", "montant")
dfStat[(dfStat.dureeMois < 40) & (dfStat.montant < 10000000)].plot.scatter("dureeMoisCalculee", "montant")

# Type -> Marché/Contrat de concession
dfStat[dfStat.montant < 1000000].boxplot(column = "montant", by = "type") 
dfStat[dfStat.dureeMoisCalculee < 100].boxplot(column = "dureeMoisCalculee", by = "type") 

# Montant / Region
dfStat[dfStat.montant < 400000].boxplot(column = "montant", by = "regionAcheteur", rot=90) 

# Montant / nature
dfStat[dfStat.montant < 400000].boxplot(column = "montant", by = "nature", rot=90)
dfStat[(dfStat.montant < 400000) & ((dfStat.nature=='Marché')|(dfStat.nature=='Accord-cadre')|(dfStat.nature=='Marché subséquent'))].boxplot(column = "montant", by = "nature", rot=90)

# distance entre entreprise et commune 

#################################### Villes ###################################
# Levallois-Perret
dfLP = dfStat[dfStat.codeCommuneAcheteur == '92044']
#dfLP = dfStat[dfStat['acheteur.id'] == '21920044100018']
dfLP.formePrix.value_counts()
dfLP.siretEtablissement.value_counts()
dfLP.montant.plot(kind='box')
dfLP.plot.scatter("dureeMoisCalculee", "montant")
test = dfLP[dfLP.siretEtablissement == '81031603400018']

# Puteaux
dfPT = dfStat[dfStat.codeCommuneAcheteur == '92062']
#dfPT = dfStat[dfStat['acheteur.id'] == '21920062300011']
dfPT.formePrix.value_counts()
dfPT.nature.value_counts()
dfPT.siretEtablissement.value_counts()
dfPT.montant[dfPT.montant<4000000].plot(kind='box')
dfPT.plot.scatter("dureeMoisCalculee", "montant")
test = dfPT[dfPT.siretEtablissement == '30666424400036']

# Issy-les-Moulineaux
dfIM = dfStat[dfStat.codeCommuneAcheteur == '92040']
dfIM.siretEtablissement.value_counts()
test = dfIM[dfIM.siretEtablissement == '39882733700021']
'''
'''
######################################################################
######################################################################
######## Enrichissement latitude & longitude avec adresse précise 
df_decp.adresseEtablissement = df_decp.adresseEtablissement.astype(str).str.upper() 
df_code_adresse = df_decp[['codeCommuneEtablissement', 'adresseEtablissement']]
df_code_adresse = df_code_adresse[df_code_adresse.codeCommuneEtablissement.notnull()]
df_code_adresse = df_code_adresse[df_code_adresse.adresseEtablissement != 'NAN']
df_code_adresse.columns = ['code_insee', 'nom_voie']
df_code_adresse = df_code_adresse[['nom_voie', 'code_insee']]

result = pd.DataFrame(columns = ['code_insee', 'nom_voie', 'x', 'y', 'lon', 'lat'])
for gm_chunk in pd.read_csv('H:/Desktop/Data/Json/fichierPrincipal/adresses-france.csv', chunksize=10000, 
                           sep=';', header = 0, error_bad_lines=False, 
                           usecols=['nom_voie', 'code_insee', 
                                    'x', 'y', 'lon', 'lat']):
    gm_chunk.nom_voie = gm_chunk.nom_voie.astype(str).str.upper()
    gm_chunk.code_insee = gm_chunk.code_insee.astype(str)
    df_temp = pd.merge(df_code_adresse, gm_chunk, how='inner', 
                       on=['code_insee', 'nom_voie'])
    result = pd.concat([result, df_temp], axis=0)
result = result.drop_duplicates(subset=['nom_voie', 'code_insee'], keep='first')


# Connaitre le nombre de lignes d'un fichier csv sans l'ouvrir
with open('H:/Desktop/Data/Json/fichierPrincipal/adresses-france.csv') as fp:
    for (count, _) in enumerate(fp, 1):
       pass
'''
######################################################################
######################################################################
######## Enrichissement latitude & longitude avec adresse la ville 
df_villes = pd.read_csv('H:/Desktop/Data/Json/fichierPrincipal/code-insee-postaux-geoflar.csv', 
                        sep=';', header = 0, error_bad_lines=False,
                        usecols=['CODE INSEE', 'geom_x_y', 'Superficie', 'Population'])
df_villes['ordre']=0
df_villes2 = pd.read_csv('H:/Desktop/Data/Json/fichierPrincipal/code-insee-postaux-geoflar.csv', 
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
#(df_villes.ordre==1).sum()
#(df_villes.ordre==0).sum()
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
df_villes.columns = ['codeCommuneAcheteur', 'superficieAcheteur', 'populationAcheteur', 'latitudeAcheteur','longitudeAcheteur']
df_decp = pd.merge(df_decp, df_villes, how='left', on='codeCommuneAcheteur')

# Ajout pour les etablissement
df_villes.columns = ['codeCommuneEtablissement', 'superficieEtablissement', 'populationEtablissement', 'latitudeEtablissement','longitudeEtablissement']
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

# Analyse des résultats
df_decp.distanceAcheteurEtablissement.describe()
df_decp.distanceAcheteurEtablissement.plot(kind='box')

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
###############################################################################
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
dfDM = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).distanceAcheteurEtablissement.mean().to_frame('distanceMoyenne').reset_index()

df_carte = pd.merge(df_carte, dfMT, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfMM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfIN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfSN, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])
df_carte = pd.merge(df_carte, dfDM, how='left', on=['latitudeAcheteur', 'longitudeAcheteur'])

del dfMM, dfMT, dfIN, dfSN, dfDM

df_carte.montantTotal = round(df_carte.montantTotal, 0)
df_carte.montantMoyen = round(df_carte.montantMoyen, 0)
df_carte.nbMarches = round(df_carte.nbMarches, 0)
df_carte.nbEntreprises = round(df_carte.nbEntreprises, 0)
df_carte.distanceMoyenne = round(df_carte.distanceMoyenne, 0)

### Mise en forme
from folium.plugins import MarkerCluster
c= folium.Map(location=[47, 2.0],zoom_start=6, tiles='OpenStreetMap')
marker_cluster = MarkerCluster().add_to(c)
for i in range (len(df_carte)):
    folium.Marker([df_carte.latitudeAcheteur[i],  df_carte.longitudeAcheteur[i]],
                  popup = folium.Popup('<b>' + df_carte.libelleCommuneAcheteur[i] + '</b>' + '</br>'
                  + 'Nombre total de marchés : ' + df_carte.nbMarches[i].astype(str) + '</br>'
                  + 'Montant total des marchés : ' + df_carte.montantTotal[i].astype(str) + ' €' + '</br>'
                  + 'Montant moyen des marchés : ' + df_carte.montantMoyen[i].astype(str) + ' €' + '</br>'
                  + "Nombre total d'entreprises ayant passées un marché : " + df_carte.nbEntreprises[i].astype(str) + '</br>'
                  + "Distance moyennes des entreprises : " + df_carte.distanceMoyenne[i].astype(str) + ' Km'
                  , max_width = 400, min_width = 300), clustered_marker = True).add_to(marker_cluster)
c.save('carteDECP.html')
'''
###############################################################################
###############################################################################
###############################################################################
############........ CARTE DES MARCHES PAR ETABLISSEMENT 
df_carte2 = df_decp[['latitudeAcheteur', 'longitudeAcheteur', 'libelleCommuneAcheteur', 'dateNotification', 'referenceCPV',
                     'montant', 'montantTotalMarche', 'nbTitulairesSurCeMarche', 'dureeMois', 'communeEtablissement',
                     'codePostalEtablissement', 'denominationSocialeEtablissement', 'distanceAcheteurEtablissement']]
df_carte2 =df_carte2[df_carte2['latitudeAcheteur'] != 'nan']
df_carte2 = df_carte2[df_carte2['longitudeAcheteur'] != 'nan']
df_carte2.reset_index(inplace=True, drop=True)

df_carte2.montant = round(df_carte2.montant, 0) ; df_carte2.montant = df_carte2.montant.astype(str)
df_carte2.nbTitulairesSurCeMarche = round(df_carte2.nbTitulairesSurCeMarche, 0) ; df_carte2.nbTitulairesSurCeMarche = df_carte2.nbTitulairesSurCeMarche.astype(str)
df_carte2.montantTotalMarche = round(df_carte2.montantTotalMarche, 0) ;  df_carte2.montantTotalMarche = df_carte2.montantTotalMarche.astype(str)
df_carte2.dureeMois = round(df_carte2.dureeMois, 0) ; df_carte2.dureeMois = df_carte2.dureeMois.astype(str)
df_carte2.distanceAcheteurEtablissement = round(df_carte2.distanceAcheteurEtablissement, 0) ; df_carte2.distanceAcheteurEtablissement = df_carte2.distanceAcheteurEtablissement.astype(str)
#df_carte2.dtypes

c= folium.Map(location=[47, 2.0],zoom_start=6, tiles='OpenStreetMap')
marker_cluster = MarkerCluster().add_to(c)
for i in range (len(df_carte2)):
    folium.Marker([df_carte2.latitudeAcheteur[i],  df_carte2.longitudeAcheteur[i]],
                  popup = folium.Popup('<b>' + df_carte2.libelleCommuneAcheteur[i] + '</b>' + '</br>'
                                       #+ 'Date de notification du marché : ' + df_carte2.dateNotification[i] + '</br>'
                                       #+ 'Etablissement en lien avec ce marché : ' + df_carte2.denominationSocialeEtablissement[i] + '</br>'
                                       #+ 'Commune de cet établissement : ' + df_carte2.communeEtablissement[i] + '</br>'
                                       #+ 'Distance : ' + df_carte2.distanceAcheteurEtablissement[i] + ' Km' + '</br>'
                                       #+ 'Montant réparti par entreprises : ' + df_carte2.montant[i] + ' €' + '</br>'
                                       #+ "Nombre d'entreprises : " + df_carte2.nbTitulairesSurCeMarche[i] + '</br>'
                                       #+ 'Montant total : ' + df_carte2.montantTotalMarche[i] + ' €' + '</br>'
                                       #+ 'Reference du marché : ' + df_carte2.referenceCPV[i] + '</br>'
                                       #+ 'Duree du marché en mois : ' + df_carte2.dureeMois[i]
                                       , max_width = 400, min_width = 300)).add_to(marker_cluster)
c.save('carte2DECP.html')

del df_carte, df_carte2, i
'''
###############################################################################
###############################################################################
del df_decp['superficieEtablissement'], df_decp['populationEtablissement'], df_decp['latitudeAcheteur'], df_decp['longitudeAcheteur'], df_decp['latitudeEtablissement'], df_decp['longitudeEtablissement']