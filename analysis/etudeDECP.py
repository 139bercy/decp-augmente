# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15
@author: Lucas GEFFARD
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
dfStat.formePrix.value_counts(normalize=True).plot(kind='barh', legend=True)
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
df_villes.columns = ['codeCommuneAcheteur', 'populationAcheteur', 'superficieAcheteur', 'latitudeAcheteur','longitudeAcheteur']
df_decp = pd.merge(df_decp, df_villes, how='left', on='codeCommuneAcheteur')

# Ajout pour les etablissement
df_villes.columns = ['codeCommuneEtablissement', 'populationEtablissement', 'superficieEtablissement', 'latitudeEtablissement','longitudeEtablissement']
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
#dfDM = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).distanceAcheteurEtablissement.mean().to_frame('distanceMoyenne').reset_index()
dfDM = df_decp.groupby(['latitudeAcheteur', 'longitudeAcheteur']).distanceAcheteurEtablissement.median().to_frame('distanceMoyenne').reset_index()

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
df_carte.distanceMoyenne = np.where(df_carte.distanceMoyenne.isnull(), 0, df_carte.distanceMoyenne)

'''
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
### Mise en forme
from folium.plugins import MarkerCluster

c= folium.Map(location=[47, 2.0],zoom_start=6, tiles='OpenStreetMap')
marker_cluster = MarkerCluster().add_to(c)
for i in range (len(df_carte)):
    folium.Marker([df_carte.latitudeAcheteur[i],  df_carte.longitudeAcheteur[i]], 
                  icon=folium.features.CustomIcon('https://icon-library.com/images/map-pin-icon/map-pin-icon-17.jpg', icon_size=(max(20, min(40,df_carte.distanceMoyenne[i]/2)), max(20, min(40,df_carte.distanceMoyenne[i]/2)))),
                  popup = folium.Popup('<b>' + df_carte.libelleCommuneAcheteur[i] + '</b></br>'
                  + '<b>' + df_carte.nbMarches[i].astype(str) + '</b> marchés '
                  #+ 'Montant total des marchés : ' + df_carte.montantTotal[i].astype(str) + ' €' + '</br>'
                  + 'pour un montant moyen de <b>' + round(df_carte.montantMoyen[i]/1000,0).astype(int).astype(str) + ' mille euros</b> '
                  + "</br>avec <b>" + df_carte.nbEntreprises[i].astype(str) + ' entreprises</b> '
                  + "à une distance médiane de <b>" + df_carte.distanceMoyenne[i].astype(str) + ' km</b> ',
                  max_width = 320, min_width = 200)  
                  , clustered_marker = True).add_to(marker_cluster)
c.save('carteDECP.html')


'''
folium.CircleMarker
..
, radius=df_carte.distanceMoyenne[i],
                  color='crimson',
                  fill=True,
                  fill_color='crimson'
                  
                  
                   + '<div style="width: 50px; height: 50px; border-radius: 50%; border : solid 2px; background: orange; position: absolute; margin-top: 35px; margin-left: 130px; opacity:0.5"></div>', 

#, radius = 10 # CircleMarker icon = folium.Icon(color='green')

...
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
del i
#df_carte,
###############################################################################
############################ Segmentation de marché ###########################
###############################################################################
#... Créer une bdd par villes (acheteur/client)
# Selection des variables qualitatives
dfBIN = df_decp[['type', 'nature', 'procedure', 'lieuExecutionTypeCode', 'regionAcheteur']]
#... Vérification des caractéristiques
# Arrangement du code du lieu d'exécution
dfBIN['lieuExecutionTypeCode'] = dfBIN['lieuExecutionTypeCode'].str.upper()
dfBIN['lieuExecutionTypeCode'] = np.where(dfBIN['lieuExecutionTypeCode'] == 'CODE DÉPARTEMENT', 'CODE DEPARTEMENT', dfBIN['lieuExecutionTypeCode'])
dfBIN['lieuExecutionTypeCode'] = np.where(dfBIN['lieuExecutionTypeCode'] == 'CODE RÉGION', 'CODE REGION', dfBIN['lieuExecutionTypeCode'])
dfBIN['lieuExecutionTypeCode'] = np.where(dfBIN['lieuExecutionTypeCode'] == 'CODE ARRONDISSEMENT', 'CODE DEPARTEMENT', dfBIN['lieuExecutionTypeCode'])
dfBIN['lieuExecutionTypeCode'] = np.where((dfBIN['lieuExecutionTypeCode'] == 'CODE COMMUNE') | (dfBIN['lieuExecutionTypeCode'] == 'CODE POSTAL'), 'CODE COMMUNE/POSTAL', dfBIN['lieuExecutionTypeCode'])
# Vérification des types
dfBIN.dtypes
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
dfNoBin = df_decp[['libelleCommuneAcheteur', 'montant', 'dureeMois', 
                   'dureeMoisCalculee', 'distanceAcheteurEtablissement']]
# Création d'une seule colonne pour la durée du marché
dfNoBin['duree'] = round(dfNoBin.dureeMoisCalculee, 0)
del dfNoBin['dureeMois'], dfNoBin['dureeMoisCalculee']
# On modifie les valeurs manquantes pour la distance en appliquant la médiane
dfNoBin.distanceAcheteurEtablissement = np.where(dfNoBin['distanceAcheteurEtablissement'].isnull(), dfNoBin['distanceAcheteurEtablissement'].median(), dfNoBin['distanceAcheteurEtablissement'])
# Vérification des types
dfNoBin.dtypes


# On obtient alors notre df prêt sans variables qualitatives (sauf libellé)
df = dfNoBin.join(dfBIN)
del dfNoBin, dfBIN
df = df[df['libelleCommuneAcheteur'].notnull()]
df['nbContrats'] = 1 # Trouver autre solution

#... Gestion des régions
df = df.groupby(['libelleCommuneAcheteur']).sum().reset_index()
ensemble = ['regionAcheteur_Auvergne-Rhône-Alpes',
       'regionAcheteur_Bourgogne-Franche-Comté', 'regionAcheteur_Bretagne',
       'regionAcheteur_Centre-Val de Loire',
       "regionAcheteur_Collectivité d'outre mer", 'regionAcheteur_Corse',
       'regionAcheteur_Grand Est', 'regionAcheteur_Guadeloupe',
       'regionAcheteur_Guyane', 'regionAcheteur_Hauts-de-France',
       'regionAcheteur_La Réunion', 'regionAcheteur_Martinique',
       'regionAcheteur_Mayotte', 'regionAcheteur_Normandie',
       'regionAcheteur_Nouvelle-Aquitaine', 'regionAcheteur_Occitanie',
       'regionAcheteur_Pays de la Loire',
       "regionAcheteur_Provence-Alpes-Côte d'Azur",
       'regionAcheteur_Île-de-France']
df['HighScore'] = df[ensemble].max(axis=1)
for x in ensemble:
    df[x] = np.where(df[x] == df['HighScore'], 1, 0)

#... Fréquence 
ensemble = ['nature_Accord-cadre', 'nature_CONCESSION DE SERVICE',
       'nature_CONCESSION DE SERVICE PUBLIC', 'nature_CONCESSION DE TRAVAUX',
       'nature_Concession de service', 'nature_Concession de service public',
       'nature_Concession de travaux', 'nature_DELEGATION DE SERVICE PUBLIC',
       'nature_Délégation de service public', 'nature_Marché',
       'nature_Marché de partenariat', 'nature_Marché hors accord cadre',
       'nature_Marché subséquent', "procedure_Appel d'offres ouvert",
       "procedure_Appel d'offres restreint", 'procedure_Dialogue compétitif',
       'procedure_Marché négocié sans publicité ni mise en concurrence préalable',
       'procedure_Marché public négocié sans publicité ni mise en concurrence préalable',
       'procedure_Procédure adaptée', 'procedure_Procédure avec négociation',
       'procedure_Procédure non négociée ouverte',
       'procedure_Procédure non négociée restreinte',
       'procedure_Procédure négociée ouverte',
       'procedure_Procédure négociée restreinte',
       'lieuExecutionTypeCode_CODE CANTON',
       'lieuExecutionTypeCode_CODE COMMUNE/POSTAL',
       'lieuExecutionTypeCode_CODE DEPARTEMENT',
       'lieuExecutionTypeCode_CODE PAYS', 'lieuExecutionTypeCode_CODE REGION']
for x in ensemble:
    df[x] = df[x]/df['nbContrats']
del df['HighScore'], ensemble, x

#... Duree, montant et distance moyenne par ville (par rapport au nb de contrats)
df.distanceAcheteurEtablissement = round(df.distanceAcheteurEtablissement/df['nbContrats'],0)
df.duree = round(df.duree/df['nbContrats'],0)
df['montantMoyen'] = round(df.montant/df['nbContrats'],0)

#... Finalement les données spatiales ne sont pas gardés pour réaliser la segmentation
df.drop(columns = df.columns[36:55], axis = 1, inplace = True)

# Renomme des colonnes
df.columns = ['libelleCommuneAcheteur', 'montantTotal', 'distanceMoyenne', 'dureeMoyenne', 'nbContratDeConcession', 'nbMarché',
       'nature_Accord-cadre', 'nature_CONCESSION DE SERVICE', 'nature_CONCESSION DE SERVICE PUBLIC', 'nature_CONCESSION DE TRAVAUX', 'nature_Concession de service', 'nature_Concession de service public',
       'nature_Concession de travaux', 'nature_DELEGATION DE SERVICE PUBLIC', 'nature_Délégation de service public', 'nature_Marché',
       'nature_Marché de partenariat', 'nature_Marché hors accord cadre', 'nature_Marché subséquent', "procedure_Appel d'offres ouvert",
       "procedure_Appel d'offres restreint", 'procedure_Dialogue compétitif',
       'procedure_Marché négocié sans publicité ni mise en concurrence préalable',
       'procedure_Marché public négocié sans publicité ni mise en concurrence préalable',
       'procedure_Procédure adaptée', 'procedure_Procédure avec négociation',
       'procedure_Procédure non négociée ouverte', 'procedure_Procédure non négociée restreinte',
       'procedure_Procédure négociée ouverte', 'procedure_Procédure négociée restreinte',
       'lieuExecutionTypeCode_CODE CANTON', 'lieuExecutionTypeCode_CODE COMMUNE/POSTAL', 'lieuExecutionTypeCode_CODE DEPARTEMENT', 
       'lieuExecutionTypeCode_CODE PAYS', 'lieuExecutionTypeCode_CODE REGION', 'nbContrats', 'montantMoyen']

#... Mettre les valeurs sur une même unité de mesure
from sklearn.preprocessing import StandardScaler
df_nom = pd.DataFrame(df.libelleCommuneAcheteur)
del df['libelleCommuneAcheteur']
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
# Vérification
np.around(scaled_df.mean(axis = 0),10) # Doit être égal à 0
np.around(scaled_df.std(axis = 0),10) # Doit être égal à 1
### On obtient le df nécessaire pour réaliser la segmentation de marché !
scaled_df[0] # Aperçu de la première ligne

#... On réassemble le df
df = df_nom.join(df)
del df_nom

###############################################################################
### Réalisation de l'ACP
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

### Essayer sur 3 axes

#print(scaled_df.shape) # n=2340 et p=37
acp = PCA(svd_solver='full')
coord = acp.fit_transform(scaled_df)
#proportion de variance expliquée
print(acp.explained_variance_ratio_)
#scree plot
eigval = (2340-1)/2340*acp.explained_variance_
plt.plot(np.arange(1,36+1), eigval)
#cumul de variance expliquée
plt.plot(np.arange(1,36+1),np.cumsum(acp.explained_variance_ratio_))

# Test des bâtons brisés
bs = np.cumsum(1/np.arange(36,0,-1))[::-1]
# D'après les résultats aucun facteur n'est valide...
print(pd.DataFrame({'Val.Propre':eigval,'Seuils':bs}))

###############################################################################
### Application de l'algorithme des k-means
from sklearn.cluster import KMeans

'''
# TESTS
for n in range(19):
    n=n+1
    print('\nTest avec ', n)
    model=KMeans(n_clusters=n)
    model.fit(scaled_df)
    for i in range(n):
        print((model.labels_==i).sum())
'''
# K-means - on prend 7 grappes
model=KMeans(n_clusters=7)
model.fit(scaled_df)
print(model.cluster_centers_)
print(model.labels_)
res = model.labels_
    
# Graphique du résultat
for point in scaled_df:
    clusterID = model.predict(point.reshape(1,-1))
    if clusterID == [0]:
        plt.scatter(point[0], point[1], c='b')
    elif clusterID == [1]:
        plt.scatter(point[0], point[1], c='g')
    elif clusterID == [2]:
        plt.scatter(point[0], point[1], c='r')
    elif clusterID == [3]:
        plt.scatter(point[0], point[1], c='c')
    elif clusterID == [4]:
        plt.scatter(point[0], point[1], c='m')
    elif clusterID == [5]:
        plt.scatter(point[0], point[1], c='y')
    elif clusterID == [6]:
        plt.scatter(point[0], point[1], c='k')
#for center in model.cluster_centers_:
#    plt.scatter(center[0],center[1])
plt.show()

# Nombre de communes par grappe
for i in range(7):
    print((model.labels_==i).sum())

# Ajout des résultats
res = pd.DataFrame(res, columns=['segmentation_KMEANS'])
df = df.join(res)
del coord, eigval, i, point, res, bs, clusterID #center

'''
from pandas.plotting import scatter_matrix
dftest = pd.DataFrame(scaled_df)
dftest = dftest[dftest.columns[0:9]]
dftest = dftest[dftest<10]
dftest = dftest.dropna()
scatter_matrix(dftest,figsize=(9,9))
del dftest[3], dftest[6], dftest[7], dftest[8]
scatter_matrix(dftest,figsize=(5,5))

df['montantTotal'].decribe()
df.montantTotal[(df.montantTotal<10000000)].plot.hist()
dftest[0][dftest[0]<-0.1].hist()
'''

###############################################################################
### Application de l'algorithme de classification ascendante hiérarchique - CAH
#graphique - croisement deux à deux des variables

#librairies pour la CAH
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
'''
# Générer la matrice des liens
Z = linkage(df.iloc[:,1:38] ,method='ward',metric='euclidean')
# Dendrogramme
plt.title('CAH avec matérialisation des X classes')
dendrogram(Z,labels=df.index,orientation='left',color_threshold=0.08e10)
plt.show()
# Récupération des classes
groupes_cah = pd.DataFrame(fcluster(Z,t=0.08e10,criterion='distance'), columns = ['segmentation_CAH'])
'''
############ Avec les données normalisée
# Générer la matrice des liens
Z = linkage(scaled_df ,method='ward',metric='euclidean')
# Dendrogramme
plt.title('CAH avec matérialisation des X classes')
dendrogram(Z,labels=df.index,orientation='left',color_threshold=65)
plt.show()
# Récupération des classes
groupes_cah = pd.DataFrame(fcluster(Z,t=65,criterion='distance'), columns = ['segmentation_CAH'])

### Ajout au df 
df = df.join(groupes_cah)
del Z, groupes_cah, scaled_df

###############################################################################
### Comparons les résultats des deux méthodes
df.segmentation_KMEANS.value_counts()
df.segmentation_CAH.value_counts() # Regarder les données autres clusters principaux

pd.crosstab(df.segmentation_CAH, df.segmentation_KMEANS)

df.describe()
resTest = df[(df['segmentation_CAH']==1) | (df['segmentation_CAH']==6) | (df['segmentation_CAH']==5)].groupby(['segmentation_CAH']).mean()
resTest2 = df[(df['segmentation_KMEANS']==0) | (df['segmentation_KMEANS']==1) | (df['segmentation_KMEANS']==4)].groupby(['segmentation_KMEANS']).mean()
del resTest, resTest2

# Conclusion
# Séelctionner le clustering avec le CAH car :
    # Résultat reproductible (pas d'aléatiore)
    # Forme des amas non hyper-sphérique
    # Gère mieux les outliers / plus précis
    # On ne connait pas à l'avance le nombre k de cluster

# Règle avec le montant total
df['segmentation_RULES']=3
df['segmentation_RULES'] = np.where(df['montantTotal']<=2500000, 2, df['segmentation_RULES'])
df['segmentation_RULES'] = np.where(df['montantTotal']<=1800000, 1, df['segmentation_RULES'])
pd.crosstab(df.segmentation_CAH, df.segmentation_RULES) # Ne fonctionne pas
# Règle avec le montant moyen
df['segmentation_RULES']=3
df['segmentation_RULES'] = np.where(df['montantMoyen']<=300000, 2, df['segmentation_RULES'])
df['segmentation_RULES'] = np.where(df['montantMoyen']<=100000, 1, df['segmentation_RULES'])
pd.crosstab(df.segmentation_CAH, df.segmentation_RULES) # Ne fonctionne pas non plus
# Règle avec la distance moyenne
df['segmentation_RULES']=3
df['segmentation_RULES'] = np.where(df['distanceMoyenne']<=110, 2, df['segmentation_RULES'])
df['segmentation_RULES'] = np.where(df['distanceMoyenne']<=60, 1, df['segmentation_RULES'])
pd.crosstab(df.segmentation_CAH, df.segmentation_RULES) # Ne fonctionne tjrs pas !
# Règle avec le nombre de marché
df['segmentation_RULES']=3
df['segmentation_RULES'] = np.where(df['nbMarché']<60, 2, df['segmentation_RULES'])
df['segmentation_RULES'] = np.where(df['nbMarché']<12, 1, df['segmentation_RULES'])
pd.crosstab(df.segmentation_CAH, df.segmentation_RULES) # NON

### CONCLUSION :
# La CAH est la bonne solution pour réaliser nos clusters
#Test = df[['libelleCommuneAcheteur', 'montantTotal', 'montantMoyen', 'distanceMoyenne',
#           'dureeMoyenne', 'nbContrats', 'segmentation_CAH']]
###############################################################################
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
df_bar.to_csv(r'H:/Desktop/Data/df_Ratio.csv', sep=';',index = False, header=True, encoding='utf-8')

'''
### HeatMap du ratio calculé
#from folium.plugins import HeatMap
df_carte2 = df_carte[['latitudeAcheteur','longitudeAcheteur','ratioEntreprisesMarchés']]
df_carte2.ratioEntreprisesMarchés = np.where(df_carte2.ratioEntreprisesMarchés>=2, 2, df_carte2.ratioEntreprisesMarchés)
df_carte2.ratioEntreprisesMarchés = round(df_carte2.ratioEntreprisesMarchés*100,0).astype(int)
'''
'''
df_HeatMap = pd.DataFrame(columns=['latitudeAcheteur','longitudeAcheteur'])
for i in range(len(df_carte2)):
    print(i)
    for j in range(df_carte2.ratioEntreprisesMarchés[i]):
        ar = np.array([[df_carte2.latitudeAcheteur[i],df_carte2.longitudeAcheteur[i]]])
        df_temp = pd.DataFrame(ar, columns = ['latitudeAcheteur', 'longitudeAcheteur'])
        df_HeatMap = pd.concat([df_HeatMap, df_temp])
df_HeatMap['Count']=1        

################################ GAIN DE TEMPS ################################
df_HeatMap = pd.DataFrame(columns=['latitudeAcheteur','longitudeAcheteur'])
l=0
for k in [250,500,750,1000,1250,1500,1750,2000,len(df_carte2)]:
    df_temp2 = pd.DataFrame(columns=['latitudeAcheteur','longitudeAcheteur'])
    for i in range(l, k):
        #print(i)
        for j in range(df_carte2.ratioEntreprisesMarchés[i]):
            ar = np.array([[df_carte2.latitudeAcheteur[i],df_carte2.longitudeAcheteur[i]]])
            df_temp = pd.DataFrame(ar, columns = ['latitudeAcheteur', 'longitudeAcheteur'])
            df_temp2 = pd.concat([df_temp2, df_temp])
    df_HeatMap = pd.concat([df_HeatMap, df_temp2])
    l=k
del l, i, j, k, ar, df_temp, df_temp2   
###############################################################################
import random
pd.set_option('display.precision',12)
df_HeatMap.reset_index(inplace=True, drop=True)
df_HeatMap.latitudeAcheteur = df_HeatMap.latitudeAcheteur.astype(float)
df_HeatMap.longitudeAcheteur = df_HeatMap.longitudeAcheteur.astype(float)
for i in range(len(df_HeatMap)):
    df_HeatMap.latitudeAcheteur[i] = df_HeatMap.latitudeAcheteur[i] + round(random.random()*0.000000001,12)
    df_HeatMap.longitudeAcheteur[i] = df_HeatMap.longitudeAcheteur[i] + round(random.random()*0.000000001,12)
df_HeatMap.latitudeAcheteur = df_HeatMap.latitudeAcheteur.astype(str)
df_HeatMap.longitudeAcheteur = df_HeatMap.longitudeAcheteur.astype(str)

### Code pour la carte HeatMap
from folium.plugins import HeatMap
base_map = folium.Map(location=[47, 2.0],zoom_start=6, tiles='OpenStreetMap')
marker_cluster = MarkerCluster().add_to(base_map)
HeatMap(data=df_HeatMap, radius=20, max_zoom=13).add_to(base_map)
for i in range (len(df_carte)):
    folium.Marker([df_carte.latitudeAcheteur[i],  df_carte.longitudeAcheteur[i]], 
                  icon=folium.features.CustomIcon('https://images.emojiterra.com/google/android-nougat/512px/2753.png', icon_size=(10,10)),
                  popup = folium.Popup('<b>' + df_carte.libelleCommuneAcheteur[i] + '</b></br>'
                  + '<b>' + df_carte.nbMarches[i].astype(str) + '</b> marchés '
                  #+ 'Montant total des marchés : ' + df_carte.montantTotal[i].astype(str) + ' €' + '</br>'
                  +  " pour <b>" + df_carte.nbEntreprises[i].astype(str) + '</b> entreprises ',
                  max_width = 320, min_width = 200)  
                  , clustered_marker = True).add_to(marker_cluster)
base_map.save('carte2DECP.html')
'''

### HeatMap montantTotal / Population
df_HeatMap = pd.merge(df_carte, df_decp[['populationAcheteur','libelleCommuneAcheteur']], how='inner', on=['libelleCommuneAcheteur'])
df_HeatMap = df_HeatMap.drop_duplicates(subset=['latitudeAcheteur', 'longitudeAcheteur'], keep='first')
df_HeatMap = df_HeatMap[df_HeatMap.populationAcheteur.notnull()]
df_HeatMap.reset_index(inplace=True, drop=True)
df_HeatMap['ratioMontantTTPopulation'] = df_HeatMap.montantTotal / df_HeatMap.populationAcheteur
df_HeatMap['ratioMontantTTPopulation'] = np.where(df_HeatMap['populationAcheteur']==0, 0, df_HeatMap['ratioMontantTTPopulation'])
#df_HeatMap.ratioMontantTTPopulation.describe()
df_HeatMap.ratioMontantTTPopulation = round(df_HeatMap.ratioMontantTTPopulation/10,0).astype(int)
df_HeatMap['ratioMontantTTPopulation'] = np.where(df_HeatMap['ratioMontantTTPopulation']>300, 300, df_HeatMap['ratioMontantTTPopulation'])
#df_HeatMap['ratioMontantTTPopulation'].sum()


df_HeatMap2 = pd.DataFrame(columns=['latitudeAcheteur','longitudeAcheteur'])
l=0
for k in [250,500,750,1000,1250,1500,1750,2000,len(df_HeatMap)]:
    df_temp2 = pd.DataFrame(columns=['latitudeAcheteur','longitudeAcheteur'])
    for i in range(l, k):
        print(i)
        for j in range(df_HeatMap.ratioMontantTTPopulation[i]):
            ar = np.array([[df_HeatMap.latitudeAcheteur[i],df_HeatMap.longitudeAcheteur[i]]])
            df_temp = pd.DataFrame(ar, columns = ['latitudeAcheteur', 'longitudeAcheteur'])
            df_temp2 = pd.concat([df_temp2, df_temp])
    df_HeatMap2 = pd.concat([df_HeatMap2, df_temp2])
    l=k
del l, i, j, k, ar, df_temp, df_temp2  

from folium.plugins import HeatMap
base_map = folium.Map(location=[47, 2.0], zoom_start=6, max_zoom=12, min_zoom=5, tiles='OpenStreetMap')
HeatMap(data=df_HeatMap2[['latitudeAcheteur', 'longitudeAcheteur']], radius=8).add_to(base_map)
marker_cluster = MarkerCluster().add_to(base_map)
for i in range (len(df_carte)):
    folium.Marker([df_carte.latitudeAcheteur[i],  df_carte.longitudeAcheteur[i]], 
                  icon=folium.features.CustomIcon('https://images.emojiterra.com/google/android-nougat/512px/2753.png', icon_size=(10,10)),
                  popup = folium.Popup('<b>' + df_carte.libelleCommuneAcheteur[i] + '</b></br>'
                  + '<b>' + df_carte.nbMarches[i].astype(str) + '</b> marchés '
                  #+ 'Montant total des marchés : ' + df_carte.montantTotal[i].astype(str) + ' €' + '</br>'
                  +  " pour <b>" + df_carte.nbEntreprises[i].astype(str) + '</b> entreprises ',
                  max_width = 320, min_width = 200)  
                  , clustered_marker = True).add_to(marker_cluster)
base_map.save('carte2DECP.html')

# Répartition des marchés
a = folium.Map(location=[47, 2.0], zoom_start=6, max_zoom=8, min_zoom=5, tiles='OpenStreetMap')
HeatMap(data=df_carte[['latitudeAcheteur', 'longitudeAcheteur']], radius=8).add_to(a)
a.save('carteRépartitionDECP.html')

del df_HeatMap, df_HeatMap2, df_bar, i


###############################################################################
###############################################################################
###############################################################################
### Récap des erreurs
df_ERROR = df_decp[(df_decp.montantEstEstime=='Oui') | (df_decp.dureeMoisEstEstime=='Oui') 
                    | (df_decp.geomAcheteur.isnull()) | (df_decp.geomEtablissement.isnull())]

df_ERROR = df_ERROR[['identifiantMarche','objetMarche', 'acheteurId','acheteurNom', 
                     'idEtablissement', 'montantOriginal',  'dureeMois',
                     'montantEstEstime', 'dureeMoisEstEstime', 'geomAcheteur', 'geomEtablissement']]
df_ERROR.columns = ['identifiantMarche','objetMarche', 'acheteurId','acheteurNom', 'EtablissementID',
                     'montantOriginal', 'dureeMoisOriginal', 'montantAberrant', 'dureeMoisAberrant',
                     'siretAcheteur', 'siretEtablissement']
df_ERROR.siretAcheteur = np.where(df_ERROR.siretAcheteur.isnull(), 'MAUVAIS', 'BON')
df_ERROR.siretEtablissement = np.where(df_ERROR.siretEtablissement.isnull(), 'MAUVAIS', 'BON')


df_Classement = pd.DataFrame.copy(df_ERROR, deep = True)
df_Classement = df_Classement[['acheteurNom', 'montantAberrant', 'dureeMoisAberrant', 'siretAcheteur', 'siretEtablissement']]
df_Classement.columns = ['acheteurNom', 'montantEstEstime', 'dureeMoisEstEstime', 'siretAcheteur', 'siretEtablissement']
df_Classement.montantEstEstime = np.where(df_Classement.montantEstEstime=='Oui',1,0)
df_Classement.dureeMoisEstEstime = np.where(df_Classement.dureeMoisEstEstime=='Oui',1,0)
df_Classement.siretAcheteur = np.where(df_Classement.siretAcheteur=='MAUVAIS',1,0)
df_Classement.siretEtablissement = np.where(df_Classement.siretEtablissement=='MAUVAIS',1,0)

df_Classement = df_Classement.groupby(['acheteurNom']).sum().reset_index()
df_50 = pd.DataFrame(df_Classement[(df_Classement.montantEstEstime >= 50) |
        (df_Classement.dureeMoisEstEstime >= 300) |
        (df_Classement.siretAcheteur >= 180) |
        (df_Classement.siretEtablissement >= 50)])
df_50['Note']=df_50.montantEstEstime*4+df_50.dureeMoisEstEstime*1+df_50.siretAcheteur*1+df_50.siretEtablissement*2
df_50=df_50.sort_values(by = 'Note', ascending = False)
del df_50['Note']

#siretEtablissement
df_decp.idEtablissement[df_decp.siretEtablissement.isnull()]

#### BILAN
#(df_ERROR.montantEstEstime=='Oui').sum() # 2952
#(df_ERROR.dureeMoisEstEstime=='Oui').sum() # 24732
#(df_ERROR.siretAcheteur=='MAUVAIS').sum() # 14285
#(df_ERROR.siretEtablissement=='MAUVAIS').sum() # 6691
    
Bilan=pd.DataFrame(df_Classement.sum()[1:5]).T; Bilan.columns=['Montant aberrant ','Durée en mois aberrante ','Siret acheteur mauvais ','Siret entreprise mauvais ']

# Les pires lignes (4 erreurs): 
F = df_ERROR[(df_ERROR.montantAberrant=='Oui') & (df_ERROR.dureeMoisAberrant=='Oui') & (df_ERROR.siretAcheteur=='MAUVAIS') & (df_ERROR.siretEtablissement=='MAUVAIS')]

# Liste de tous les acheteurs ayant fait au moins 10 supposées erreurs :
df_Classement['Total'] = df_Classement.montantEstEstime + df_Classement.dureeMoisEstEstime + df_Classement.siretAcheteur + df_Classement.siretEtablissement
ListeMauvaixAcheteurs = pd.DataFrame(np.array([df_Classement.acheteurNom[df_Classement['Total']>10].unique()]).T, columns=['Acheteur'])


###############################################################################
###############################################################################
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
df_RECAP.columns=['Montant original', 'Montant calculé', 'Durée en mois originale', 'Durée en mois calculée','Distance acheteur - établissement']
df_RECAP = df_RECAP[1:8]

# Récupération sous format CSV
df_ERROR.to_csv(r'H:/Desktop/Data/df_ERROR.csv', sep=';',index = False, header=True, encoding='utf-8')
ListeMauvaixAcheteurs.to_csv(r'H:/Desktop/Data/ListeMauvaixAcheteurs.csv', sep=';',index = False, header=True, encoding='utf-8')
df_50.columns = ['acheteurNom', 'montantAberrant', 'dureeMoisAberrant', 'siretAcheteurFAUX', 'siretEtablissementFAUX']
df_50.to_csv(r'H:/Desktop/Data/df_50.csv', sep=';',index = False, header=True, encoding='utf-8')

del F, ListeMauvaixAcheteurs, df_ERROR, df_RECAP, df_50, df_Classement, Bilan
#del df, df_carte





# min 20 marchés pour classement/graphe préférence






