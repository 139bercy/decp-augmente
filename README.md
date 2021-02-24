[![CircleCI](https://circleci.com/gh/139bercy/decp-augmente.svg?style=svg)](https://circleci.com/gh/139bercy/decp-augmente)

<img src="https://github.com/139bercy/decp-augmente/blob/V_Python_3_7/decp-augmente.png" alt="logoDECP"/>

# PROJET DECP - Données Essentielles de la Commande Publique

## Description


## Les éléments nécessaires 
### Version
Pour l'instant (28/08/2020) le script est fonctionnel uniquement sur <b>Python 3.7</b> et <b>Spyder 3.3.6</b>.
Une version mise à jour est en implémentation pour Python 3.8 et Syper 4.

### Les librairies Python utilisées

<table>
   <tr>
       <td>pandas</td>
       <td>numpy</td>
       <td>json</td>
       <td>os</td>
       <td>time</td>
       <td>tqdm</td>
       <td>lxml</td>
       <td>requests</td>
       <td>pickle (optionel)</td>
   </tr>
   <tr>
       <td>selenium</td>
       <td>urllib</td>
       <td>scikit-learn</td>
       <td>scipy</td>
       <td>matplotlib</td>
       <td>folium</td>
       <td>colorama</td>
       <td>math</td>
       <td>bokeh</td>
   </tr>
</table>

### Données
#### JSON
Le fichier des DECP utilisé dans ce projet est en format <b>JSON</b>, il se nomme : 'decp.json'. Ce fichier est mis à jour régulièrement sur le site de data.gouv : 
https://www.data.gouv.fr/fr/datasets/fichiers-consolides-des-donnees-essentielles-de-la-commande-publique/

#### CSV
Plusieurs données sous format csv - xlsx sont nécessaires afin d'enrichir les données :
- code-insee-postaux-geoflar.csv : https://public.opendatasoft.com/explore/dataset/code-insee-postaux-geoflar/export/?flg=fr
- cpv_2008_ver_2013.xlsx : https://simap.ted.europa.eu/fr/web/simap/cpv
- departements-francais.csv : https://www.regions-et-departements.fr/departements-francais
- StockEtablissement_utf8.csv : https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/

#### GEOJSON
Pour réaliser la représentation graphique des données, ségmentée par régions et départements, certaines donneés en geojson sont récupérées directement via leur URL.
- https://france-geojson.gregoiredavid.fr/repo/regions.geojson
- https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-avec-outre-mer.geojson

### Webdriver
Afin d'optimiser l'enrichissement via le code siret/siren, une partie des données sont récupérées via une méthode de scraping sur le site infogreffe.fr. Pour une partie du code, le fichier exécutable <b>geckodriver.exe</b> est obligatoire. 


## To Do
- Posséder tous les éléments cités ci-dessus
- Changer le chemin indiquer dans le code à la ligne 39 (mettre le chemin où se situe le code et les données)
- S'assurer que les données sont dans les bons sous-chemins (le fichier decp.zip permet de vérifier cela)
- Exécuter tout le code 

## Détails sur le script
### Importation et mise en forme des données 
- Importation des librairies (certaines ont besoin d’être installées au préalable)
- Chargement des données JSON puis <b>aplatissement par marchés</b>
- Colonnes titulaires et concessionnaires réunies puis traitées ensemble
- Aplatissement des données par titulaires on obtient alors les données par "contrats" (relation acheteurs – entreprises), moins de 10% des marchés ont plus d'un titulaire donc l’impact est faible
- Suppression des doublons
- Non prise en charge de la colonne ‘<b>Modification</b>’

### Premier traitement – Valorisation 
- <b>Identification et suppression des montants aberrants</b> (imputés plus tard)
- Gestion des id et codes CPV manquants
- Création d’une colonne permettant de connaitre le nombre de titulaires par marchés
- Montant total réparti de façon uniforme en fonction du nombre de titulaires sur un marché (montant total sauvegardé dans une nouvelle colonne)
- Suppression des caractères spéciaux et alphabétique dans les codes SIRET pour tenter de les récupérer
- Création de deux colonnes permettant d’attribuer précisément une région et un département à chaque marché (localisation générale des acheteurs)
- Mise en forme et correction des données temporelles (date/durée)
- <b>Imputation des montants</b> en utilisant la médiane stratifiée (le plus possible : Région, code CPV, forme du marché), et création d’une colonne permettant d’identifier les montants imputés
- Identification et <b>rectification des durées</b> (en mois) exprimées en jours

### Enrichissement des données 
- Enrichissement des données des acheteurs et des entreprises via le code SIRET/SIREN (et dans le pire des cas grâce à la dénomination sociale des entreprises)
   - 1er enrichissement réalisé avec le code SIRET en fusionnant avec une BDD INSEE
   - 2nd enrichissement réalisé avec le code SIREN en fusionnant avec une BDD INSEE
   - 3e enrichissement en utilisant plusieurs méthodes de scraping sur le site INFOGREFFE (en utilisant code SIRET, SIREN, et dénomination sociale)
- Si certains codes SIRET sont déjà identifiés comme faux (lors d’un lancement ultérieur du code) alors ils sont automatiquement supprimés des méthodes d’enrichissement pour gagner du temps
- Enrichissement via les codes CPV : identification précise de leur référence via une BDD
- Ajout de la <b>géolocalisation précise des acheteurs et des entreprises</b> : latitude et longitude de la ville dans laquelle ils sont identifiés via les codes SIRET/SIREN
   - La géolocalisation précise permet ensuite de calculer la distance entre les acheteurs et les entreprises pour chaque marché (utilisation de la formule de Vincenty avec le rayon moyen de la Terre)
- <b>Segmentation de marché</b> : utilisation de la classification par ascendant hiérarchique (CAH) afin de classer les acheteurs dans des clusters (au total <b>3 clusters</b> principaux et quelques données hors-clusters)

### Vérification de la qualité des données
- Utilisation de l’<b>algorithme de Luhn</b> pour détecter les SIREN faux
- Calcul du ratio nb entreprises / nb marchés (avec tous les montants et avec les montants>40K)
- <b>Récapitulatif de toutes les erreurs supposées</b> répertoriées dans le df_ERROR
- Calcul et sauvegarde du nombre d'erreurs par commune

### Réalisation d'un dashboard 
- <b>Représentation cartographique</b> des données
  - Données par commune, information sur le montant total, le nombre de marchés, le nombre d’entreprises, la distance médiane (acheteurs – entreprises) et la segmentation de l’acheteur
  - HeatMap des contrats au niveau national
  - Répartition des montants totaux par région
  - Répartition des montants / nb population par département
- Réalisation de <b>vignettes</b> mettant en avant les chiffres / indicateurs quantitatifs les plus importants
- Réalisation de nombreux <b>graphiques</b> permettant d'avoir une première approche facile et rapide des données 

## Règles d'imputations
### Montants aberrants
Les montants corrigés sont ceux :
- manquants
- inférieurs à <b>200€</b> 
- supérieurs à <b>999 000 000€</b>

### Durées exprimées en jours
Les durées corrigées sont celles :
- manquantes
- durée égale au montant
- montant / durée < <b>100</b>
- montant / durée < <b>1000</b> pour une durée > <b>12</b>
- duréee == <b>30</b> ou <b>31</b> et montant < <b>200 000</b>
- duréee == <b>360</b> ou <b>365</b> ou <b>366</b> et montant < <b>10 000 000</b>
- durée > <b>120</b> et montant < <b>2 000 000</b>
