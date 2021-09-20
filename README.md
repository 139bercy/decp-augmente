[![CircleCI](https://circleci.com/gh/139bercy/decp-augmente.svg?style=svg)](https://circleci.com/gh/139bercy/decp-augmente)

<img src="decp-augmente.png" alt="decp augmenté - logo"/>

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
       <td>logging</td>
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
       <td></td>
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
- StockEtablissement_utf8.csv : https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/
- StockUniteLegale_utf8.csv : https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/
- departement2020.csv : https://www.insee.fr/fr/information/4316069
- region2020.csv : https://www.insee.fr/fr/information/4316069

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
- Dans le dossier confs, le fichier var_to_export correspond à toutes les variables dans le DataFrame decp avant export final. 
   - Mettre à 1 les colonnes que l'on souhaite exporter
   - 0 Sinon.
- Exécuter tout le code 

## Valorisation de la donnée

### Nettoyage 
Explication de l'ensemble du traitement réalisé pour la partie nettoyage des données. 

#### Travail sur les titulaires
Au sein des decp on distingue deux types de données: les marchés et les concessions. Nous avons donc fusionner les colonnes comportant des informations semblables.

- La colonne concessionnaire a fusionné avec la colonne <b>titulaire</b> 
- La colonne valeur globale a fusionné avec la colonne <b>montant</b>
- La colonne autoriteConcedante.id a fusionné avec la colonne <b>acheteur.id</b>
- La colonne autoriteConcedante.nom a fusionné avec la colonne <b>acheteur.nom</b>
- On ne conserve que les données ou un titulaire est renseigné.

#### Travail sur les montants
Dans un soucis de conservation de l'information source, la colonne montant est renomée en <b>montantOriginal</b> et l'ensemble des opérations suivantes seront appliquées à la colonne <b>montantCalcule</b>.

- Les valeurs manquantes sont remplacées par 0
- Les montants inférieurs à 200€ et supérieur à 999 999 999€ sont remis à 0
- Les montants composés d'au moins 5 fois le même chiffre (hors 0) sont remis à 0.

#### Travail sur des codes manquants
Ce qui est appelé code correspond aux variables d'identifications. On parle ici aussi bien de la viariable id (identifiant des lignes de la base), que id.Titulaires (ientifiant des entreprises) ou encore code_CPV permettant l'identification des types de marchés. 

- Remplacement des valeurs manquantes de <b>id</b> par '0000000000000000' (la colonne id sera retravaillé un peu plus tard dans le processus de nettoyage)
- Rerait des caractères spéciaux présent dans <b>idTitulaires</b>. On obtient le numéro SIRET
- Récupération du NIC et stockage dans une colonne <b>nic</b>
- Création d'une colonne <b>CPV_min</b> composé des deux premiers chiffre du code CPV. Cela permet d'identifier le type de marché (Fournitures/Travaux/Services)

#### Travail sur les régions
Récupération des codes de départements des marchés. On en profite pour ajouter les libellés et la région. 

- Extraction du code département de la colonne <b>lieuExecution.code</b> et stocké dans <b>codeDepartementExecution</b>
- Ajout du libellé du département / code de la région / libellé de la région respectivement dans <b>libelleDepartementExecution</b>, <b>codeRegionExecution</b> et <b>libelleRegionExecution</b>

#### Travail sur les dates

- Récupération de l'année et du mois de notification du marché et stockage dans <b>anneeNotification</b> et <b>moisNotification</b>
- Remplacement par np.NaN des dates lorsque l'années est inférieur à 1980 et supérieur à 2100

#### Travail sur la durée du marché
Au sein des DECP les durées de marché sont exprimées en mois. De même que pour les montants, on conserve les durées initiales et on modifie les durées dans la colonne <b>dureeMoisCalculee</b>. De plus, on ajoute la colonne <b>dureeMoisEstimee</b>: Booléenne, la durée est elle estimée ? <br>
Les durées corrigées sont celles :
- manquantes: corrigées à 0 <br>

Tous les cas suivant seront corrigées en divisant par 30 (conversion en mois)
- durée égale au montant 
- montant / durée < <b>100</b>
- montant / durée < <b>1000</b> pour une durée > <b>12</b> mois
- duréee == <b>30</b> ou <b>31</b> et montant < <b>200 000</b> €
- duréee == <b>360</b> ou <b>365</b> ou <b>366</b> et montant < <b>10 000 000</b> €
- durée > <b>120</b> et montant < <b>2 000 000</b> €

Un début de travail sur de l'imputation a aussi été réalisé. On se place dans le cas des marchés qui ne sont pas des travaux (CPV_min != 45) et qui ont une duréeCalculee supérieure à 120. <br>
    - Imputation par la médiane des durées de mois pour un <b>CPV_min</b> équivalent.

#### Travail sur la variable qualitative objet.
Remplacement du caractère '�' par 'XXXXX' dans la colonne objet

## Enrichissement des données

La partie enrichissement des données va nous permettre d'ajouter, grâce à des sources externes, de la donnée dans nos DECP. Pour cela nous utiliserons les sources de données suivantes:
   - INSEE
       - Code des régions, départements, arrondissements, cantons.
       - La base SIREN 
   - OpenDatasoft
       - Géolocalisations des communes
   - SIMAP (Système d'Information pour les MArchés Publics)
       - Nomencalture européenne des codes CPV

### Réalisation d'un dashboard 
   Un dashboard a été fait et est disponible [ici](https://datavision.economie.gouv.fr/decp/?view=France)