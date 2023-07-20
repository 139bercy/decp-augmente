[![CircleCI](https://circleci.com/gh/139bercy/decp-augmente.svg?style=svg)](https://circleci.com/gh/139bercy/decp-augmente)

![Image](decp-augmente.png)

# PROJET DECP - Données Essentielles de la Commande Publique

## Description


## Les éléments nécessaires 
### Version
Pour l'instant (20/07/2023) le script est fonctionnel uniquement sur <b>Python 3.7</b>.  <brQ>

### Les librairies Python utilisées

Sont renseignées dans le fichier **requirements.txt**

### Données
#### Format des données
Cette version de augmenté est relié à la sortie de decp-rama-V2

#### JSON
Le fichier des DECP utilisés dans ce projet est en format <b>JSON</b>, il se nomme : 'decp.json'. Ce fichier est mis à jour régulièrement sur le site de data.gouv : 
https://www.data.gouv.fr/fr/datasets/fichiers-consolides-des-donnees-essentielles-de-la-commande-publique/

#### CSV
- cpv_2008_ver_2013.xlsx : https://simap.ted.europa.eu/fr/web/simap/cpv


### Pour lancer en local
- Dans le fichier utils.py mettre la variable *USE_S3* à False
- S'assurer que les données sont dans les bons sous-chemins
- Dans le dossier confs, le fichier var_to_export correspond à toutes les variables dans le DataFrame decp avant export final. 
   - Mettre à 1 les colonnes que l'on souhaite exporter
   - 0 Sinon.
- Exécuter tout le code 


## Fonctionnement général
### En amont
En amont de cette pipeline les données sont traitées par decp-rama-v2 puis uploadés sur data.gouv "decpv2.json".

### La pipeline

 - **1.** Code sur Github
 - **2.** Tests de non-régression sur CircleCI
 - **3.** Exécution du code sur un échantillon fixe du dataset sur CircleCI
 - **4.** Exécution du code chez Axus


#### Choix des données exportées
- marchés valides
- marchés non valides
- concessions valides
- concessions non valides


### Autre scripts
#### upload_dataeco.py
Fin de la chaine, après enrichissement le script permet de mettre sur le serveur dataeco le résultat des pipelines.
Ainsi, le fichier uploadé via lftp est visible à l'adresse data eco souhaitée.
#### utils.py
Gère tout ce qui est lié au S3.


## Comment fonctionne la CI sur ce projet ?

La branche utilisée actuellement pour la CICD est :
<div align="center">
:last_quarter_moon_with_face: master :first_quarter_moon_with_face:
</div>

### CI (Github - circleCI
Lorsqu'on push le code sur Github, on effectue via un workflow CircleCI des tests de non-régression (via le job pytest).
Puis, on exécute tout le code sur un échantillon fixe du dataset.

:guardsman:

### Quelques remarques
- pour le moment, tout s'effectue sur la branche master en local
- le fichier utils.py est à modifier pour lancer en local

### Réalisation d'un tableau de bord 
   :chart_with_upwards_trend: Un dashboard a été fait et est disponible [ici](https://datavision.economie.gouv.fr/decp/?view=France)
