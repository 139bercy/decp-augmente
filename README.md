[![CircleCI](https://circleci.com/gh/139bercy/decp-augmente.svg?style=svg)](https://circleci.com/gh/139bercy/decp-augmente)

![Image](decp-augmente.png)

# PROJET DECP - Données Essentielles de la Commande Publique

## Description


## Les éléments nécessaires 
### Version
Pour l'instant (21/08/2023) le script est fonctionnel uniquement sur <b>Python 3.9</b>.  <brQ>

### Les librairies Python utilisées

Sont renseignées dans le fichier **requirements.txt**

### Données
#### Format des données
Cette version de augmenté est relié à la sortie de decp-rama-V2

#### JSON
Le fichier des DECP utilisés dans ce projet est en format <b>JSON</b>, il se nomme : 'decpv2.json'. Ce fichier est mis à jour régulièrement sur le site de data.gouv : 
https://www.data.gouv.fr/fr/datasets/fichiers-consolides-des-donnees-essentielles-de-la-commande-publique/

#### CSV
- cpv_2008_ver_2013.xlsx : https://simap.ted.europa.eu/fr/web/simap/cpv


### Pour lancer en local
- S'assurer que les données sont dans les bons sous-chemins, placer les fichiers dans le dossier data (cpv_2008_ver_2013.xlsx et decpv2.json)
- Exécuter tout le code en lançant le script main.py


## Fonctionnement général
### En amont
En amont de cette pipeline les données sont traitées par decp-rama-v2 puis uploadés sur data.gouv "decpv2.json".

### La pipeline

 - **1.** Code sur Github
 - **2.** Tests de non-régression sur CircleCI (en cours)
 - **3.** Exécution du code sur un échantillon fixe du dataset sur CircleCI
 - **4.** Exécution du code chez Axus pour générer les csv
 - **5.** Upload des csv sur dataeco via lftp


#### Choix des données exportées
- marchés valides
- marchés non valides
- concessions valides
- concessions non valides


### Autre scripts
#### upload_dataeco.py
Non utilisé actuelement !

le script permet de mettre sur le serveur dataeco le résultat des pipelines.
Ainsi, le fichier uploadé via lftp est visible à l'adresse data eco souhaitée.


## Comment fonctionne la CI sur ce projet ?

La branche utilisée actuellement pour la CI est :
<div align="center">
:last_quarter_moon_with_face: master :first_quarter_moon_with_face:
</div>

### CI (Github - circleCI
Lorsqu'on push le code sur Github, on effectue via un workflow CircleCI des tests de non-régression (via le job pytest).
Puis, on exécute tout le code sur un échantillon fixe du dataset. :guardsman:

### Quelques remarques
- pour le moment, tout s'effectue sur la branche master en local
- le fichier upload_dataeco.py n'est pas utilisé actuellement mais permet de mettre sur le serveur dataeco le résultat des pipelines. Ainsi, le fichier uploadé via lftp est visible à l'adresse data eco souhaitée.

### Réalisation d'un tableau de bord 
   :chart_with_upwards_trend: Un dashboard a été fait et est disponible [ici](https://datavision.economie.gouv.fr/decp/?view=France)
