# decp-projet
Projet sur les données essentielles de la commande publique

## Explication du projet


# Les éléments nécessaires 



## Détails du script

### Importation et mise en forme des données 
Importation des librairies (certaines ont besoin d’être installées au préalable)
Chargement des données JSON puis aplatissement par marchés
Colonnes titulaires et concessionnaires réunies puis traitées ensemble
Aplatissement des données par titulaires on obtient alors les données par contrats (relation acheteurs – entreprises), moins de 10% des marchés ont plus de 1 titulaires donc l’impact est faible
Suppression des doublons
Non prise en charge de la colonne ‘Modification’

### Premier traitement – Valorisation 
Identification et suppression des montants aberrants (imputés plus tard)
Gestion des id et codes CPV manquants
Création d’une colonne permettant de connaitre le nombre de titulaires par marchés
Montant total réparti de façon uniforme en fonction du nombre de titulaires sur un marché (montant total sauvegardé dans une nouvelle colonne)
Suppression des caractères spéciaux et alphabétique dans les codes SIRET pour tenter de les récupérer
Création de deux colonnes permettant d’attribuer précisément une région et un département à chaque marché (localisation générale des acheteurs)
Mise en forme et correction des données temporelles (date/durée)
Imputation des montants en utilisant la médiane stratifiée (le plus possible), et création d’une colonne permettant d’identifier les montants imputés
Identification et rectification des durées (en mois) exprimées en jours

### Enrichissement des données 
Enrichissement des données des acheteurs et des entreprises via le code SIRET/SIREN (et dans le pire des cas grâce à la dénomination sociale des entreprises)
Si certains codes SIRET sont déjà identifiés comme faux (lors d’un lancement ultérieur du code) alors ils sont automatiquement supprimés des méthodes d’enrichissement pour gagner du temps
1er enrichissement réalisé avec le code SIRET en mergeant avec une BDD INSEE
2nd enrichissement réalisé avec le code SIREN en mergeant avec une BDD INSEE
3e enrichissement en utilisant plusieurs méthodes de scraping sur le site INFOGREFFE (en utilisant code SIRET, SIREN, et dénomination sociale)
Enrichissement via les codes CPV : identification précise de leur référence via une BDD
Ajout de la géolocalisation précise des acheteurs et des entreprises : latitude et longitude de la ville dans laquelle ils sont identifié via les codes SIRET/SIREN
La géolocalisation précise permet ensuite de calculer la distance entre les acheteurs et les entreprises pour chaque marché (utilisation de la formule de Vincenty avec le rayon moyen de la Terre)
Analyse des données
Segmentation de marché : utilisation de la classification par ascendant hiérarchique (CAH) afin de classer les acheteurs dans des clusters (au total 3 clusters principaux + quelques données hors-clusters)
Représentation cartographique des données
Données par commune, information sur le montant total, le nombre de marchés, le nombre d’entreprises, la distance médiane (acheteurs – entreprises) et la segmentation de l’acheteur
HeatMap des contrats au niveau national
Répartition des montants totaux par région
Répartition des montants / nb population par département
 

### Vérification de la qualité des données
Utilisation de l’algorithme de Luhn pour détecter les SIREN faux
Calcul du ratio nb entreprises / nb marchés (avec tous les montants et avec montants>40K)
Récapitulatif de toutes les erreurs supposées répertoriées dans le df_ERROR

### Réalisation d'un dashboard 

