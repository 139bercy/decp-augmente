mkdir data && cd data
wget 'https://public.opendatasoft.com/explore/dataset/geoflar-communes-2015/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B' -O geoflar-communes-2015.csv
#wget 'https://public.opendatasoft.com/explore/dataset/code-insee-postaux-geoflar/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B' -O code-insee-postaux-geoflar.csv
wget https://simap.ted.europa.eu/documents/10184/36234/cpv_2008_xls.zip
unzip cpv_2008_xls.zip
wget https://www.insee.fr/fr/statistiques/fichier/4316069/departement2020-csv.zip
unzip departement2020-csv.zip
wget https://www.insee.fr/fr/statistiques/fichier/4316069/region2020-csv.zip
unzip region2020-csv.zip
wget https://www.insee.fr/fr/statistiques/fichier/5057840/commune2021-csv.zip
unzip commune2021-csv.zip
wget https://www.insee.fr/fr/statistiques/fichier/5057840/arrondissement2021-csv.zip
unzip arrondissement2021-csv.zip
wget https://files.data.gouv.fr/insee-sirene/StockEtablissement_utf8.zip
unzip StockEtablissement_utf8.zip
wget https://files.data.gouv.fr/insee-sirene/StockUniteLegale_utf8.zip
unzip StockUniteLegale_utf8.zip
# Supprimer les archives que l'on a extraite
rm -rf *.zip
ls
