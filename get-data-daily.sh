mkdir data && cd data
wget https://www.data.gouv.fr/fr/datasets/r/16962018-5c31-4296-9454-5998585496d2 -O decp.json
# Supprimer les archives que l'on a extraite
rm -rf *.zip
ls
