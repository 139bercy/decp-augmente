import json
import wget
import boto3
import json
import os
from zipfile import ZipFile 
import pathlib


url_geoflar = "https://public.opendatasoft.com/explore/dataset/geoflar-communes-2015/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"
url_cpv = "https://simap.ted.europa.eu/documents/10184/36234/cpv_2008_xls.zip"
url_departement = "https://www.insee.fr/fr/statistiques/fichier/4316069/departement2020-csv.zip"
url_region = "https://www.insee.fr/fr/statistiques/fichier/4316069/region2020-csv.zip"
url_commune = "https://www.insee.fr/fr/statistiques/fichier/5057840/commune2021-csv.zip"
url_arrondissement = "https://www.insee.fr/fr/statistiques/fichier/5057840/arrondissement2021-csv.zip"
url_stockEtablissement = "https://files.data.gouv.fr/insee-sirene/StockEtablissement_utf8.zip"
url_stockUniteLegale = "https://files.data.gouv.fr/insee-sirene/StockUniteLegale_utf8.zip"
urls = [url_geoflar, url_cpv, url_departement, url_region, url_commune, url_arrondissement, url_stockEtablissement, url_stockUniteLegale]
data_path = "data"

def load_files_and_unzip(urls):
    """
    Cette fonction télécharge les fichiers utiles pour l'enrichissement. Puis les unzip.
    """
    data_exists = os.path.exists(data_path)
    if not data_exists:
        os.mkdir(data_path)
    # Téléchargements des fichiers
    for url in urls :
        if url == url_geoflar: # Traitement spécifique pour ce téléchargement
            wget.download(url, out=os.path.join(data_path, "geoflar-communes-2015.csv" ), bar=None)
        else:
            wget.download(url, out=os.path.join(data_path, url.split('/')[-1]))
    # Unzip 
    extension = ".zip"
    for file in os.listdir(data_path):
        if file.endswith(extension): # On unzip que les fichiers zip
            file_path = os.path.join(pathlib.Path(__file__).parent, data_path, file) # Full path
            with ZipFile(file_path, 'r') as zobj:
                zobj.extractall(path=data_path)
            os.remove(file_path) # Supprime les .zip
    return None

def upload_on_s3(local_credentials="saagie_cred.json", bucket_name="bercy"):
    """
    Cette fonction se connecte au bucket S3 et y upload tous les fichiers du folder data sauf decp.json.
    """
    #Upload via s3
    local_credentials_exist = os.path.exists(local_credentials)
    if local_credentials_exist : # Dans le cas où on fait tourner ça en local
        with open(local_credentials, "r") as f:
            credentials = json.load(f)
        access_key = credentials["ACCESS_KEY"]
        secret_key = credentials["SECRET_KEY"]
        user = credentials["USER_SAAGIE"]
        password = credentials["PASSWORD_SAAGIE"]
    else :  # Sur la CI ou Saagie
        access_key = os.environ.get("ACCESS_KEY")
        secret_key = os.environ.get("SECRET_KEY")
        user = os.environ.get("USER_SAAGIE")
        password = os.environ.get("PASSWORD_SAAGIE")
    # Connexion
    s3 = boto3.resource('s3', 
                        aws_access_key_id=access_key, 
                        aws_secret_access_key=secret_key, 
                        region_name="eu-west-3"
                        )
    for file in os.listdir(data_path):
        object = s3.Object(bucket_name, file)
        full_path =  os.path.abspath(os.path.join(data_path, file))

        result = object.put(Body=open(full_path, "rb"))
        res = result.get('ResponseMetadata')
        if res.get('HTTPStatusCode') == 200:
            print(f'File {file} Uploaded Successfully')
        else:
            print(f'File {file} Not Uploaded')
    return None

def main():
    load_files_and_unzip(urls)
    upload_on_s3()



if __name__ == "__main__":
    main()
