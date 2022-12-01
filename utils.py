import os
import boto3
import json
import pickle


local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist : # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    ACCESS_KEY = credentials["ACCESS_KEY"]
    SECRET_KEY = credentials["SECRET_KEY"]
    USER =credentials["USER_SAAGIE"]
    PASSWORD = credentials["PASSWORD_SAAGIE"]
else :  # Sur la CI ou Saagie
    ACCESS_KEY = os.environ.get("ACCESS_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")
    USER =os.environ.get("USER_SAAGIE")
    PASSWORD = os.environ.get("PASSWORD_SAAGIE")
PROJECT_NAME = "BercyHub - OpenData"
BUCKET_NAME = "bercy"
USE_S3 = os.environ.get("USE_S3") # Boolean pour savoir si l'on va utiliser S3 ou non.

s3 = boto3.resource('s3', 
                aws_access_key_id=ACCESS_KEY, 
                aws_secret_access_key=SECRET_KEY, 
                region_name="eu-west-3"
                )

def download_data_nettoyage(path_json_needed="confs/config_data.json", useful_bases = ["departements-francais", "region-fr"]):
    """
    Cette fonction télécharge les bases de données utiles pour nettoyage.py

    Arguments
    -----------
    path_json_needed : chemin du JSON où sont inscris les informations des bases de données utilisés pour le script
    useful_bases : list des clefs pour savoir quelle base utiliser.
    """
    bucket = s3.Bucket(BUCKET_NAME)
    content_object = s3.Object(BUCKET_NAME, path_json_needed)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    path_data = "data"
    for base in useful_bases :
        path_base_to_download = json_content[base]
        bucket.download_file(os.path.join(path_data, path_base_to_download), os.path.join(path_data, path_base_to_download))
        print(f"{base} est téléchargé.")

    return True

def download_data_enrichissement():
    """
    Cette fonction télécharge les bases de données utiles pour enrichissement.py

    """
    download_datas() # Toutes les bases présentes dans data/ sont utiles. Data comporte égale le cache du df d'un traitement de flux à l'autre.

     # les caches également



    return True
def download_datas():
    data_path = "data/"
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=data_path):
        print(f"{obj.key} , {str(obj.key)} va se télécharger")
        bucket.download_file(obj.key, str(obj.key))
        print(f"{obj.key} , {str(obj.key)} est téléchargé")
    return True

def download_confs():
    conf_path = "confs/"
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=conf_path):
        print(f"{obj.key} , {str(obj.key)} va se télécharger")
        bucket.download_file(obj.key, str(obj.key))
        print(f"{obj.key} , {str(obj.key)} est téléchargé")
    return True

def write_cache_on_s3(path_pickle: str, object_to_pickle):
    """
    Cette fonction permet d'uploader un objet sur le Bucket S3 en format pickle et le stock dans le folder cache.

    Arguments
    -----------
    path_pickle (str) : Nom du fichier pickle que l'on va stocker sur le S3
    object_to_pickle : Objet à stocker dans le pickle

    """
    path_cache = "cache"
    path_cache_s3 = os.path.join(path_cache, path_pickle)
    object = s3.Object(BUCKET_NAME, path_cache_s3)
    pickle_byte_obj = pickle.dumps(object_to_pickle)
    response = object.put(Body=pickle_byte_obj)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Cache {path_pickle} uploadé sur le Bucket S3")
        return True
    else:
        print(f" ERROR Cache {path_pickle} n'est PAS upload sur le Bucket S3")
        return False

def write_object_file_on_s3(file_name: str, object_to_pickle):
    """
    Cette fonction permet d'uploader un objet sur le Bucket S3 en format pickle et le stocker à la racine du S3.

    Arguments
    -----------
    path_pickle (str) : Nom du fichier pickle que l'on va stocker sur le S3
    object_to_pickle : Objet à stocker dans le pickle

    """
    object = s3.Object(BUCKET_NAME, file_name)
    pickle_byte_obj = pickle.dumps(object_to_pickle)
    response = object.put(Body=pickle_byte_obj)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"Objet uploadé sur le Bucket S3")
        return True
    else:
        print(f" ERROR Objet n'est PAS upload sur le Bucket S3")
        return False




def download_file(file_name_s3: str, file_name_local: str):
    """
    Cette fonction charge un fichiers de s3.

    Arguments
    -------------
    (file_name_s3) Le nom du fichier à traiter sur s3
    (file_name_local) Le nom à donner au fichier en local
    """
    print(f"{file_name_s3} Va etre téléchargé")
    print(f"file_name_s3 {file_name_s3} file loca {file_name_local}")
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=file_name_s3):
        bucket.download_file(file_name_s3, file_name_local)
    print(f"{file_name_s3} est téléchargé")
    return None


