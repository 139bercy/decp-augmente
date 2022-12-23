import os
import boto3
import json
import pickle
import botocore


local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist : # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    ACCESS_KEY = credentials["ACCESS_KEY"]
    SECRET_KEY = credentials["SECRET_KEY"]
    USER =credentials["USER_SAAGIE"]
    PASSWORD = credentials["PASSWORD_SAAGIE"]
    ENDPOINT_S3 = credentials["ENDPOINT_S3"]
    PROJECT_NAME = credentials["PROJECT_NAME"]
    BUCKET_NAME = credentials["BUCKET_NAME"]
else :  # Sur la CI ou Saagie
    ACCESS_KEY = os.environ.get("ACCESS_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")
    USER =os.environ.get("USER_SAAGIE")
    PASSWORD = os.environ.get("PASSWORD_SAAGIE")
    ENDPOINT_S3 = os.environ.get("ENDPOINT_S3")
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    
USE_S3 = os.environ.get("USE_S3") # Boolean pour savoir si l'on va utiliser S3 ou non.
USE_S3 = True
s3 = boto3.resource(service_name = 's3', 
                aws_access_key_id=ACCESS_KEY, 
                aws_secret_access_key=SECRET_KEY, 
                region_name="gra",
                endpoint_url="https://"+str(ENDPOINT_S3)
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
        bucket.download_file(os.path.join(path_data, path_base_to_download), path_base_to_download)
        if not(os.path.exists(path_data)): # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(path_data)
        os.replace(path_base_to_download , os.path.join(path_data, path_base_to_download))
        # /!\ Ne pas modifier le téléchargement à la racine puis le déplacement du fichier, sinon S3 créé des temp files qui bug le téléchargement.
        print(f"{base} est téléchargé.")

    return True

def download_data_enrichissement():
    """
    Cette fonction télécharge les bases de données utiles pour enrichissement.py

    """
    download_datas() # Toutes les bases présentes dans data/ sont utiles. Data comporte égale le cache du df d'un traitement de flux à l'autre.
    download_cache() # les caches également

    return True

def download_cache():
    print('Download du cache')
    cache_path = "cache"
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=cache_path):
        print(f"{obj.key} , {str(obj.key)} va se télécharger")
        path, filename = os.path.split(obj.key)
        if not(os.path.exists(cache_path)): # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(cache_path)
        bucket.download_file(obj.key, filename)
        os.replace(filename , os.path.join(cache_path, filename))
        print(f"{obj.key} , {str(obj.key)} est téléchargé")
    pass
def download_datas():
    data_path = "data"
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=data_path):
        print(f"{obj.key} , {str(obj.key)} va se télécharger")
        path, filename = os.path.split(obj.key)
        bucket.download_file(obj.key, filename)
        if not(os.path.exists(data_path)): # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(data_path)
        os.replace(filename , os.path.join(data_path, filename))
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
        print(f"Objet {file_name} uploadé sur le Bucket S3")
        return True
    else:
        print(f" ERROR Objet n'est PAS upload sur le Bucket S3")
        return False



def download_file(file_name_s3: str, file_name_local:str, verbose=False):
    """
    Cette fonction charge un fichiers de s3.

    Arguments
    -------------
    (file_name_s3) Le nom du fichier à traiter sur s3
    (file_name_local) Le nom à donner au fichier en local
    """
    bucket = s3.Bucket(BUCKET_NAME) 
    path, filename = os.path.split(file_name_local) # On télécharge d'abord à la racine du répertoire courant. Puis on déplace. Sinon erreur s3.
    if verbose:
        print(f"Fichier {file_name_s3} téléchargé vers {filename}")
    bucket.download_file(file_name_s3, filename)
    if "/" in file_name_local:
        if not(os.path.exists(path)): # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
            os.mkdir(path)
        os.replace(filename , os.path.join(path, filename))
        if verbose:
            print(f"fichier{filename} déplacé vers {os.path.join(path, filename)}")
    return None

def get_object_content(file_name_s3: str):
    """
    Cette fonction retourne le contenu de l'objet correspondant sur S3    
    """
    bucket = s3.Bucket(BUCKET_NAME)
    object = s3.Object(BUCKET_NAME, file_name_s3)
    try:
        object.load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"L'objet {file_name_s3} recherché n'existe pas")
            return None
        else :
            print(f"L'objet {file_name_s3} existe mais il y a un problème")
            return None
    if file_name_s3.endswith("json"):
        object_content = object.get()['Body'].read().decode('utf-8')
        return json.loads(object_content)
    if file_name_s3.endswith("pkl"):
        object_content = object.get()['Body'].read()
        return pickle.loads(object_content)
    else :
        print(f"{file_name_s3} n'est ni un pickle ni un json. On le considère comme un pickle")
        object_content = object.get()['Body'].read()
        return pickle.loads(object_content)