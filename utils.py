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

s3 = boto3.resource('s3', 
                aws_access_key_id=ACCESS_KEY, 
                aws_secret_access_key=SECRET_KEY, 
                region_name="eu-west-3"
                )

def download_confs():
    conf_path = "confs"
    bucket = s3.Bucket(BUCKET_NAME)
    for obj in bucket.objects.filter(Prefix=conf_path):
        bucket.download_file(obj.key, str(obj.key))
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

def write_object_file_on_s3(file_name, object_to_pickle):
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
        print(f"DataFrame flux uploadé sur le Bucket S3")
        return True
    else:
        print(f" ERROR DataFrame flux n'est PAS upload sur le Bucket S3")
        return False
