import csv
import json
import os
import pickle
import utils
import subprocess


PATH_FILE_CONFIG = "confs/config_data.json"
local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist :  # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    ACCESS_KEY = credentials["ACCESS_KEY"]
    SECRET_KEY = credentials["SECRET_KEY"]
    USER =credentials["USER_SAAGIE"]
    PASSWORD = credentials["PASSWORD_SAAGIE"]
    USER_DATAECO = credentials["USER_DATAECO"]
    PWD_DATAECO = credentials["PASSWORD_DATAECO"]
    HOST_DATAECO = credentials["HOST_DATAECO"]
else :  # Sur la CI ou Saagie
    ACCESS_KEY = os.environ.get("ACCESS_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")
    USER =os.environ.get("USER_SAAGIE")
    PASSWORD = os.environ.get("PASSWORD_SAAGIE")
    USER_DATAECO = os.environ.get("USER_DATAECO")
    PWD_DATAECO = os.environ.get("PASSWORD_DATAECO")
    HOST_DATAECO = os.environ.get("HOST_DATAECO")
 
if utils.USE_S3:
    res = utils.download_file(PATH_FILE_CONFIG, PATH_FILE_CONFIG)
    pass
with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)

path_file_to_upload = "decp_augmente_flux_final.pkl"
if utils.USE_S3:
    utils.download_file(path_file_to_upload, path_file_to_upload, verbose=True)
    pass
else:
    print(f"Vous allez upload {path_file_to_upload} depuis votre version local")
# Ouverture du pkl et conversion en CSV
with open(path_file_to_upload, "rb") as f:
    df = pickle.load(f)
path_file_to_upload_csv = path_file_to_upload[:-4]+".csv"
df.to_csv(path_file_to_upload_csv, quoting=csv.QUOTE_NONNUMERIC, sep=";", index=False)
bash_cmd = [f" lftp -u {USER_DATAECO}:{PWD_DATAECO} {HOST_DATAECO} -e 'set ftp:ssl-force true ; set ssl:verify-certificate false;cd decp; put {path_file_to_upload_csv}; quit'"] # Je n'ai pas trouvé de biblio ftp python satisfaisante. Donc ce sera en bash
subprocess.call(bash_cmd, shell=True)
# Commande bash à utiliser pour upload en ftp
