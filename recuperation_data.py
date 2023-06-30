import pandas as pd
import os
import numpy as np
import pickle
import json
import datetime
import logging.handlers
from pandas.util import hash_pandas_object
from pandas import json_normalize
import argparse
import utils

logger = logging.getLogger("main.gestion_flux")
logger.setLevel(logging.DEBUG)

path_to_conf = "confs"
if not (os.path.exists(path_to_conf)):  # Si le chemin confs n'existe pas (dans le cas de la CI et de Saagie)
    os.mkdir(path_to_conf)
# Chargement des fichiers depuis le S3:
res = utils.download_confs()
if res:
    logger.info("Chargement des fichiers confs depuis le S3")
else:
    logger.info("ERROR Les fichiers de confs n'ont pas pu être chargés")

with open(os.path.join("confs", "config_data.json")) as f:
    conf_data = json.load(f)

with open(os.path.join("confs", "var_glob.json")) as f:
    conf_glob = json.load(f)

with open(os.path.join("confs", "var_debug.json")) as f:
    conf_debug = json.load(f)["nettoyage"]

path_to_data = conf_data["path_to_data"]
if not (os.path.exists(path_to_data)):  # Si le chemin data n'existe pas (dans le cas de la CI et de Saagie)
    os.mkdir(path_to_data)
decp_file_name = conf_data["decp_file_name"]


def main():
    decp_path = os.path.join(path_to_data, decp_file_name)
    if utils.USE_S3:
        data = utils.get_object_content(decp_path)
    else:
        # utilisation du fichier local decp.json téléchargé sur data.gouv.fr
        logger.info("Ouverture du fichier decp.json")
        with open("data/decpv2.json", encoding='utf-8') as json_data:
            data = json.load(json_data)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="run script in test mode with a small sample of data")
    args = parser.parse_args()

    if args.test:  # Dans le cas de la CI
        bucket = utils.s3.Bucket(utils.BUCKET_NAME)
        bucket.objects.filter(Prefix="data/hash_keys").delete()
        seed = int(os.environ.get('SEED'))
        np.random.seed(seed)
        n_subset = int(os.environ.get("TAILLE_SUBSET"))
        random_i = list(np.random.choice(len(data["marches"]), n_subset))
        accessed_mapping = map(data['marches'].__getitem__, random_i)
        accessed_list = list(accessed_mapping)
        data['marches'] = accessed_list

    df_decp = json_normalize(data['marches'])
    with open("data/decpv2.pkl", "wb") as file:
        pickle.dump(df_decp, file)
    return df_decp


if __name__ == '__main__':
    main()