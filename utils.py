import os
import logging
import pandas as pd

logger = logging.getLogger("main.utils")
logger.setLevel(logging.DEBUG)


def save_csv(df: pd.DataFrame, file_name: str):
    """
    Cette fonction permet de sauvegarder un dataframe en csv

    Arguments
    -------------
    df : dataframe à sauvegarder
    file_name : nom du fichier à sauvegarder
    """
    path = "data_eclatees"

    if not (os.path.exists(path)):  # Si le chemin data n'existe pas
        os.mkdir(path)

    df.to_csv(os.path.join(path, file_name), index=False, sep=";")
    return None
