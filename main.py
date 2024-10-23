import nettoyage
import logging.config
import argparse
import utils
import data_management

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler("decp_augmente.log", maxBytes=100000000, backupCount=5)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# Initialize parser
args = utils.parse_args()

def main(data_format:str = '2022'):
    
    logger.info(f"Téléchargement des fichiers de données")
    data_management.main()
    logger.info("Fichiers mis à jour dans le dossier data")

    logger.info(f"Application règles métier format {data_format}")
    nettoyage.main(data_format)
    logger.info("csv généré dans le dossier data")

    # logger.info("Enrichissement des données")
    # enrichissement2.main()
    # logger.info("csv enrichi dans le dossier data")
    if not args.test and not args.local:
        utils.export_all_csv(data_format,args.local)

if __name__ == "__main__":
    all_data_format = ['2022']
    for data_format in all_data_format:
        try:
            main(data_format)
        except ValueError as e:
            print(f"Erreur lors du traitement du format {data_format}: {e}")
