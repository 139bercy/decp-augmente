import nettoyage
import enrichissement
import logging
import logging.config
import cProfile
import pstats
import pickle
import argparse

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
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="run script in test mode with a small sample of data")
args = parser.parse_args()


def main():
    # vérification des arguments fournis en entrée de script, si l'argument -T est présent on lance les tests
    if args.test:
        test_check = True
    else:
        test_check = False

    logger.info("Début du script de nettoyage")
    nettoyage.main(test_check)
    logger.info("Fin du script de nettoyage")
    logger.info("Début du script d'enrichissement des données")
    enrichissement.main()
    logger.info("Fin du script d'enrichissement")


if __name__ == "__main__":
    main()
