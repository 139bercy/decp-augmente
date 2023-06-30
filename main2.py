import nettoyage2
import logging.config
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
    logger.info("Application règles métier")
    nettoyage2.main()
    logger.info("csv généré dans le dossier data")

if __name__ == "__main__":
    main()
