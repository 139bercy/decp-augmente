import nettoyage
import enrichissement
import time
import logging.config


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


def main():
    start_time = time.time()
    logger.info("Decp-augmente")
    nettoyage.main()
    logger.info("Début du script d'enrichissement des données")
    enrichissement.main()
    logger.info("Fin de l'execution de decp-augmente en {} minutes".format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
