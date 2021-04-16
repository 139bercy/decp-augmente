import nettoyage
import enrichissement
import logging

#logging.config.fileConfig('logging.conf')
# create logger
#logger = logging.getLogger('main.py')

#logging.config.fileConfig('logging.conf')
# create logger
#logger2 = logging.getLogger('nettoyage.py')

def main():
    #logger.info("Début du script de nettoyage")
    nettoyage.main()
    #logger.info("Fin du script de nettoyage")
    #logger.info("Début du script d'enrichissement des données")
    enrichissement.main()
    #logger.info("Fin du script d'enrichissement")


if __name__ == "__main__":
    main()
