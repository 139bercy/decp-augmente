import pandas as pd
import enrichissement

"""Peut prendre du temps si on ne commente pas la ligne utils.download_data_enrichissement() dans enrichissement.
Pour les tests il vaut mieux avoir les fichiers en local puis commenter la ligne et executer pytest.

Ces tests sont principalement des tests de non-régression. Qu'on s'assure que les modifications faites ne perturbent pas certains fonctionnement.
"""


def test_enrichissement_acheteur():
    df = pd.DataFrame(columns=["siret", "acheteur.id"], data=[['21740276700016', '21740276700016']])
    df = enrichissement.enrichissement_acheteur(df)
    assert df.loc[0, "libelleCommuneAcheteur"] == "TANINGES"


def test_renommage_et_recategorisation():
    df = pd.DataFrame(columns=["nomAcheteur", "idAcheteur", "sirenEtablissement", "denominationSociale_x"], data=[['MauvaisNom', '24840025100158', "877785295", "UneDenomination"],['MauvaisNom', '24840025100158', "877785295", "UneDenomination"]])
    df = enrichissement.renommage_et_recategorisation(df)
    assert (df.nomAcheteur.values == ["COMMUNAUTE D'AGGLOMERATION DU GRAND AVIGNON"]*2).all()
    assert (df.denominationUniteLegale.values == ["MK ETANCHEITE"]*2).all()
    assert (df.loc[:, "denominationSociale_x"].values == ["UneDenomination"]*2).all()


def test_apply_luhn():
    df = pd.DataFrame(columns=["idAcheteur", "sirenEtablissement", "siretEtablissement", "typeIdentifiantEtablissement"], data=[['24840025100158', "877785295", "87778529500000", "SIRET"]])   
    df = enrichissement.apply_luhn(df)
    assert df.sirenAcheteurValide.values == True
    assert df.sirenEtablissementValide.values == True
    assert df.siretEtablissementValide.values == "False" # En effet il est faux, les deux premiers sont extrait de vrais entités (cf les deux fonctions plus haut). le siret est créé sans considération dela formule.
    # Pourquoi a-t-on un string au lieu d'un boolean ? A cause du np.where où l'une des possibilités est un string. Ce qui donne une colonne de string. Donc un "False"


def test_enrichissement_cpv():
    """
    """
    df = pd.DataFrame(columns=['codeCPV'], data=['33121100-5', "45421000-4"])
    df = enrichissement.enrichissement_cpv(df)
    assert df.refCodeCPV.tolist() == ['Électroencéphalographes', 'Travaux de menuiserie']
