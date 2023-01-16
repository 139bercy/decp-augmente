import pandas as pd
import enrichissement



def test_enrichissement_acheteur():
    df = pd.DataFrame(columns=["siret"], data=[['21740276700016', '21740276700016']])
    df = enrichissement.enrichissement_acheteur(df)
    print(df)
    assert df.loc[0, "libelleCommuneAcheteur"] == "TANINGES"

