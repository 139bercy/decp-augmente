import pytest
import json
import os
import pandas as pd
import nettoyage
import numpy as np
from pandas.testing import assert_frame_equal

# Partie recuperation de données.
with open(os.path.join("confs", "config_data.json")) as f:
    conf = json.load(f)
path_to_data = conf["path_to_data"]
decp_file_name = conf["decp_file_name"]
with open(os.path.join(path_to_data, decp_file_name), encoding='utf-8') as json_data:
    data = json.load(json_data)
# On diminue la taille du fichier pour rendre le traitement plus rapide
L_data = data["marches"][:5000]
# On reforme la bonne structure
data = {"marches": L_data}
# Les 2 fonctions ci dessous sont testées
df = nettoyage.manage_modifications(data)
df = nettoyage.regroupement_marche_complet(df)


class TestNetoyageMethods:
    """Classe permettant de tester le module nettoyage"""

    def test_check_reference_files(self):
        json_test = {"key1": "departement2020.csv",
                     "key2": "decp.json",
                     "path_to_data": "data"
                     }
        value = nettoyage.check_reference_files(json_test)
        assert(not value)  # value doit etre egal à None
        json_test2 = {"path_to_data": "data",
                      "key1": "NoFile"
                      }
        with pytest.raises(ValueError):
            nettoyage.check_reference_files(json_test2)

    def test_prise_en_compte_modifications(self):
        test_no_colonne_modification = pd.DataFrame([1], columns=["test"])
        test_with_colonne_modification_input = pd.DataFrame([[1, [{"col1Modification": "2",
                                                                  "col2": "3"
                                                                   }]
                                                              ],
                                                             [2, []]
                                                             ], columns=["id", "modifications"])
        test_with_colonne_modification_output = pd.DataFrame([[1, [{"col1Modification": "2",
                                                                    "col2": "3"
                                                                    }
                                                                   ], 1, "2", "3"
                                                               ],
                                                              [2, [], 0, "", ""]
                                                              ], columns=["id", "modifications", "booleanModification", "col1Modification", "col2Modification"])
        # Test du message d'erreur
        with pytest.raises(ValueError):
            nettoyage.prise_en_compte_modifications(test_no_colonne_modification)
        nettoyage.prise_en_compte_modifications(test_with_colonne_modification_input)
        assert_frame_equal(test_with_colonne_modification_input, test_with_colonne_modification_output)

    def test_regroupement_marche(self):
        pass

    def test_manage_titulaires(self):
        df_input = df.reset_index()
        taille_input = df_input.shape
        nb_ligne_nulle = sum(df_input.titulaires.isna()&df_input.concessionnaires.isna())
        df_output = nettoyage.manage_titulaires(df_input)
        nb_ligne_finale = sum(df_input.titulaires.apply(len)) - nb_ligne_nulle
        # test sur la taille finale
        # Pour avoir le nombre de ligne finale, il faut connaitre le nombre de ligne ou titulaires est nulle
        taille_output = df_output.shape
        assert(nb_ligne_finale == taille_output[0])  # la taille finale correspond à la somme des len de titulaires
        assert(taille_output[1] == taille_input[1] - 11 + 3)  # Ajout de 3 colonnes, suppressions de 11 colonnes
        # test sur le contenu
        # Premier test sur les nouvelles colonnes
        compteur = 0
        for i in range(len(df_input.titulaires)):
            if df_input.titulaires[i] == '0':
                compteur -= 1
            else:
                nb_tit = 0
                for j in range(len(df_input.titulaires[i])):
                    nb_tit += 1  # valeur nbTitulairesSurCeMarche théorique = jmax
                    if j >= 1:
                        compteur += 1
                    valeur_initiale = df_input.titulaires[i][j]
                    if 'typeIdentifiant' in valeur_initiale.keys():
                        assert(valeur_initiale['typeIdentifiant'] == df_output.typeIdentifiant[i + compteur])
                    if 'id' in valeur_initiale.keys():
                        assert(valeur_initiale['id'] == df_output.idTitulaires[i + compteur])
                    if 'denominationSociale' in valeur_initiale.keys():
                        assert(valeur_initiale['denominationSociale'] == df_output.denominationSociale[i + compteur])

    def test_manage_duplicates(self):

        ligne = df[:5]
        ligne2 = df[:2]
        df_input = ligne.append(ligne2).append(ligne2).reset_index()  # il y a donc deux doublons de lignes
        df_input = df_input.append(df[df.procedure == 'Appel d’offres restreint']).reset_index()
        df_input = nettoyage.manage_titulaires(df_input)  # Fonction testée
        nb_ligne, nb_col = df_input.shape
        nb_ligne_finale = nb_ligne - 4  # 4 doublons
        df_output = nettoyage.manage_duplicates(df_input)
        nb_ligne_output, nb_col_output = df_output.shape
        # Test sur les tailles des entrées/sorties
        assert(nb_ligne_finale == nb_ligne_output)
        assert(nb_col == nb_col_output)
        # Test sur les 3 np.where
        # On vérifie que les formePrix sont correctement modifiés
        nb_formePrix_a_modifier = len(df_input.formePrix[df_input.formePrix == "Ferme, actualisable"])
        nb_formePrix_restant_a_modifier = len(df_output.formePrix[df_input.formePrix == "Ferme, actualisable"])
        nb_formePrix_modifie = len(df_output.formePrix[df_input.formePrix == 'Ferme et actualisable'])
        assert(nb_formePrix_a_modifier == nb_formePrix_modifie)
        assert(0 == nb_formePrix_restant_a_modifier)

    def test_is_false_amount(self):
        L = [567893256, 444444444, 3456.55555, 55555645.678, 55655655, 10000000]  # G, F, G, F, G
        L_attendu = [False, True, False, True, True, False]
        for i in range(len(L)):
            assert(L_attendu[i] == nettoyage.is_false_amount(L[i]))

    def test_manage_amount(self):
        df_test = pd.DataFrame([
                               [1, 1],
                               [2000000000, 1],
                               [56789, 1],
                               [555555, 1],
                               [0, 1],
                               [100000, 2]],
                               index=[0, 1, 2, 3, 4, 5],
                               columns=['montant', 'nbTitulairesSurCeMarche'])
        df_attendu = pd.DataFrame([
                                  [float(0), 1, 1, True],
                                  [float(0), 1, 2000000000, True],
                                  [float(56789), 1, 56789, False],
                                  [float(0), 1, 555555, True],
                                  [float(0), 1, 0, False],
                                  [float(50000), 2, 100000, True]],
                                  index=[0, 1, 2, 3, 4, 5],
                                  columns=['montantCalcule', 'nbTitulairesSurCeMarche', 'montantOriginal', 'montantEstime'])
        df_output = nettoyage.manage_amount(df_test)
        assert_frame_equal(df_attendu, df_output)

    def test_manage_missing_code(self):
        df_test = pd.DataFrame([
                               ['AETYHGF', np.nan, 'SIRET', "123456789", "Fournitures", "Coucou"],
                               [np.nan, '46562345', 'SIRET', "4444455555", "Fournitures", np.nan]],
                               index=[0, 1],
                               columns=['id', 'codeCPV', 'typeIdentifiant', 'idTitulaires', 'natureObjet', 'denominationSociale'])
        df_attendu = pd.DataFrame([
                                  ['AETYHGF', '00000000', 'SIRET', "123456789", "Fournitures", "Coucou", "56789", "00"],
                                  ['0000000000000000', '46562345', 'SIRET', "4444455555", 'Services', np.nan, "55555", "46"]],
                                  index=[0, 1],
                                  columns=['id', 'codeCPV', 'typeIdentifiant', 'idTitulaires', 'natureObjet', 'denominationSociale', 'nic', "CPV_min"])
        df_output = nettoyage.manage_missing_code(df_test)
        assert_frame_equal(df_attendu, df_output)

    def test_manage_region(self):
        pass

    def test_manage_date(self):
        # aaaa-mm-jj
        df_test = pd.DataFrame([
            ["1996-12-04", "2000-01-01"],
            ["1946-12-04", "2120-08-30"],
            ["2101-03-31", "1940-09-20"],
            ["", ""]],
            index=[0, 1, 2, 3],
            columns=["datePublicationDonnees", "dateNotification"])
        df_attendu = pd.DataFrame([
            ["1996-12-04", "2000-01-01", "2000", "01"],
            ["1946-12-04", np.NaN, str(np.NaN), np.NaN],
            ["2101-03-31", np.nan, str(np.NaN), np.nan],
            [np.nan, np.nan, str(np.NaN), np.nan]],
            index=[0, 1, 2, 3],
            columns=["datePublicationDonnees", "dateNotification", "anneeNotification", "moisNotification"])
        df_output = nettoyage.manage_date(df_test)
        assert_frame_equal(df_attendu, df_output)

    def test_correct_date(self):
        df_test = pd.DataFrame([
            [1000000, 48],
            [12, 100],
            [345, 360],
            [3000000, 123],
            [1, 1]],
            index=[0, 1, 2, 3, 4],
            columns=["montantCalcule", "dureeMois"])
        df_attendu = pd.DataFrame([
            [1000000, 48, str(False), 48.0],
            [12, 100, str(True), 3.0],
            [345, 360, str(True), 12.0],
            [3000000, 123, str(False), 123.0],
            [1, 1, str(True), 1.0]],
            index=[0, 1, 2, 3, 4],
            columns=["montantCalcule", "dureeMois", "dureeMoisEstimee", "dureeMoisCalculee"])
        df_output = nettoyage.correct_date(df_test)
        assert_frame_equal(df_attendu, df_output)
