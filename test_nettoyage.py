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
        test_with_colonne_modification_source = pd.DataFrame([[1, [{"col1Modification": "2",
                                                                    "col2": "3"
                                                                    }]
                                                               ],
                                                              [2, []]
                                                              ], columns=["id", "modifications"])
        test_with_colonne_modification_obtenu = pd.DataFrame([[1, [{"col1Modification": "2",
                                                                    "col2": "3"
                                                                    }
                                                                   ], 1, "2", "3"
                                                               ],
                                                              [2, [], 0, "", ""]
                                                              ], columns=["id", "modifications", "booleanModification", "col1Modification", "col2Modification"])
        nettoyage.prise_en_compte_modifications(test_with_colonne_modification_source)
        assert_frame_equal(test_with_colonne_modification_source, test_with_colonne_modification_obtenu)

    def test_prise_en_compte_modifications_no_modifications(self):
        test_no_colonne_modification = pd.DataFrame([1], columns=["test"])
        # Test du message d'erreur
        with pytest.raises(ValueError):
            nettoyage.prise_en_compte_modifications(test_no_colonne_modification)

    def test_fusion_source_modification(self):
        df_source = pd.DataFrame([["A", "B", "C", "D", "E"],
                                  ["F", "G", "H", "I", "J"],
                                  ["K", "L", "M", "N", ""]
                                  ], columns=["col1", "col2", "col3", "col1Modification", "col3Modification"])
        raw = df_source.iloc[0]
        raw2 = df_source.iloc[2]
        col_modification = ["col1Modification", "col3Modification"]
        dic_modif = {"col1Modification": "col1",
                     "col3Modification": "col3"
                     }
        nettoyage.fusion_source_modification(raw, df_source, col_modification, dic_modif)
        nettoyage.fusion_source_modification(raw2, df_source, col_modification, dic_modif)
        df_attendu = pd.DataFrame([["D", "B", "E", "D", "E"],
                                   ["F", "G", "H", "I", "J"],
                                   ["N", "L", "M", "N", ""]
                                   ], columns=["col1", "col2", "col3", "col1Modification", "col3Modification"])
        assert_frame_equal(df_source, df_attendu)

    def test_regroupement_marche(self):
        df_source = pd.DataFrame([["13456", "Objet1", "A", "B", "C", "D", "E", 1, "0", "Date="],
                                  ["245", "Objet2", "F", "G", "H", "I", "J", 1, "1", "Date!="],
                                  ["134567", "Objet1", "K", "L", "M", "N", "", 1, "2", "Date="],
                                  ["15", "Objet3", "K", "L", "M", "", "", 0, "3", "Date!!="]
                                  ], columns=["id", "objet", "col1", "col2", "col3", "col1Modification", "col3Modification", "booleanModification", "id_technique", "datePublicationDonnees"])
        dict_modification = {"col1Modification": "col1",
                             "col3Modification": "col3"
                             }
        df_obtenu = nettoyage.regroupement_marche(df_source, dict_modification)
        df_attendu = pd.DataFrame([["13456", "Objet1", "D", "B", "E", "D", "E", 1.0, "0", "Date=", "2", "2"],
                                   ["245", "Objet2", "I", "G", "J", "I", "J", 1.0, "1", "Date!=", "1", "1"],
                                   ["134567", "Objet1", "N", "L", "M", "N", "", 1.0, "2", "Date=", "2", "2"],
                                   ["15", "Objet3", "K", "L", "M", "", "", 0.0, "3", "Date!!=", "", "3"]
                                   ], columns=["id", "objet", "col1", "col2", "col3", "col1Modification", "col3Modification", "booleanModification", "id_technique", "datePublicationDonnees", "idtech", "idMarche"])
        assert_frame_equal(df_attendu, df_obtenu)

    def test_regroupement_marche_complet(self):
        df_source = pd.DataFrame([["Objet1", "2", "Date=", 2000, 1],
                                  ["Objet2", "1", "Date!=", 2000, 2],
                                  ["Objet1", "4", "Date=", 2000, 3],
                                  ["Objet3", "3", "Date!!=", 3000, 4],
                                  ["Objet3", "5", "Date!!=", 30001, 5]
                                  ], columns=["objet", "id", "datePublicationDonnees", "montant", "useless"])

        df_obtenu = nettoyage.regroupement_marche_complet(df_source)
        df_attendu = pd.DataFrame([["Objet1", "4", "Date=", 2000, 1, 2],
                                   ["Objet2", "1", "Date!=", 2000, 2, 1],
                                   ["Objet1", "4", "Date=", 2000, 3, 2],
                                   ["Objet3", "3", "Date!!=", 3000, 4, 1],
                                   ["Objet3", "5", "Date!!=", 30001, 5, 1]
                                   ], columns=["objet", "id", "datePublicationDonnees", "montant", "useless", "nombreTitulaireSurMarchePresume"])
        assert_frame_equal(df_obtenu, df_attendu)

    def test_manage_titulaires(self):
        donnee_tit = [{"typeIdentifiant": "SIRET",
                       "id": "39068118700014",
                       "denominationSociale": "SEDI"
                       },
                      {"typeIdentifiant": "SIRET2",
                       "id": "39068118700014",
                       "denominationSociale": "SEDI2"
                       }
                      ]
        df_source = pd.DataFrame([["To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", 0, "0", "0", donnee_tit],
                                  ["To_del", "To_del", "200", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", "To_del", np.NaN, "0", "0", [{"typeIdentifiant": "SIRET3",
                                                                                                                                                                  "id": "39068118700014",
                                                                                                                                                                  "denominationSociale": "SEDI3"
                                                                                                                                                                  }]
                                   ]],
                                 columns=['dateSignature', 'dateDebutExecution', 'valeurGlobale', 'donneesExecution', 'concessionnaires',
                                          'montantSubventionPublique', 'modifications', 'autoriteConcedante.id', 'autoriteConcedante.nom', 'idtech', "id_technique", "montant", "acheteur.id", "acheteur.nom", "titulaires"])

        df_attendu = pd.DataFrame([[0, "0", "0", 2, "SIRET", "39068118700014", "SEDI"],
                                   [0, "0", "0", 2, "SIRET2", "39068118700014", "SEDI2"],
                                   ["200", "0", "0", 1, "SIRET3", "39068118700014", "SEDI3"]
                                   ],
                                  columns=["montant", "acheteur.id", "acheteur.nom", "nbTitulairesSurCeMarche", "typeIdentifiant", "idTitulaires", "denominationSociale"])
        df_obtenu = nettoyage.manage_titulaires(df_source)
        assert_frame_equal(df_obtenu, df_attendu)

    def test_manage_duplicates(self):
        ligne1 = [1, 2, 3, 'Appel d’offres restreint', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'Ferme, actualisable', 16, 17, 18, 19, 20]
        ligne2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        colonne = ['source', '_type', 'nature', 'procedure', 'dureeMois',
                   'datePublicationDonnees', 'lieuExecution.code', 'lieuExecution.typeCode',
                   'lieuExecution.nom', 'id', 'objet', 'codeCPV', 'dateNotification', 'montant',
                   'formePrix', 'acheteur.id', 'acheteur.nom', 'typeIdentifiant', 'idTitulaires',
                   'denominationSociale']
        df1 = pd.DataFrame([ligne1], columns=colonne)
        df2 = pd.DataFrame([ligne2], columns=colonne)

        df_source = pd.concat([df1, df2, df2]).reset_index(drop=True)  # il y a donc deux doublons de lignes

        df_obtenu = nettoyage.manage_duplicates(df_source)
        ligne3 = [1, 2, 3, "Appel d'offres restreint", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'Ferme et actualisable', 16, 17, 18, 19, 20]
        df_attendu = pd.DataFrame([ligne3, ligne2], columns=colonne)
        assert_frame_equal(df_attendu, df_obtenu)

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
                                  [1, 1, float(0), True],
                                  [2000000000, 1, float(0), True],
                                  [56789, 1, float(56789), False],
                                  [555555, 1, float(0), True],
                                  [0, 1, float(0), False],
                                  [100000, 2, float(50000), True]],
                                  index=[0, 1, 2, 3, 4, 5],
                                  columns=['montant', 'nbTitulairesSurCeMarche', 'montantCalcule', 'montantEstime'])
        df_obtenu = nettoyage.manage_amount(df_test)
        assert_frame_equal(df_attendu, df_obtenu)

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
        df_obtenu = nettoyage.manage_missing_code(df_test)
        assert_frame_equal(df_attendu, df_obtenu)

    def test_manage_region(self):
        df_source = pd.DataFrame([["97130", "Code departement"],
                                  ["75000", "Code"],
                                  ["60000", "Code departement"],
                                  ["1200", "Code departement"],
                                  ["60000", "Code région"],
                                  ["11", "Code région"]
                                  ], columns=['lieuExecution.code', 'lieuExecution.typeCode'])
        df_attendu = pd.DataFrame([["97130", "Code departement", "971", "01", "Guadeloupe", "Guadeloupe"],
                                   ["75000", "Code", "75", "11", "Paris", "Île-de-France"],
                                   ["60000", "Code departement", "60", "32", "Oise", "Hauts-de-France"],
                                   ["1200", "Code departement", "12", "76", "Aveyron", "Occitanie"],
                                   ["60000", "Code région", "Valeur manquante", "Valeur manquante", "Valeur manquante", "Valeur manquante"],
                                   ["11", "Code région", "Valeur manquante", "11", "Valeur manquante", "Île-de-France"]
                                   ], columns=['lieuExecution.code', 'lieuExecution.typeCode', "codeDepartementExecution", "codeRegionExecution", "libelle", "libelleRegionExecution"])
        df_obtenu = nettoyage.manage_region(df_source)

        # gestion des na qui font planter le test
        df_obtenu.fillna("Valeur manquante", inplace=True)
        assert_frame_equal(df_attendu, df_obtenu)

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
        df_obtenu = nettoyage.manage_date(df_test)
        assert_frame_equal(df_attendu, df_obtenu)

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
        df_obtenu = nettoyage.correct_date(df_test)
        assert_frame_equal(df_attendu, df_obtenu)

    def test_data_inputation(self):
        df_source = pd.DataFrame([["Objet1", 12, False, 12, "45", 12200],
                                  ["Objet2", 120, False, 120, "45", 12200],
                                  ["Objet3", 1200, False, 1200, "45", 12200],
                                  ["Objet4", 12, False, 12, "34", 12200],
                                  ["Objet5", 120, False, 121, "34", 12200],
                                  ["Objet6", 1200, False, 1200, "34", 12200],
                                  ["Objet7", 11, False, 11, "34", 12200]
                                  ], columns=["objet", "dureeMois", "dureeMoisEstimee", "dureeMoisCalculee", "CPV_min", "montantCalcule"])
        mediane = np.median([11, 12, 121, 1200])
        df_attendu = pd.DataFrame([["Objet1", 12, 'False', 12, "45", 12200],
                                   ["Objet2", 120, 'False', 120, "45", 12200],
                                   ["Objet3", 1200, 'False', 1200, "45", 12200],
                                   ["Objet4", 12, 'False', 12, "34", 12200],
                                   ["Objet5", 120, 'True', mediane, "34", 12200],
                                   ["Objet6", 1200, 'True', mediane, "34", 12200],
                                   ["Objet7", 11, 'False', 11, "34", 12200]
                                   ], columns=["objet", "dureeMois", "dureeMoisEstimee", "dureeMoisCalculee", "CPV_min", "montantCalcule"])
        df_obtenu = nettoyage.data_inputation(df_source)
        assert_frame_equal(df_obtenu, df_attendu)

