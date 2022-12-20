import wget
import os
import json

#url_decp = os.environ.get("URL_DECP")

directory_data="data"
api_key = os.environ.get("API_KEY_GASPARD")
url_v2 = "https://data.economie.gouv.fr/api/v2/catalog/datasets/decp-rama-v2/exports/json?limit=-1&offset=0&timezone=UTC&apikey="+str(api_key)
os.makedirs(directory_data, exist_ok=True)
wget.download(url_v2, os.path.join(directory_data, "decp.json"))
##

# Comme ça vient de data-eco le format du JSON est un peu différent, on le ré écrit pour le mettre dans le même format que le v1
with open(os.path.join(directory_data, "decp.json"), "r") as file:
    fv2_json = json.load(file)

fv1_json = {"marches":fv2_json}

with open(os.path.join(directory_data, "decpv2.json"), "w") as file:
    json.dump(fv1_json, file)