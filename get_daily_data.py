import wget
import os
import boto3
import json

url_decp = "https://www.data.gouv.fr/fr/datasets/r/16962018-5c31-4296-9454-5998585496d2"
output_directory_local = r"decp.json"
output_directory_s3 = r"data/decp.json" 
json_decp = wget.download(url_decp, output_directory_local)

access_key = os.environ.get("ACCESS_KEY")
secret_key = os.environ.get("SECRET_KEY")

local_credentials="saagie_cred.json"
local_credentials_exist = os.path.exists(local_credentials)
if local_credentials_exist :  # Dans le cas où on fait tourner ça en local
    with open(local_credentials, "r") as f:
        credentials = json.load(f)
    access_key = credentials["ACCESS_KEY"]
    secret_key = credentials["SECRET_KEY"]
    user = credentials["USER_SAAGIE"]
    password = credentials["PASSWORD_SAAGIE"]
else :  # Sur la CI ou Saagie
    access_key = os.environ.get("ACCESS_KEY")
    secret_key = os.environ.get("SECRET_KEY")
    user = os.environ.get("USER_SAAGIE")
    password = os.environ.get("PASSWORD_SAAGIE")
s3_resource = boto3.resource('s3', 
                      aws_access_key_id=access_key, 
                      aws_secret_access_key=secret_key, 
                      region_name="eu-west-3"
                      )
with open(output_directory_local, encoding='utf-8') as json_data:
    data = json.load(json_data)
s3object  = s3_resource.Object('bercy', output_directory_s3)
result = s3object.put(Body=bytes(json.dumps(data).encode('utf-8')))
res = result.get('ResponseMetadata')

if res.get('HTTPStatusCode') == 200:
    print('File DECP.JSON Uploaded Successfully')
else:
    print('File DECP.JSON Not Uploaded')
