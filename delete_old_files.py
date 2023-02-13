import utils
import date
from datetime import date
from dateutil.relativedelta import relativedelta

two_month_ago = date.today() + relativedelta(months=-2) # Date à partir de laquelle on veut supprimer les fichiers

prefix_files_to_delete = ["data/df_cache", "data/hash", "df_nettoye", "df_flux"] # Prefix des noms des fichiers qu'on veut delete
response = utils.s3.meta.client.list_objects_v2(Bucket=utils.BUCKET_NAME)
old_keys = [{'Key': object['Key']} 
                  for object in response['Contents'] 
                  if object['LastModified'].date()<two_month_ago
                 ]
keys_to_delete = []
for d_key in old_keys:
    key = d_key['Key']
    if key.startswith(tuple(prefix_files_to_delete)):
        keys_to_delete.append(key)
utils.s3.meta.client.delete_objects(Bucket=utils.BUCKET_NAME, Delete={'Objects': keys_to_delete})