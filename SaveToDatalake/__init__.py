import os
from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd
from pandas.compat import StringIO #if this doesn't work try: from io import StringIO

import azure.functions as func

import logging
logging.basicConfig(filename = 'log.log', level = logging.INFO)

# datalake keys
adls_conn_str = os.environ["datalakeConnstr"]
service_client = DataLakeServiceClient.from_connection_string(adls_conn_str)
file_system_client = service_client.get_file_system_client(file_system="calculatefeature/stagingdata")


def addFeaturesToDatalake(feature_df, uploadDirName, FileName, file_system_client):

    try:
        feature_df = feature_df.to_parquet(engine='pyarrow')
        directory_client = file_system_client.create_directory(uploadDirName)
        file_client = directory_client.create_file(FileName)
        file_client.upload_data(feature_df, overwrite=True)
        logging.info("File uploaded to DataLake")

    except Exception as err:
        logging.error(f"{err} Error in addFeaturesToDatalake")


def main(message: func.ServiceBusMessage):

    body = message.get_body().decode('utf-8')
    features_str = body['features']
    features = pd.read_csv(StringIO(features_str), sep='\s+')
    uploadDirName = body['uploadDirName']
    FileName = body['FileName']

    # save to datalake
    addFeaturesToDatalake(features, uploadDirName, FileName, file_system_client)
