import sys
import os 
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import json

from azure.cosmosdb.table.tableservice import TableService
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.storage.filedatalake import DataLakeServiceClient

from .utils import addFeaturesToDatalake, create_table, insert_features_data, get_feature, insert_features, ConvertDateTime
from .features_library import feature_dict

import logging
logging.basicConfig(filename = 'log.log', level = logging.INFO)

import pyodbc
from azure.cosmosdb.table.models import Entity


# blob storage
conn_string = os.environ["blobprodConnstr"]
container_name = "scout-events" #os.environ["container_name"]
# datalake keys
adls_conn_str = os.environ["datalakeConnstr"]
service_client = DataLakeServiceClient.from_connection_string(adls_conn_str)
file_system_client = service_client.get_file_system_client(file_system="calculatefeature/stagingdata")
# table storage keys
storage_key = os.environ["faunadata"]
table_service = TableService(connection_string=storage_key)
# sql key
sql_conn = os.environ["faunadb"]

        
def main(message: func.ServiceBusMessage):
    
    body = json.loads(message.get_body().decode('utf-8'))
    file_location = body['location']
    operations_pool = body['operations']
    sessionid = body['session_id']

    details = file_location.split('/')
    serial_number, year_id, month_id, day_id, hour_id, minute_id, second_id = details[:7]
    meas_dateid = year_id + month_id + day_id
    meas_timeid = hour_id + minute_id + second_id
    meas_code = details[7].split('.')[0]

    currentDate = datetime.utcnow()
    start = currentDate - timedelta(days=0)
    calc_dateid = start.strftime('%Y%m%d')
    calc_timeid = currentDate.strftime("%H%M%S")
    
    uploadDirName = f'{serial_number}/{year_id}/{month_id}/{day_id}/{hour_id}/{minute_id}/{second_id}/'
    FileName = f'{meas_code}.parquet'

    try:
        download = get_blob(file_location)
        logging.warn(f"{file_location} found :-)")
    except:
        features=[("MissingBlob", 0, 'N/A', 1, meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid, sessionid)]
        features = pd.DataFrame(features, columns=["Name", "Segment", "Wavelength", "Value","MeasurementCode","Measdateid","Meastimeid","Calcdateid","Calctimeid", "SessionId"])
        addFeaturesToDatalake(features, uploadDirName, FileName, file_system_client)
        table_name = create_table(sessionid, table_service)
        insert_features_data(features, table_name, table_service, sql_conn)
        logging.error("Blob not found! Setting 'MissingBlob' in database")
        logging.error(f"Could not find {file_location}!")
        sys.exit(1)

    event = json.loads(download)
    channels = event["channels"]
    features = []

    # ordinal 0: 
    operations = list(set(operations_pool) & set([*feature_dict[0]]))
    fs = 20831 #body['sample_rate']
    ts = channels[0]['value']-np.min(channels[0]['value'])
    for op in operations:
        features.append((op, 0, 'N/A', feature_dict[0][op](ts, fs), meas_code,
                         meas_dateid, meas_timeid, calc_dateid, calc_timeid, sessionid))

    wavelengths, segmentIds, eventchannels = [], [], []

    # ordinal 1, 2
    for channel in channels:
        wavelengths.append(channel["wavelength"])
        segmentIds.append(channel["segmentid"])
        eventchannels.append(channel["value"])
        wavelength = str(channel['wavelength'])
        seg = channel['segmentid']
        ts = channel['value']-np.min(channel['value'])
        for ordinal in [1,2]:
            operations = list(set(operations_pool) & set([*feature_dict[ordinal]]))
            for op in operations:
                features.append((op, seg, wavelength,  feature_dict[ordinal][op](ts, fs), meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid, sessionid))

    # ordinal 3: 
    operations = list(set(operations_pool)&set([*feature_dict[3]]))
    present_segments = np.unique(segmentIds)
    eventchannels = np.array(eventchannels)
    
    for seg in present_segments:
        ws = np.where(segmentIds==seg) 
        for w in ws[0]:
            if wavelengths[w]==np.min(wavelengths): w1 = w
            else: w2 = w       
        wave = str(wavelengths[w1]) + "+" + str(wavelengths[w2])

        ts = eventchannels[(w1, w2), :] - np.min(eventchannels[(w1,w2),:])
        for op in operations:
                features.append((op, seg, wave, feature_dict[3][op](ts, fs), meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid, sessionid))

    features = pd.DataFrame(features, columns=["Name", "Segment", "Wavelength", "Value", "MeasurementCode", "Measdateid", "Meastimeid", "Calcdateid", "Calctimeid", "SessionId"])

    # save to datalake
    addFeaturesToDatalake(features, uploadDirName, FileName, file_system_client)
    # save to table storage
    table_name = create_table(sessionid, table_service)
    start = time.time()
    insert_features_data(features, table_name, table_service, sql_conn)
    duration = time.time() - start
    logging.warn(f"{duration} for insert")

def get_blob(file_location, container='scout-events'):
    connstr = conn_string
    bsclient = BlobServiceClient.from_connection_string(connstr)
    with bsclient.get_blob_client(container, file_location) as client:
            data = client.download_blob().readall()
    return data

# if __name__ == '__main__':
#     message = {"location":"128/2022/01/27/18/40/11/613d35ad8b27b000.json", 
#                "operations":["mel_media_tot", "bw_ratio_median", "WBF_sgo", "WBF_SGO_combined", "BW_ratio", "BS_ratio", "SW_ratio", "median", "mean", "max", "length", "phase_shift_error"], 
#                "sample_rate":20849, 
#                "session_id":1413}
#     main(message)