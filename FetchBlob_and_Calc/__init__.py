import os 
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient, ServiceBusMessage

from .features_library import feature_dict

import logging
logging.basicConfig(filename = 'log.log', level = logging.INFO)


# blob storage
conn_string = os.environ["blobprodConnstr"]
container_name = "scout-events" 


sb_connection = os.environ['iaservicebus'] 
queue_service = ServiceBusClient.from_connection_string(sb_connection)
sb_topic = "calcfeatures-completed"


def get_blob(file_location, container='scout-events'):
    connstr = conn_string
    bsclient = BlobServiceClient.from_connection_string(connstr)
    with bsclient.get_blob_client(container, file_location) as client:
            data = client.download_blob().readall()
    return data

def get_details(file_location):

    details = file_location.split('/')
    serial_number, year_id, month_id, day_id, hour_id, minute_id, second_id = details[:7]
    meas_code = details[7].split('.')[0]
    uploadDirName = f'{serial_number}/{year_id}/{month_id}/{day_id}/{hour_id}/{minute_id}/{second_id}/', f'{meas_code}'
    meas_dateid = year_id + month_id + day_id
    meas_timeid = hour_id + minute_id + second_id

    currentDate = datetime.utcnow()
    start = currentDate - timedelta(days=0)
    calc_dateid = start.strftime('%Y%m%d')
    calc_timeid = currentDate.strftime("%H%M%S")

    return uploadDirName, meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid

def calculate_features(download, operations_pool, sessionid, file_location):
    
    uploadDirName, meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid = get_details(file_location)

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

    return features


def main(message: func.ServiceBusMessage):
    
    body = json.loads(message.get_body().decode('utf-8'))
    file_location = body['location']
    sessionid = body['session_id']
    uploadDirName, meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid = get_details(file_location)

    try:
        download = get_blob(file_location)
        logging.warn(f"{file_location} found :-)")
        features = calculate_features(download, body['operations'], sessionid, file_location)

    except: 
        logging.error(f"Could not find {file_location}!")
        logging.error("Blob not found! Setting 'MissingBlob' in database")

        features=[("MissingBlob", 0, 'N/A', 1, meas_code, meas_dateid, meas_timeid, calc_dateid, calc_timeid, sessionid)]
        features = pd.DataFrame(features, columns=["Name", "Segment", "Wavelength", "Value","MeasurementCode","Measdateid","Meastimeid","Calcdateid","Calctimeid", "SessionId"])
  
    msg_data = {'features':features.to_string(), 'uploadDirName':uploadDirName, 'FileName':f'{meas_code}.parquet','sessionid':sessionid} 
    msg = ServiceBusMessage(msg_data)
    sender = queue_service.get_topic_sender(topic_name=sb_topic)
    sender.send_messages(msg)


# if __name__ == '__main__':
#     message = {"location":"128/2022/01/27/18/40/11/613d35ad8b27b000.json", 
#                "operations":["mel_media_tot", "bw_ratio_median", "WBF_sgo", "WBF_SGO_combined", "BW_ratio", "BS_ratio", "SW_ratio", "median", "mean", "max", "length", "phase_shift_error"], 
#                "sample_rate":20849, 
#                "session_id":1413}
#     main(message)



## to send a pandas dataframe to azure servvice bus first convert it to string
# dataframe.to_string()
# ## theen to read it again:
# from pandas.compat import StringIO #if this doesn't work try: from io import StringIO
# df = pd.read_csv(StringIO(dfstr), sep='\s+')




