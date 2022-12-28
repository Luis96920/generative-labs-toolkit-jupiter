import datetime
import requests
from google.cloud import secretmanager
import json
import googleapiclient.discovery
from google.oauth2 import service_account




def send_job(route, data, job_id, db):
    payload = {
                "prompt" : data.get("params").get('prompt'),
                "n_samples" : 1,
                "n_iter" : 1,
                "H" : int(data.get("params").get("height", 512)),
                "W" : int(data.get("params").get("width", 512)), 
                "scale" : int(data.get("params").get("scale", 65))/10 ,
                "ddim_steps" : int(data.get("params").get("ddim_step", 75)), 
                "job_id" : job_id, 
                "skip_grid": True,
                "init-img":data.get("init-img", None),
                "strength":data.get("strength", 0.99)
                }

    ### Checkear IP privada en vez de publica      
    #url = "http://10.128.0.36:8081"      

    # VM IP    
    url=f"http://34.123.102.51:8081/{route}"
    print(url)

    # # Web Hook
    # url = 'https://eo11v6dgiysh2tn.m.pipedream.net'

    r = requests.post(url, json=payload)
 
    if r.status_code != 200:
        print("Error:", r.status_code)
        print("Error:", r)
        db.collection("job_collection").document(job_id).update({"status": "Error", "params":payload, 'url_to_images' : [] })

    else:
        process_end_time = datetime.datetime.now()
        process_total_time = (process_end_time-data['created_at']).total_seconds()
        db.collection("job_collection").document(job_id).update({"status": "Success", "procces_end_time" : process_end_time, "process_time" : process_total_time, "params":payload})
    return



def from_secret_manager(project_id, secret_name):

    client = secretmanager.SecretManagerServiceClient()
    request = {"name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"}
    response = client.access_secret_version(request)
    credentials = response.payload.data.decode("UTF-8")
    credentials = json.loads(credentials)

    return credentials


def connect_vm(action):
    # Functions - STOP_STATUS_START VM
    scopes = ['https://www.googleapis.com/auth/cloud-platform']
    credentials = service_account.Credentials.from_service_account_file("genLabs_sa.json", scopes=scopes)
    compute = googleapiclient.discovery.build('compute', 'v1', credentials=credentials)

    payload = {
    "PROJECT_ID":"mightyhive-data-science-poc",
    "ZONE_NAME": "us-central1-a", 
    "VM_NAME": "genlabs-stable-diffusion-v6"    
    }
    payload['action'] = action

    if payload['action'] == "stop":
        response = compute.instances().stop(project=payload['PROJECT_ID'], zone=payload['ZONE_NAME'], instance=payload['VM_NAME']).execute()
        return response

    elif payload['action'] == "start":
        response = compute.instances().start(project=payload['PROJECT_ID'], zone=payload['ZONE_NAME'], instance=payload['VM_NAME']).execute()
        return response

    elif payload['action'] == "get_status":
        response = compute.instances().get(
            project=payload['PROJECT_ID'],
            zone=payload['ZONE_NAME'],
            instance=payload['VM_NAME']).execute()
        return response
    else:
        return print("Error")