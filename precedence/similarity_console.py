import requests
import json
import re
import os
import time
import argparse

SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'

OKTA_USERNAME=os.getenv("OKTA_USERNAME")
OKTA_PASSWORD=os.getenv("OKTA_PASSWORD")
OKTA_AUTHN_URL='https://aonaonips.okta.com/api/v1/authn'
OKTA_AUTHORIZE_URL='https://aonaonips.okta.com/oauth2/v1/authorize'
REDIRECT_URL='http://localhost:8080/implicit/callback'
CLIENT_ID='0oa8x9iepqmx4shFH356'
API_BASE_URL=os.getenv("API_BASE_URL")


def get_okta_access_token():
    headers = {
            'Accept': 'application/json', 
            'Content-Type': 'application/json'
        }
    data = {
            "username": OKTA_USERNAME, 
            "password": OKTA_PASSWORD
        }
    resp = requests.post(OKTA_AUTHN_URL, data=json.dumps(data), headers=headers)
    session_token = resp.json()['sessionToken']

    params = {
            'sessionToken': session_token, 
            'response_mode': 'fragment', 
            'redirect_uri': REDIRECT_URL, 
            'client_id': CLIENT_ID, 
            'response_type': 'token id_token', 
            'scope': 'openid profile email', 
            'state': 'testing', 
            'nonce': 'testing nonce'
        }
    resp = requests.get(OKTA_AUTHORIZE_URL, params=params, headers=headers, allow_redirects=False)
    access_token = re.search('access_token=(.*)&token_type=', str(resp.headers)).group(1)
    return access_token

def similarity(access_token, source_ids, source_type):
    body = { "source_ids": source_ids,
             "source_type": source_type
            }

    return requests.post(
            API_BASE_URL + '/similarity/jobs/',
            json=body,
            headers={'Authorization': "Bearer " + access_token}
        )

def get_status(access_token, job_id):
    assert job_id
    
    return requests.get(API_BASE_URL + f"/similarity/jobs/{job_id}/status/",
            headers={'Authorization': "Bearer " + access_token}
        )

def get_result(access_token, job_id):
    assert job_id
    
    return requests.get(API_BASE_URL + f"/similarity/jobs/{job_id}/results/",
            headers={'Authorization': "Bearer " + access_token}
        )

def parse_status(body):
    return body.get('status')

def get_job_id(body):
    return body.get('job_id')

def write_file(file_path, down_load_url):
    assert file_path
    assert down_load_url

    with open(file_path, 'wb') as f:
        f.write(requests.get(down_load_url).content)

def wait_for_job(access_token, job_id, timeout, period=5):
        force_end = time.time() + timeout
        while time.time() < force_end:
            task_resp = get_status(access_token, job_id)
            code = task_resp.status_code
            if not code:
                print(f"unexpected status: {code}")
                continue
            body = task_resp.json()
            status = parse_status(body)
            if status == SUCCESS:
                return True, status
            elif status == FAILURE:
                print(body)
                return False, status

            time.sleep(period)

        return False, None

def sc(source_type=None, source_ids=None, file_path=None):
    assert source_type
    assert source_ids
    assert file_path

    source_ids = [id.strip() for id in source_ids.split('|')]
    if not source_ids:
        raise ValueError("Could not parse source_ids")

    print("Logging into OKTA")
    access_token = get_okta_access_token()

    if access_token:
        print("Login Success")
    else:
        raise RuntimeError("Okta Login Failed")

    print("Requesting Similarity Job")
    similarity_response = similarity(access_token, source_ids,source_type)
    
    code = similarity_response.status_code
    body = similarity_response.json()
    print(body)
    assert code == 200
    job_id = get_job_id(body)
    assert job_id
    print(f"Job started, job_id = {job_id}")

    print("Waiting for Job, jobs take about 2 minutes")
    success, status = wait_for_job(access_token, job_id,600)
    print("Job Complete")
    assert success
    assert status == SUCCESS

    print("Checking Results")
    result_resp = get_result(access_token, job_id)
    body = result_resp.json()
    assert body.get('job_id')
    download = body.get('results_download')
    assert download
    print("Downloading results file")
    write_file(file_path, download)
    print(f"Results file downloaded to {file_path}")    