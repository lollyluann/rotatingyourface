import requests
import hashlib
import random
import os
import json
import getpass
import io
import argparse
import traceback
import datetime

import typing as T


SERVER_URL = "100.25.4.76"
SERVER_PORT = 5000
VERSION = "0.1"

class MPResponse:
    def __init__(self, ok: bool, message: str, payload: T.Optional[T.Any]):
        self.ok = ok
        self.message = message
        self.payload = payload
    
    def __str__(self):
        s = io.StringIO()
        if self.ok:
            print("SUCCESS.", file=s)
        else:
            print("FAILURE.", file=s)
        print("Message: ", self.message, file=s)
        if self.payload is not None:
            print("Payload: ", json.dumps(self.payload, indent=4), file=s)
        s.seek(0)
        return s.read()
        

def plaintext_to_password(password:str) -> str:
    hasher = hashlib.md5()
    hasher.update(password.encode())
    return hasher.hexdigest()

def login(prompt_if_needed = True):
    my_path = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(my_path, "credentials.json")
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            creds = json.load(f)
        return creds
    elif prompt_if_needed:
        kerb = input("Kerberos? - ")
        pw = getpass.getpass("Miniplaces Password [Hidden]? - ")
        pw = plaintext_to_password(pw)
        creds = {"kerberos": kerb, "password": pw}
        with open(cache_path, "w") as f:
            json.dump(creds, f, indent=4)
        return creds
    else:
        return None

def get_url(endpoint):
    return f"http://{SERVER_URL}:{SERVER_PORT}/{endpoint}"

def server_get(endpoint, payload=None, attach_login=False):
    url = get_url(endpoint)
    if payload is None and not attach_login:
        resp = requests.get(url)
    else:
        pl = {}
        if payload is not None:
            pl.update(payload)
        if attach_login:
            pl.update(login())
        resp = requests.post(url, json=pl)
    
    try:
        body = resp.json()
        return MPResponse(body['success'], body['message'], body['payload'])
    except json.decoder.JSONDecodeError as err:
        return MPResponse(False, f"Server Failure - Expected JSON, got: {resp.content.decode()}", None)
    except Exception as err:
        traceback.print_exc()
        return MPResponse(False, f"Client Failure - {err}", None)

def create_user():
    creds = login(prompt_if_needed=False)
    assert creds is None, "Already found credentials. Delete credentials.json to log in again."

    print("Creating a user. Please enter your kerb create a password.")
    creds = login()
    conf = plaintext_to_password(getpass.getpass())
    assert conf == creds['password'], "Mismatch password. Terminating."
    return server_get("create_user", attach_login=True)

def create_team():
    team_name = input("Please enter a team name:\n")
    creds = login()

    return server_get("create_team", {"team_name": team_name}, attach_login=True)

def join_team():
    team_name_or_id = input("Please enter a team ID [start with #] or name:")
    if team_name_or_id.startswith("#"):
        team_id = int(team_name_or_id[1:])
        pl = {"team_id": team_id}
    else:
        team_name = team_name_or_id
        pl = {"team_name": team_name}
    
    join_code = int(input("Enter the join code: "))
    pl.update({"join_code": join_code})

    return server_get("join_team", pl, attach_login=True)

def my_team():
    return server_get("my_team", None, attach_login=True)

def submit():
    input_file = input("Please enter the path to the submission:\n")
    assert os.path.isfile(input_file)
    with open(input_file, "r") as f:
        answers = json.load(f)
    assert type(answers) == dict
    assert all(map(lambda k: type(k) == str, answers.keys())), "JSON should map filenames to lists of string guesses"
    assert all(map(lambda v: type(v) == list, answers.values())), "JSON should map filenames to lists of string guesses"
    assert all(map(lambda v: all(map(lambda guess: type(guess) == str, v)), answers.values())), "Found a non-string in guess list"
    return server_get("submit", {"answers": answers}, attach_login=True)

def tableify(data, headers, formatters=None) -> str:
    assert type(data) == list, "Expected data to be a list"
    assert all(map(lambda v: type(v) == list, data)), "Expected data to be a list of lists"
    assert all(map(lambda v: len(v) == len(headers), data)), "Expected data to be same length as headers"
    
    def fmt(ix, val):
        if formatters is None:
            return val
        else:
            formatter = formatters.get(headers[ix], None)
            if formatter is None:
                return str(val)
            else:
                return str(formatter(val))
        
    max_lens = [len(hdr) for hdr in headers]
    stringified = []
    stringified.append(headers[:])
    for each_row in data:
        stringified.append([])
        for (ix, each_column) in enumerate(each_row):
            pretty = fmt(ix, each_column)
            stringified[-1].append(pretty)
            if len(pretty) > max_lens[ix]:
                max_lens[ix] = len(pretty)
    
    s = io.StringIO()
    for each_row in stringified:
        for (ix, each_column) in enumerate(each_row):
            field_len = max_lens[ix]+3
            print(f"%{field_len}s" % each_column, end="", file=s)
        print("", file=s)
    
    s.seek(0)
    return s.read()
    
def time_formatter(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp) 
    return dt.strftime("%m-%d %H:%M")

def view_leaderboard():
    return server_get("leaderboard")

def view_my_recent_submissions():
    return server_get("my_submissions", {"kind": "recent"}, attach_login=True)

def view_my_best_submissions():
    return server_get("my_submissions", {"kind": "best"}, attach_login=True)

def get_server_version():
    return server_get("version")

def main():
    resp = get_server_version()
    if resp.ok:
        server_version = resp.message
        if server_version != VERSION:
            print(f"SERVER VERSION is {server_version}. CLIENT VERSION is {VERSION}. Please update.")
            exit()
    commands = {
        fn.__name__.split(".")[-1]: fn for fn in
        [create_user, create_team, join_team, my_team, submit, view_leaderboard, view_my_recent_submissions, view_my_best_submissions, get_server_version]
    }
    parser = argparse.ArgumentParser(description="Connects to Miniplaces Server")
    parser.add_argument("command", choices=commands.keys(), help="Action to perform")
    args = parser.parse_args()
    fn_to_exec = commands[args.command]
    server_return = fn_to_exec()

    if server_return.ok and args.command in ['view_leaderboard', 'view_my_recent_submissions', 'view_my_best_submissions']:
        to_print = tableify(
            server_return.payload,
            ["Team Name", "Score", "Submission ID", "Timestamp", "Period"],
            formatters={"Timestamp": time_formatter}
        )
    else:
        to_print = str(server_return)

    print(to_print)

if __name__ == "__main__":
    main()
















