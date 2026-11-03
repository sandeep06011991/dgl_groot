#!/usr/bin/env bash
# mc_status.sh — RUN ON THE CONTROL MACHINE. List Massed Compute instances (name, status,
# ip, uuid) so you can find the SSH target or a uuid to terminate.
#   Prereqs: export MC_API_TOKEN=<token>
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
[ -n "${MC_API_TOKEN:-}" ] || { echo "ERROR: export MC_API_TOKEN=<token> first"; exit 2; }
curl -s --max-time 25 -H "Authorization: Bearer $MC_API_TOKEN" "$API/instance" | python3 -c "
import json,sys
d=json.load(sys.stdin)
items=d.get('instances') or d.get('data') or (d if isinstance(d,list) else [])
def find(o,keys):
    if isinstance(o,dict):
        for k,v in o.items():
            if k.lower() in keys and isinstance(v,(str,int)): return v
            r=find(v,keys);
            if r: return r
    if isinstance(o,list):
        for v in o:
            r=find(v,keys)
            if r: return r
    return None
if not items: print('(no instances)'); raise SystemExit
print(f'{\"name\":22} {\"status\":12} {\"ip\":16} uuid')
for it in items:
    print(f'{str(find(it,{\"instancename\",\"name\"})):22} {str(find(it,{\"status\",\"state\"})):12} {str(find(it,{\"ip\",\"ipaddress\",\"public_ip\",\"publicip\"})):16} {find(it,{\"uuid\",\"id\",\"instanceuuid\"})}')
"
