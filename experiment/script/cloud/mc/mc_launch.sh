#!/usr/bin/env bash
# mc_launch.sh — RUN ON THE CONTROL MACHINE (laptop/WSL). Launch a Massed Compute GPU
# VM via the VM API and print how to SSH in. Massed Compute replaces RunPod when A100-SXM
# capacity there is dry; see RUNBOOK §"Massed Compute".
#
# Prereqs: export MC_API_TOKEN=<your massed compute API token>  (never commit it).
#   Usage:  mc_launch.sh [instance-name]
#
# Env (defaults in parens):
#   PRODUCT     (gpu_8x_A100_SXM4)   product from GET /gpu-inventory (8x A100 SXM4 80GB)
#   IMAGE_ID    (84)                 "Ubuntu Server 22.04 w/ drivers" (GET /images)
#   SSH_KEY_ID  (auto)               uuid of a registered key (GET /ssh-keys); auto = first
#   VOLUME_IDS  ()                   comma list of persistent-storage volume ids (beta,
#                                    max 8). API field is `volumeIds` (volume_ids -> 422).
#                                    Each attaches at LAUNCH ONLY and mounts on the VM at
#                                    /persistent/<volume-name>. List ids via GET
#                                    /storage-volumes (also: experiment/script/cloud/mc/mc_volume.sh).
#   REGION      (any)                regionName; MC has no datacenter concept
#   POLL_SECS   (15)  MAX_POLLS (40) how long to wait for the VM to get an IP
#
# On success prints POD_ID, the ssh command (using ~/.ssh/id_ed25519), and exits 0.
set -uo pipefail
API="https://vm.massedcompute.com/api/v1"
NAME="${1:-groot-sec1-$(date +%s)}"
PRODUCT="${PRODUCT:-gpu_8x_A100_SXM4}"
IMAGE_ID="${IMAGE_ID:-84}"
REGION="${REGION:-any}"
POLL_SECS="${POLL_SECS:-15}"; MAX_POLLS="${MAX_POLLS:-40}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"

[ -n "${MC_API_TOKEN:-}" ] || { echo "ERROR: export MC_API_TOKEN=<token> first"; exit 2; }
AUTH=(-H "Authorization: Bearer $MC_API_TOKEN")
jq_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)" 2>/dev/null; }

# 1) token sanity
if ! curl -s --max-time 20 -X POST "${AUTH[@]}" "$API/account/token/validation" | grep -qi valid; then
  echo "ERROR: MC_API_TOKEN failed validation"; exit 2
fi

# 2) capacity + price preflight
inv=$(curl -s --max-time 25 "${AUTH[@]}" "$API/gpu-inventory")
read -r cap price < <(echo "$inv" | python3 -c "
import json,sys
d=json.load(sys.stdin)['gpu_inventory'].get('$PRODUCT',{})
it=d.get('instance_type',{})
print(d.get('capacity_available',0), (it.get('price_cents_per_hour',0) or 0)/100)
")
echo "product=$PRODUCT  capacity=${cap:-?}  price=\$${price:-?}/hr  image=$IMAGE_ID  region=$REGION"
[ "${cap:-0}" -gt 0 ] 2>/dev/null || { echo "ERROR: no capacity for $PRODUCT right now"; exit 1; }

# 3) resolve an SSH key NAME (first registered, unless SSH_KEY_NAME given).
#    NOTE: /instance/launch's sshKeys takes the key *name* (not the uuid from /ssh-keys).
SSH_KEY_NAME="${SSH_KEY_NAME:-$(curl -s --max-time 20 "${AUTH[@]}" "$API/ssh-keys" | jq_get "d['sshKeys'][0]['name']")}"
[ -n "${SSH_KEY_NAME:-}" ] || { echo "ERROR: no registered SSH key (GET /ssh-keys empty)"; exit 2; }

# 4) build + send launch
#    volumeIds (camelCase) attaches persistent-storage volumes AT LAUNCH; they mount at
#    /persistent/<name> on the VM. Strip whitespace so "id1, id2" -> ["id1","id2"].
vol_json=""
if [ -n "${VOLUME_IDS:-}" ]; then
  ids_clean=$(echo "$VOLUME_IDS" | tr -d '[:space:]')
  vol_json=",\"volumeIds\":[\"$(echo "$ids_clean" | sed 's/,/","/g')\"]"
  echo ">>> attaching persistent volume(s): $ids_clean  (mount at /persistent/<name>, launch-time only)"
fi
body="{\"imageId\":$IMAGE_ID,\"productName\":\"$PRODUCT\",\"regionName\":\"$REGION\",\"instanceName\":\"$NAME\",\"sshKeys\":[\"$SSH_KEY_NAME\"]$vol_json}"
echo ">>> launching: $body"
resp=$(curl -s --max-time 40 -w $'\n%{http_code}' -X POST "${AUTH[@]}" -H "Content-Type: application/json" -d "$body" "$API/instance/launch")
code=$(echo "$resp" | tail -n1); payload=$(echo "$resp" | sed '$d')
echo "launch HTTP $code: $payload"
if [ "$code" != "202" ] && [ "$code" != "200" ] && [ "$code" != "201" ]; then
  echo "ERROR: launch rejected (402=raise spend limit; 403=feature/capacity; 422=bad field)"; exit 1
fi

# 5) poll GET /instance for this name until it has an IP
echo ">>> waiting for IP (every ${POLL_SECS}s, up to $((POLL_SECS*MAX_POLLS))s)"
for i in $(seq 1 "$MAX_POLLS"); do
  sleep "$POLL_SECS"
  all=$(curl -s --max-time 20 "${AUTH[@]}" "$API/instance")
  line=$(echo "$all" | python3 -c "
import json,sys,re
d=json.load(sys.stdin)
items=d.get('runningInstances') or d.get('instances') or d.get('data') or (d if isinstance(d,list) else [])
def find(o,keys):
    if isinstance(o,dict):
        for k,v in o.items():
            if k.lower() in keys and isinstance(v,(str,int)): return v
            r=find(v,keys)
            if r: return r
    if isinstance(o,list):
        for v in o:
            r=find(v,keys)
            if r: return r
    return None
for it in items:
    nm=find(it,{'instancename','name'})
    if nm=='$NAME':
        ip=find(it,{'ip','ipaddress','public_ip','publicip'})
        user=find(it,{'username','user','ssh_user','sshusername'}) or 'Ubuntu'
        uuid=find(it,{'uuid','id','instanceuuid'})
        status=find(it,{'status','state'})
        print(f'{ip or \"\"}|{user}|{uuid or \"\"}|{status or \"\"}')
        break
")
  ip=$(echo "$line" | cut -d'|' -f1); user=$(echo "$line" | cut -d'|' -f2)
  uuid=$(echo "$line" | cut -d'|' -f3); status=$(echo "$line" | cut -d'|' -f4)
  echo "  [try $i/$MAX_POLLS] status=${status:-?} ip=${ip:-none} uuid=${uuid:-?}"
  if [ -n "$ip" ]; then
    echo "POD_ID=$uuid"
    echo "POD_SSH=$ip"
    echo "SSH: ssh -i $SSH_KEY_FILE -o StrictHostKeyChecking=no ${user}@${ip}"
    echo "TERMINATE (stop billing): MC_API_TOKEN=... bash experiment/script/cloud/mc/mc_terminate.sh $uuid"
    exit 0
  fi
done
echo "WARN: no IP after polling; check 'mc_status.sh' — instance may still be provisioning."
exit 1
