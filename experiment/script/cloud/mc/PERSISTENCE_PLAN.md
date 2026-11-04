# MC environment + data persistence — status & reference

**Status: DONE & VALIDATED (2026-11-03).** Persistence is live: the `mc-persistent-storage`
implementation is **merged into `final_dissertation`**, the env snapshot is in B2 and on the
`groot-sec1` volume, and a cold restart was validated on a 1×A100 (details below). The
day-to-day procedure is **RUNBOOK §0b (COLD vs WARM)** — this file is the record of what was
built + the MC-specific facts/gotchas behind it.

## Why
MC has **no Stop/pause** — an idle VM bills the full $11.04/hr until `terminate`, which
destroys the VM disk. The only way to get "almost like a stop" (fast, stable restart
without paying the ~40-min rebuild) is: **persist the built env + data on a persistent
volume that survives terminate, then terminate when idle and restore on the next
launch.** The conda env + compiled `.so` cannot *run* off the geesefs volume (object-FUSE),
so we store a **tarball** on the volume and restore it to local `/home` at bring-up.

## VALIDATED 2026-11-03 (1×A100 + volume, ~$0.50)
A one-off 1×A100 launch (cheapest MC box) with `VOLUME_IDS=12cd1bbd-…` confirmed the whole path:
- **Launch-time attach works** — `groot-sec1` mounted live at `/persistent/groot-sec1`
  (`fuse.geesefs`, world-writable, systemd `persistent-groot-sec1.mount`). The earlier empty
  read on the just-attached box was only boot timing; it mounts a few seconds after boot.
- **Fast restore skips the build** — `mc_bootstrap.sh` restored the prebuilt env from B2 and
  logged `[3-5/6] SKIPPED (restored prebuilt env)`, `import dgl` OK — minutes, not ~40 min.
- **Volume seeded + survives terminate** — env tarball copied onto the volume (7,643,476,610 B,
  exact match to B2); after `terminate`, the volume is still `status=active` with the tarball.

Two gotchas hit + fixed (encoded in `mc_volume.sh seed-from-b2`):
1. **Never `rclone copyto B2 → the geesefs mount`** — geesefs rejects rclone's `chtimes`
   ("operation not permitted"). Download to a local dir, then plain `cp` onto the volume.
2. **Stage locally under `$HOME`, not `/home`** — on a fresh VM `/home` is root-owned until
   `mc_bootstrap` chowns it, so an `rclone`/download to `/home/...` is permission-denied.

## Lesson (why persistence wasn't on from the first launch, 2026-11-03)
The first VM was launched **bare** (no `VOLUME_IDS`) because no volume existed yet, and
MC **attach is launch-only** — a volume attached to an already-running VM does NOT hot-mount
(it only fills `attachedTo` metadata; it mounts on the next launch that passes its id). So:
**the FIRST launch of a project should create the volume first, then launch with it** — never
launch bare intending to attach later. Create a volume via the MC web console (or
`POST /api/v1/storage-volumes`), then `VOLUME_IDS=<id> bash mc_launch.sh <name>`. This was a
documentation gap (no Massed Compute runbook / no "create+attach at launch" step); it is now
documented here and in RUNBOOK §0b.

## The asset that already exists
- Volume **`groot-sec1`**, id **`12cd1bbd-5fc8-4f74-86fe-7b106a5ad02f`**, `mountPath
  /persistent/groot-sec1`, survives terminate. **Attach is launch-only** (`volumeIds` at
  `mc_launch.sh`) — it will NOT mount on a VM it was attached to after launch.
- Code (now **merged into `final_dissertation`**): `mc_volume.sh` (detect/snapshot/restore
  env+data, `seed-from-b2`), `mc_bootstrap.sh` restore fast-path (volume → B2 → build),
  `mc_launch.sh` `VOLUME_IDS`, corrected `mc_terminate.sh` billing note, RUNBOOK §0b.
- Env snapshot `mc_a100_sm80.tar.zst` (7.2 GB) is in **B2** (`gsplit-data/groot/env/`) **and on
  the volume** (`/persistent/groot-sec1/groot/env/`). papers100M topology + `w4` map are in B2.

## How this was enabled (done 2026-11-03; kept for reference)

The steady-state is RUNBOOK §0b. The one-time enablement that got us here was:

1. ~~Merge `mc-persistent-storage`~~ **DONE** — merged into `final_dissertation`; a fresh VM
   bootstraps with the restore fast-path.

2. **Captured the build to B2 (offsite seed)** from the built sweep box before terminating it:
   ```bash
   ssh … Ubuntu@<ip> 'cd /home/dgl_groot && bash experiment/script/cloud/mc/mc_snapshot_env.sh'
   #   -> gsplit-data/groot/env/mc_a100_sm80.tar.zst  (verifies import dgl first)
   ```
   (Do NOT run this mid-sweep: it may `conda install zstd` into the live env.)

3. **Pull results + terminate** to stop billing:
   ```bash
   # pull papers_shard*.csv to WSL + push to B2 (already scripted in the run flow), then:
   MC_API_TOKEN=… bash experiment/script/cloud/mc/mc_terminate.sh ede72ea3-…  # or current uuid
   ```

4. **Seeded the volume** (DONE) — on a 1×A100 launched WITH the volume, pulled the B2 env
   tarball onto `/persistent/groot-sec1/groot/env/` (via `mc_volume.sh seed-from-b2`, which
   encodes the geesefs gotchas above). The volume now holds `mc_a100_sm80.tar.zst`.
   ```bash
   VOLUME_IDS=12cd1bbd-5fc8-4f74-86fe-7b106a5ad02f bash experiment/script/cloud/mc/mc_launch.sh groot-run
   ssh … Ubuntu@<ip> 'bash /home/dgl_groot/experiment/script/cloud/mc/mc_volume.sh seed-from-b2'
   # (data can likewise be seeded with: mc_volume.sh snapshot-data papers100M — currently data stays in B2)
   ```

5. **Validated a cold restart** (DONE) — `mc_bootstrap.sh` detected the tarball, restored,
   verified `import dgl`, and **skipped the ~40-min build** (`[3-5/6] SKIPPED`); the box was
   then terminated and the seeded volume survived (`status=active`).

## Steady-state workflow (this is now live — see RUNBOOK §0b)
Launch with `VOLUME_IDS=12cd1bbd-…` → `mc_bootstrap.sh` restores env from the volume (minutes)
→ `provision_data.sh`/`mc_volume.sh restore-data` for the graph → run → `terminate` when idle.
No rebuild, no idle billing.

## Caveats / rollback
- **Launch-only:** the volume mounts only when passed in `volumeIds` at launch; always
  include `VOLUME_IDS=12cd1bbd-…`.
- **MC→MC only:** tarball is keyed to A100 sm_80 + image 84; a different arch/image must
  rebuild. **If C++/CUDA sources changed** since the snapshot, `./build.sh release` and
  re-`snapshot-env` (a bare `git pull` leaves the `.so` stale).
- **B2 is the offsite backup**; the volume is the fast path. If the volume is empty/missing,
  `mc_bootstrap.sh` falls through to B2, then to a full build — it cannot get stuck.
- **Volume billing:** the volume bills storage-only while it exists (even detached); DELETE
  it via the web console (or `/storage-volumes/{id}` DELETE) when the project is done.
- **Rollback:** `SKIP_RESTORE=1 bash mc_bootstrap.sh` forces a clean build, ignoring any
  snapshot.
