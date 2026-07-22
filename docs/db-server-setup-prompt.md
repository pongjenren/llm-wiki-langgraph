# Prompt — 建立 llm-wiki 的 PostgreSQL + pgvector 伺服器（uv，不用 Docker）

這份檔案是要**交給另一個 Claude Code**（在同一台電腦的另一個資料夾）執行的 prompt。
把底下 `---` 之間的整段內容貼給那個 Claude Code 即可。它會在該資料夾裡建立並啟動
一個獨立、不需要 root/Docker 的本機 Postgres 伺服器，供 `llm-wiki` 應用程式連線。

> 背景：`llm-wiki` 這個 app 本身**只連線、不架 server**。它會用連線字串連到這個
> 伺服器，並自己執行 schema/extension 的 bootstrap（`CREATE EXTENSION vector`、建表、
> 建 HNSW 索引）。所以伺服器端只需要提供：一個跑著的 Postgres、正確的 role 與
> database、以及可安裝的 pgvector，讓 `CREATE EXTENSION vector` 能成功。
>
> **這個環境沒有 Docker，也不能用 `sudo apt install`**（sudo 要密碼）。唯一能用的是
> `uv`（或 `pip`）安裝 PyPI 套件。因此改用 [`pgserver`](https://pypi.org/project/pgserver/)
> 這個 pip 套件——它把一份**自帶的 PostgreSQL 16 執行檔和 pgvector 擴充**打包在 wheel
> 裡，`pip install` 就會下載到本機，完全不需要系統層級安裝或 root。我們用它附帶的
> `initdb` / `pg_ctl` 以一般使用者身分在本機跑一個 Postgres。

---

## Task

Stand up a local PostgreSQL server (with the **pgvector** extension available) in
the current directory, for a separate application called `llm-wiki` to connect
to. The application connects over TCP and bootstraps its own tables — you only
need to provide a running server, the role/database, and a creatable `vector`
extension.

Do this **with `uv` only — no Docker, no `sudo`, no system-wide install.** All
binaries come from the `pgserver` PyPI package (a self-contained PostgreSQL 16 +
pgvector build), so the only external access you need is a Python package index
(pip/uv). Everything runs as the current unprivileged user.

## Hard contract — the app connects with exactly this DSN

```
postgresql://llm_wiki:llm_wiki@localhost:5432/llm_wiki
```

So the server **must** provide, reachable from the host over TCP:

- host `localhost`, port **5432** (fail early if the port is already in use)
- role `llm_wiki` with password `llm_wiki`, able to log in over TCP with a password
- database `llm_wiki` owned by that role
- the `vector` extension (pgvector) installed and creatable — i.e. the
  `llm_wiki` role must be able to run `CREATE EXTENSION IF NOT EXISTS vector;`
  (we make `llm_wiki` the bootstrap superuser via `initdb -U llm_wiki`, so it can)

If you change the password/user/db/port, you must tell me, because the app side
then needs its `LLM_WIKI_DB_URL` env var updated to match.

## Environment

- Same machine as the app: WSL2, Ubuntu 24.04.
- `uv` is installed and on `PATH`. **No Docker. `sudo` needs a password** — do not
  use it. You can reach a PyPI index to `pip`/`uv install`.
- The `pgserver` wheel ships prebuilt Postgres + pgvector binaries, so no
  compiler or system packages are required.

## Approach — `pgserver` binaries driven by `initdb` / `pg_ctl`

`pgserver` runs its own managed server over a **unix socket** by default, which
does **not** satisfy the TCP `localhost:5432` contract above. So we don't use its
managed runner — we install the package only for its bundled binaries, then run
`initdb` + `pg_ctl` ourselves with an explicit TCP + password-auth config.

Create these three files in the current directory.

### `up.sh` — install, init, and start (idempotent)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT=5432
PGDATA="$PWD/pgdata"
LOGFILE="$PWD/postgres.log"

# 1. Get the vendored Postgres + pgvector binaries via uv (no Docker, no sudo).
[ -d .venv ] || uv venv
uv pip install --quiet pgserver
BIN="$(uv run python -c 'from pgserver._commands import POSTGRES_BIN_PATH; print(POSTGRES_BIN_PATH)')"
export PATH="$BIN:$PATH"

# 2. Fail early if port 5432 is already taken.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "Port $PORT is already in use — stop whatever is on it first." >&2
  exit 1
fi

# 3. Initialise the data dir once: llm_wiki is the bootstrap superuser,
#    password auth for both local and host (TCP) connections.
if [ ! -f "$PGDATA/PG_VERSION" ]; then
  printf 'llm_wiki\n' > .pwfile
  initdb -D "$PGDATA" -U llm_wiki -E UTF8 \
    --auth-local=scram-sha-256 --auth-host=scram-sha-256 --pwfile=.pwfile
  rm -f .pwfile
  # Listen on TCP localhost:5432. unix_socket_directories='/tmp' keeps the
  # socket path short (Postgres caps it at 107 bytes; deep project paths overflow).
  cat >> "$PGDATA/postgresql.conf" <<CONF

# --- llm-wiki local server ---
listen_addresses = 'localhost'
port = $PORT
unix_socket_directories = '/tmp'
CONF
fi

# 4. Start the server (safe to re-run; does nothing if already up).
pg_ctl -D "$PGDATA" -l "$LOGFILE" -w start || pg_ctl -D "$PGDATA" status

# 5. Ensure the llm_wiki database exists, owned by the llm_wiki role.
export PGPASSWORD=llm_wiki
if ! psql "postgresql://llm_wiki:llm_wiki@localhost:$PORT/postgres" -tAc \
      "SELECT 1 FROM pg_database WHERE datname='llm_wiki'" | grep -q 1; then
  createdb -h localhost -p "$PORT" -U llm_wiki -O llm_wiki llm_wiki
fi

# 6. Confirm pgvector is creatable (the app also does this, but verify now).
psql "postgresql://llm_wiki:llm_wiki@localhost:$PORT/llm_wiki" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Ready → postgresql://llm_wiki:llm_wiki@localhost:$PORT/llm_wiki"
```

### `down.sh` — stop the server (keeps data)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BIN="$(uv run python -c 'from pgserver._commands import POSTGRES_BIN_PATH; print(POSTGRES_BIN_PATH)')"
"$BIN/pg_ctl" -D "$PWD/pgdata" -w stop
```

### `README.md`

Document the connection string and the common commands:

- `./up.sh` — install/init if needed and start the server (idempotent; re-run
  after a reboot to bring it back up).
- `./down.sh` — stop the server, keeping all data in `./pgdata`.
- Wipe and start over: `./down.sh` (if running) then `rm -rf pgdata`, then `./up.sh`.
- Connection string: `postgresql://llm_wiki:llm_wiki@localhost:5432/llm_wiki`.
- Note: there is no auto-restart (no Docker/systemd). After a machine reboot the
  server is down until someone runs `./up.sh` again.

## Steps

1. Write `up.sh`, `down.sh`, and `README.md` as above; `chmod +x up.sh down.sh`.
2. Run `./up.sh`. It installs `pgserver` via `uv`, checks port 5432 is free,
   inits `./pgdata`, starts Postgres, and creates the role/db + `vector`.
3. Run the verification below and paste the output.

## Verification — all of these must pass

```bash
# 1. Server is accepting connections on the contract host/port/db
BIN="$(uv run python -c 'from pgserver._commands import POSTGRES_BIN_PATH; print(POSTGRES_BIN_PATH)')"
"$BIN/pg_isready" -h localhost -p 5432 -U llm_wiki -d llm_wiki

# 2. pgvector is creatable and works, over the exact contract DSN
PGPASSWORD=llm_wiki "$BIN/psql" "postgresql://llm_wiki:llm_wiki@localhost:5432/llm_wiki" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;" \
  -c "SELECT extversion FROM pg_extension WHERE extname='vector';" \
  -c "SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;"
```

Expect: `pg_isready` reports `accepting connections`, a pgvector extension
version is printed, and the `<=>` query returns a numeric cosine distance. If the
`<=>` query returns a number, the app will work against this server.

## Constraints / notes

- Do **not** use Docker and do **not** `sudo apt install postgresql` — use only
  `uv`/`pip` (the `pgserver` wheel) so it runs with just package-index access.
- Do not delete or modify anything outside the current directory. All state lives
  in `./pgdata`; `./postgres.log` has the server log if startup fails.
- If startup fails with "Unix-domain socket path … is too long", the run
  directory is too deep — the config already redirects the socket to `/tmp` to
  avoid this; keep that line.
- There is no `restart: unless-stopped` equivalent. `./down.sh` stops the server
  but keeps the volume; `rm -rf pgdata` is the "start over" that wipes all data.
