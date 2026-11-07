# 3pViNE

Code for *Privacy Preserving Distributed Virtual Network Embedding* (CSET 2026).
Note that 3pViNE is called ZK-VNE and a testbed is called a facility in the code. 

## Setup

Requires Linux, Docker with Compose v2, and Python 3.11+.

```bash
git clone https://github.com/yoontaeung/3pViNE.git
cd 3pViNE
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
docker build -t zk-vne-zk:local .
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s benchmarks/tests
```

## Demo

```bash
docker compose up -d --build --wait
.venv/bin/python scripts/system_smoke.py --fixture-index 0
docker compose down -v
```

## Reproduce

Figure 4 (proof scalability) runs in the background:

```bash
SCALING_RUN_ID=fig4 SCALING_REPETITIONS=3 ZKP_SKIP_BUILD=1 \
SCALING_PROFILES="vn3_sn10_h3 vn3_sn20_h3 vn3_sn30_h3 vn3_sn40_h3 vn3_sn50_h3 vn1_sn30_h3 vn5_sn30_h3 vn10_sn30_h3 vn15_sn30_h3 vn20_sn30_h3" \
SCALING_PREFLIGHT_PROFILES="vn3_sn50_h3 vn20_sn30_h3" \
  bash benchmarks/scripts/start_scaling_experiment.sh
docker logs -f zk-vne-fig4
```

Tables 2 and 3 (100 repetitions each):

```bash
bash benchmarks/scripts/run_tables_100.sh artifacts/benchmarks/tables_100
```

Results are written under `artifacts/benchmarks/`. The PolyViNE message
traces for Table 1 are in `benchmarks/polyvine_trace/`.
