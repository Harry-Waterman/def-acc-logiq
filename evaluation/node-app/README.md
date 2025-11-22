## Evaluation Node App

Dataset-driven benchmarking service that runs the phishing classifier against
CSV emails rather than the live Outlook integration. It exposes both an
Express server and a CLI workflow so you can automate large repeatability
runs (100+ iterations on the same email).

### Requirements
- Node.js 20+ (if running locally outside Docker)
- Docker / Docker Compose (for the one-command workflow)
- Dataset located in `dataset/*.csv` (default: `dataset/Nigerian_Fraud.csv`)

### Installation
```bash
cd evaluation/node-app
npm install
cp env.example .env    # optional overrides
```

### Running the real model
#### Option A – Docker (recommended for reproducibility)
```bash
cd evaluation/node-app
docker compose up --build
```
This spins up:
- `mlc-server`: pulls the official `mlc-llm` image and serves the Qwen model on port `8080`
- `evaluator`: mounts the repo into the container and runs `npm run benchmark`

Environment variables in `.env` (or provided on the command line) control dataset path,
sample size, number of runs, etc. By default the dataset volume is mounted from the host
(`../..:/workspace`), so make sure the CSVs exist under `dataset/` on your machine.

To keep the mlc server running but execute the evaluator manually, launch the compose stack
and then use `docker compose run evaluator npm run benchmark -- --sampleSize ...`.

#### Option B – Local processes
1. Launch an `mlc-llm` server manually (requires the [mlc-llm CLI](https://github.com/mlc-ai/mlc-llm)):
   ```bash
   mlc_llm serve \
     --model Qwen3-0.6B-q4f16_1-MLC \
     --host 127.0.0.1 \
     --port 8080
   ```
2. Point the evaluator at that endpoint:
   ```powershell
   $env:MLC_SERVER_URL="http://127.0.0.1:8080"
   npm run benchmark -- --sampleSize 5 --numRuns 20
   Remove-Item Env:MLC_SERVER_URL  # optional cleanup
   ```
   You can also set `MLC_SERVER_URL` permanently inside `.env`.

### Available Scripts
| command | purpose |
| --- | --- |
| `npm run dev` | start the Express API with hot reload |
| `npm start` | start the API without watchers |
| `npm run benchmark -- --sampleSize 25 --numRuns 100` | execute benchmarking via CLI |

### API Endpoints
- `GET /health` – simple health probe
- `GET /api/benchmark/config` – returns current dataset + threshold config
- `POST /api/benchmark/run` – body `{ sampleSize?, numRuns?, datasetPath?, seed?, includeRuns? }`

### CLI Output
```bash
npm run benchmark -- --sampleSize 10 --numRuns 50 --includeRuns false
```
The CLI prints an aggregated report plus the first 10 per-email stats. Pass
`--includeRuns true` to show every run (can be large).

### Environment Variables
| Key | Description | Default |
| --- | --- | --- |
| `DATASET_PATH` | Absolute/relative path to CSV file | `../../dataset/Nigerian_Fraud.csv` |
| `EVAL_SERVER_PORT` | Express port | `4100` |
| `SCORE_THRESHOLD` | Score ≥ threshold ⇒ malicious | `50` |
| `DEFAULT_SAMPLE_SIZE` | Emails sampled when not specified | `20` |
| `NUM_RUNS` | Runs per email | `100` |
| `TEXT_FIELDS` | Comma separated list used to build model input | `subject,body` |
| `EMAIL_FIELDS` | Fields included in the context block | `sender,receiver,date,subject,body,urls` |
| `MLC_MODEL_ID` | Model id sent to the mlc-llm server | `Qwen3-0.6B-q4f16_1-MLC` |
| `SYSTEM_PROMPT_PATH` | Optional override for the system prompt file | `../../prompts/system_prompt_v1.md` |
| `MLC_ENABLE_THINKING` | Toggle the `enable_thinking` flag for WebLLM | `true` |
| `MLC_SERVER_URL` | Base URL of your mlc-llm deployment | `http://127.0.0.1:8080` |
| `MLC_SERVER_COMPLETIONS_PATH` | Path to the chat completions endpoint | `/v1/chat/completions` |
| `MLC_SERVER_API_KEY` | Optional bearer token if your server requires auth | _(empty)_ |
| `MLC_SERVER_TIMEOUT_MS` | Request timeout for inference calls | `120000` |

### Model Runners
- **mlc-llm server runner** – the evaluator now exclusively calls the mlc-llm REST API (OpenAI-compatible). Ensure the server is running before invoking CLI or API commands.

### Extending
- Drop additional datasets into `dataset/` and point `DATASET_PATH` at them
- Override `TEXT_FIELDS` / `EMAIL_FIELDS` (comma-separated values) in `.env`
- Wrap the service in Docker or CI jobs to track regression accuracy daily


