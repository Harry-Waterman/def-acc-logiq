import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const nodeAppRoot = path.resolve(__dirname, "../..");
const evaluationRoot = path.resolve(nodeAppRoot, "..");
const repoRoot = path.resolve(evaluationRoot, "..");

const envCandidates = [
  path.resolve(nodeAppRoot, ".env"),
  path.resolve(repoRoot, ".env"),
];

envCandidates.forEach((envPath) => {
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath, override: false });
  }
});

const parseList = (value, fallback) => {
  if (!value) return fallback;
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
};

const toNumber = (value, fallback) => {
  const nextValue = Number(value);
  return Number.isFinite(nextValue) ? nextValue : fallback;
};

export const paths = {
  repoRoot,
  evaluationRoot,
  nodeAppRoot,
  datasetDefault: path.resolve(repoRoot, "dataset", "Nigerian_Fraud.csv"),
};

export const config = {
  env: process.env.NODE_ENV ?? "development",
  port: toNumber(process.env.EVAL_SERVER_PORT, 4100),
  datasetPath: process.env.DATASET_PATH ?? paths.datasetDefault,
  labelColumn: process.env.LABEL_COLUMN ?? "label",
  textFields: parseList(process.env.TEXT_FIELDS, ["subject", "body"]),
  emailFields: parseList(process.env.EMAIL_FIELDS, [
    "sender",
    "receiver",
    "date",
    "subject",
    "body",
    "urls",
  ]),
  scoreThreshold: toNumber(process.env.SCORE_THRESHOLD, 50),
  sampleSize: toNumber(process.env.DEFAULT_SAMPLE_SIZE, 20),
  numRuns: toNumber(process.env.NUM_RUNS, 100),
  modelId: process.env.MLC_MODEL_ID ?? "Qwen3-0.6B-q4f16_1-MLC",
  thinkingEnabled: (process.env.MLC_ENABLE_THINKING ?? "true") === "true",
  mlcServer: {
    baseUrl: process.env.MLC_SERVER_URL ?? "http://127.0.0.1:8080",
    completionsPath:
      process.env.MLC_SERVER_COMPLETIONS_PATH ?? "/v1/chat/completions",
    apiKey: process.env.MLC_SERVER_API_KEY ?? "",
    requestTimeoutMs: toNumber(process.env.MLC_SERVER_TIMEOUT_MS, 120000),
  },
};

export default config;

