import fs from "node:fs";
import path from "node:path";
import { parse } from "csv-parse/sync";
import config from "../config/index.js";
import logger from "../utils/logger.js";
import { createPseudoRandom } from "../utils/random.js";

const LABEL_MAP = new Map([
  ["0", "not_malicious"],
  [0, "not_malicious"],
  ["1", "malicious"],
  [1, "malicious"],
  ["phishing", "malicious"],
  ["malicious", "malicious"],
  ["legitimate", "not_malicious"],
  ["benign", "not_malicious"],
  ["ham", "not_malicious"],
  ["spam", "malicious"],
  ["not_malicious", "not_malicious"],
]);

const datasetCache = new Map();

const normalizeLabel = (value) => {
  if (value === undefined || value === null) return undefined;
  const trimmedValue = String(value).trim().toLowerCase();
  return LABEL_MAP.get(trimmedValue) ?? trimmedValue;
};

const formatFieldValue = (value) => {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value)) return value.join(", ");
  return String(value).trim();
};

const buildEmailContext = (record) => {
  const segments = config.emailFields.map((field) => {
    const presentedValue = formatFieldValue(record[field]);
    return `${field}: ${presentedValue}`;
  });

  return segments.join("\n");
};

const buildModelInput = (record) => {
  const textSegments = config.textFields
    .map((field) => formatFieldValue(record[field]))
    .filter(Boolean);

  return textSegments.join("\n\n");
};

export const loadDataset = (datasetPath = config.datasetPath) => {
  const resolvedPath = path.resolve(datasetPath);
  if (datasetCache.has(resolvedPath)) {
    return datasetCache.get(resolvedPath);
  }

  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`Dataset file not found at ${resolvedPath}`);
  }

  const rawContent = fs.readFileSync(resolvedPath, "utf-8");
  const rows = parse(rawContent, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });

  const records = rows
    .map((row, idx) => {
      const labelValue = normalizeLabel(row[config.labelColumn]);
      if (!labelValue) {
        logger.warn(
          { idx, row },
          "Skipping row without recognizable label value",
        );
        return null;
      }

      return {
        id: idx,
        original: row,
        label: labelValue,
        context: buildEmailContext(row),
        modelInput: buildModelInput(row),
      };
    })
    .filter(Boolean);

  datasetCache.set(resolvedPath, records);
  logger.info(
    { datasetPath: resolvedPath, samplesLoaded: records.length },
    "Dataset loaded",
  );
  return records;
};

export const sampleDataset = ({
  sampleSize = config.sampleSize,
  seed,
  datasetPath,
} = {}) => {
  const records = loadDataset(datasetPath);
  if (!records.length) {
    throw new Error("Dataset is empty.");
  }

  if (!sampleSize || sampleSize >= records.length) {
    return records;
  }

  const rng = createPseudoRandom(seed);
  const selected = [];
  const seen = new Set();

  while (selected.length < sampleSize) {
    const index = Math.floor(rng() * records.length);
    if (seen.has(index)) continue;
    seen.add(index);
    selected.push(records[index]);
  }

  return selected;
};

export default {
  loadDataset,
  sampleDataset,
};


