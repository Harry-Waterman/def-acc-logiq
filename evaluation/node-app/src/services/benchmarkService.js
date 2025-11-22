import config from "../config/index.js";
import logger from "../utils/logger.js";
import { sampleDataset } from "./datasetService.js";

const labelFromScore = (score, threshold = config.scoreThreshold) =>
  score >= threshold ? "malicious" : "not_malicious";

const average = (values) => {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

const stdDeviation = (values) => {
  if (values.length < 2) return 0;
  const mean = average(values);
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(variance);
};

const consistencyRating = (ratio) => {
  if (ratio >= 0.95) return "HIGHLY_REPEATABLE";
  if (ratio >= 0.8) return "REPEATABLE";
  if (ratio >= 0.6) return "MODERATELY_REPEATABLE";
  return "NOT_REPEATABLE";
};

const analyzeRepeatability = (runs) => {
  const labelCounts = runs.reduce((acc, run) => {
    acc.set(run.predictedLabel, (acc.get(run.predictedLabel) ?? 0) + 1);
    return acc;
  }, new Map());

  let dominantLabel = null;
  let dominantCount = 0;
  labelCounts.forEach((count, label) => {
    if (count > dominantCount) {
      dominantLabel = label;
      dominantCount = count;
    }
  });

  const totalRuns = runs.length || 1;
  const consistency = dominantCount / totalRuns;

  return {
    dominantLabel,
    consistency,
    rating: consistencyRating(consistency),
    labelCounts: Object.fromEntries(labelCounts),
  };
};

export const runBenchmark = async ({
  runner,
  datasetPath,
  sampleSize = config.sampleSize,
  numRuns = config.numRuns,
  seed = Date.now(),
  includeRuns = false,
} = {}) => {
  if (!runner) {
    throw new Error("Model runner instance is required");
  }

  const sample = sampleDataset({ sampleSize, datasetPath, seed });
  logger.info(
    { sampleSize: sample.length, numRuns, datasetPath: datasetPath ?? config.datasetPath },
    "Starting benchmarking run",
  );

  const summary = {
    emailsEvaluated: sample.length,
    totalRuns: sample.length * numRuns,
    correctPredictions: 0,
    incorrectPredictions: 0,
    overallAccuracy: 0,
    scoreDistribution: [],
    datasetPath: datasetPath ?? config.datasetPath,
    jsonFailures: 0,
    invalidResponseSamples: [],
  };

  const perEmail = [];
  const invalidResponseLimit = 10;

  // Execute sequentially to avoid overloading local model
  for (const email of sample) {
    const runs = [];
    let emailJsonFailures = 0;

    for (let runIndex = 0; runIndex < numRuns; runIndex += 1) {
      const output = await runner.classify(email, { runIndex });
      const numericScore = Number(output.score ?? 0);
      const score = Number.isFinite(numericScore) ? numericScore : 0;
      const predictedLabel = labelFromScore(score);
      const isCorrect = predictedLabel === email.label;
      summary.correctPredictions += isCorrect ? 1 : 0;
      summary.incorrectPredictions += isCorrect ? 0 : 1;
      summary.scoreDistribution.push(score);
      const runValidJson = output.validJson !== false;
      if (!runValidJson) {
        summary.jsonFailures += 1;
        emailJsonFailures += 1;
        if (summary.invalidResponseSamples.length < invalidResponseLimit) {
          summary.invalidResponseSamples.push({
            emailId: email.id,
            runIndex,
            rawResponse: output.rawResponse,
            parseError: output.parseError,
          });
        }
      }

      if (includeRuns) {
        runs.push({
          runIndex,
          score,
          predictedLabel,
          isCorrect,
          reasons: output.reasons ?? [],
          validJson: runValidJson,
          parseError: output.parseError,
        });
      } else {
        runs.push({
          runIndex,
          score,
          predictedLabel,
          isCorrect,
          validJson: runValidJson,
        });
      }
    }

    const correctRuns = runs.filter((run) => run.isCorrect).length;
    const scoreValues = runs.map((run) => run.score);
    const repeatability = analyzeRepeatability(runs);

    const emailStats = {
      id: email.id,
      label: email.label,
      contextPreview: email.context.slice(0, 140),
      correctRuns,
      accuracy: correctRuns / numRuns,
      averageScore: average(scoreValues),
      scoreStdDev: stdDeviation(scoreValues),
      repeatability,
      jsonFailures: emailJsonFailures,
    };

    if (includeRuns) {
      emailStats.runs = runs;
    }

    perEmail.push(emailStats);
  }

  summary.overallAccuracy =
    summary.correctPredictions / (summary.totalRuns || 1);
  summary.averageScore = average(summary.scoreDistribution);
  summary.scoreStdDev = stdDeviation(summary.scoreDistribution);

  return {
    summary,
    perEmail,
  };
};

export default {
  runBenchmark,
};


