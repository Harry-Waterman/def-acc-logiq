#!/usr/bin/env node
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import config from "../config/index.js";
import logger from "../utils/logger.js";
import MLCServerRunner from "../model/mlcServerRunner.js";
import { runBenchmark } from "../services/benchmarkService.js";

const argv = yargs(hideBin(process.argv))
  .option("sampleSize", {
    type: "number",
    description: "Number of emails to evaluate",
    default: config.sampleSize,
  })
  .option("numRuns", {
    type: "number",
    description: "Number of runs per email",
    default: config.numRuns,
  })
  .option("datasetPath", {
    type: "string",
    description: "Path to dataset CSV file",
  })
  .option("seed", {
    type: "number",
    description: "Seed for deterministic sampling",
  })
  .option("includeRuns", {
    type: "boolean",
    description: "Include per run details in the output",
    default: false,
  })
  .help()
  .parseSync();

const printReport = (result) => {
  const { summary, perEmail } = result;
  const header = [
    "MODEL BENCHMARK REPORT",
    `Dataset: ${summary.datasetPath ?? config.datasetPath}`,
    `Emails Evaluated: ${summary.emailsEvaluated}`,
    `Runs per Email: ${argv.numRuns}`,
    `Overall Accuracy: ${(summary.overallAccuracy * 100).toFixed(2)}%`,
    `Average Score: ${summary.averageScore.toFixed(2)} (σ=${summary.scoreStdDev.toFixed(2)})`,
  ];

  console.log(header.join("\n"));
  console.log("\nTop-level stats:");
  console.table([
    {
      metric: "Correct Predictions",
      value: summary.correctPredictions,
    },
    {
      metric: "Incorrect Predictions",
      value: summary.incorrectPredictions,
    },
    {
      metric: "JSON Parse Failures",
      value: summary.jsonFailures,
    },
  ]);

  if (summary.jsonFailures > 0 && summary.invalidResponseSamples.length) {
    console.log("\nSample invalid responses:");
    console.table(
      summary.invalidResponseSamples.map((sample) => ({
        emailId: sample.emailId,
        runIndex: sample.runIndex,
        parseError: sample.parseError ?? "Unknown error",
      })),
    );
  }

  console.log("\nPer-email snapshot:");
  console.table(
    perEmail.slice(0, 10).map((email) => ({
      id: email.id,
      label: email.label,
      accuracy: `${(email.accuracy * 100).toFixed(1)}%`,
      avgScore: email.averageScore.toFixed(1),
      stdDev: email.scoreStdDev.toFixed(1),
      repeatability: email.repeatability.rating,
      jsonErrors: email.jsonFailures,
    })),
  );

  if (perEmail.length > 10) {
    console.log(`... ${perEmail.length - 10} more emails omitted for brevity.`);
  }
};

const main = async () => {
  const runner = new MLCServerRunner();
  try {
    await runner.init();
    const result = await runBenchmark({
      runner,
      datasetPath: argv.datasetPath,
      sampleSize: argv.sampleSize,
      numRuns: argv.numRuns,
      seed: argv.seed,
      includeRuns: argv.includeRuns,
    });
    printReport(result);
  } finally {
    await runner.dispose();
  }
};

main().catch((error) => {
  logger.error({ error }, "Benchmark CLI failed");
  process.exit(1);
});


