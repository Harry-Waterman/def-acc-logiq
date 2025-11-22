import { z } from "zod";
import config from "../config/index.js";
import { runBenchmark } from "../services/benchmarkService.js";
import logger from "../utils/logger.js";

const runSchema = z.object({
  sampleSize: z.number().int().positive().max(500).optional(),
  numRuns: z.number().int().positive().max(1000).optional(),
  datasetPath: z.string().min(1).optional(),
  seed: z.number().int().optional(),
  includeRuns: z.boolean().optional(),
});

export const getBenchmarkConfig = (req, res) => {
  res.json({
    datasetPath: config.datasetPath,
    labelColumn: config.labelColumn,
    textFields: config.textFields,
    emailFields: config.emailFields,
    scoreThreshold: config.scoreThreshold,
    defaultSampleSize: config.sampleSize,
    defaultNumRuns: config.numRuns,
    modelId: config.modelId,
    env: config.env,
  });
};

export const triggerBenchmark = async (req, res, next) => {
  try {
    const parameters = runSchema.parse(req.body ?? {});
    const runner = req.app.locals.runner;
    if (!runner) {
      throw new Error("Model runner not initialized");
    }
    const result = await runBenchmark({ ...parameters, runner });
    res.json(result);
  } catch (error) {
    logger.error({ error }, "Failed to run benchmark");
    next(error);
  }
};

export default {
  getBenchmarkConfig,
  triggerBenchmark,
};

