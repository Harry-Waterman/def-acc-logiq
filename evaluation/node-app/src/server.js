import express from "express";
import config from "./config/index.js";
import logger from "./utils/logger.js";
import benchmarkRouter from "./routes/benchmark.js";
import MLCServerRunner from "./model/mlcServerRunner.js";

const app = express();

app.use(express.json({ limit: "1mb" }));

app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "evaluation-node-app" });
});

app.use("/api/benchmark", benchmarkRouter);

app.use((err, req, res, next) => {
  logger.error({ err }, "Request failed");
  res.status(err.status || 500).json({
    error: err.message ?? "Unexpected server error",
  });
});

const registerShutdownHooks = (runner) => {
  const closeRunner = async () => {
    if (runner && typeof runner.dispose === "function") {
      try {
        await runner.dispose();
      } catch (error) {
        logger.error({ error }, "Failed to dispose runner cleanly");
      }
    }
  };

  ["SIGINT", "SIGTERM"].forEach((signal) => {
    process.once(signal, async () => {
      await closeRunner();
      process.exit(0);
    });
  });

  process.once("beforeExit", closeRunner);
};

const startServer = async () => {
  const runner = new MLCServerRunner();
  await runner.init();
  app.locals.runner = runner;
  registerShutdownHooks(runner);
  app.listen(config.port, () => {
    logger.info(
      {
        port: config.port,
        env: config.env,
        datasetPath: config.datasetPath,
      },
      "Evaluation server running",
    );
  });
};

startServer().catch((error) => {
  logger.error({ error }, "Unable to start evaluation server");
  process.exit(1);
});


