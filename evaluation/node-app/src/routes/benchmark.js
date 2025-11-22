import { Router } from "express";
import {
  getBenchmarkConfig,
  triggerBenchmark,
} from "../controllers/benchmarkController.js";

const router = Router();

router.get("/config", getBenchmarkConfig);
router.post("/run", triggerBenchmark);

export default router;

