import fs from "node:fs";
import path from "node:path";
import { paths } from "./index.js";

const defaultPromptPath = path.resolve(
  paths.repoRoot,
  "prompts",
  "system_prompt_v1.md",
);

const promptPath = process.env.SYSTEM_PROMPT_PATH ?? defaultPromptPath;

let SYSTEM_PROMPT = `SYSTEM:
You are an email security classifier.
Return a JSON payload with "score" and "reasons" keys.`;

try {
  if (fs.existsSync(promptPath)) {
    SYSTEM_PROMPT = fs.readFileSync(promptPath, "utf-8");
  }
} catch (error) {
  // eslint-disable-next-line no-console
  console.warn("Failed to load system prompt file:", error);
}

export { SYSTEM_PROMPT, promptPath };


