import config from "../config/index.js";
import { SYSTEM_PROMPT } from "../config/prompt.js";
import logger from "../utils/logger.js";

const CLEAN_THINKING_REGEX = /<think>[\s\S]*?<\/think>/g;

const serializeError = (error) => {
  if (error instanceof Error) {
    return { message: error.message, stack: error.stack };
  }
  return error;
};

export class MLCServerRunner {
  constructor(options = {}) {
    this.modelId = options.modelId ?? config.modelId;
    this.baseUrl = options.baseUrl ?? config.mlcServer.baseUrl;
    this.completionsPath =
      options.completionsPath ?? config.mlcServer.completionsPath;
    this.apiKey = options.apiKey ?? config.mlcServer.apiKey;
    this.timeoutMs =
      options.requestTimeoutMs ?? config.mlcServer.requestTimeoutMs;
  }

  async init() {
    // Optionally, we could add a health check here in the future.
    return Promise.resolve();
  }

  async classify(emailRecord) {
    const controller = new AbortController();
    const timeoutMs = this.timeoutMs || 120000;
    const timeoutHandle = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: `Analyze this email:\n${emailRecord.context}`,
        },
      ];

      const body = {
        model: this.modelId,
        messages,
        temperature: 0.1,
        response_format: { type: "json_object" },
      };

      const headers = {
        "Content-Type": "application/json",
      };
      if (this.apiKey) {
        headers.Authorization = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.baseUrl}${this.completionsPath}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response || !response.ok) {
        const errorText = await response?.text?.().catch(() => "");
        throw new Error(
          `MLC server responded with ${response?.status}: ${errorText}`,
        );
      }

      const completion = await response.json();
      let raw = completion?.choices?.[0]?.message?.content ?? "{}";
      raw = raw.replace(CLEAN_THINKING_REGEX, "").trim();

      let parsed;
      let validJson = true;
      let parseErrorMessage;
      try {
        parsed = JSON.parse(raw);
      } catch (error) {
        validJson = false;
        parseErrorMessage =
          error instanceof Error ? error.message : String(error);
        logger.error(
          { error: parseErrorMessage, raw },
          "Failed to parse MLC server response",
        );
        parsed = {
          score: "0",
          reasons: ["Failed to parse model response"],
        };
      }

      const normalizedScore = parsed.score ?? parsed.Score ?? 0;
      const normalizedReasons = Array.isArray(parsed.reasons)
        ? parsed.reasons
        : [];

      return {
        score: String(normalizedScore),
        reasons: normalizedReasons,
        rawResponse: raw,
        validJson,
        parseError: parseErrorMessage,
        runner: "mlc-server",
      };
    } catch (error) {
      logger.error(
        { error: serializeError(error) },
        "MLC server classification failed",
      );
      throw error;
    } finally {
      clearTimeout(timeoutHandle);
    }
  }

  async dispose() {
    return Promise.resolve();
  }
}

export default MLCServerRunner;

