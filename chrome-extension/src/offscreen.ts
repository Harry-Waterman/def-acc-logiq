import { CreateMLCEngine, MLCEngineInterface } from "@mlc-ai/web-llm";

// Global engine instance
let engine: MLCEngineInterface | null = null;

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "init-engine") {
    initEngine(message.modelId, sendResponse);
    return true; // Keep channel open for async response
  }
  
  if (message.type === "chat-completion") {
    runCompletion(message.messages, message.params, sendResponse);
    return true;
  }
});

async function initEngine(modelId: string, sendResponse: (res: any) => void) {
  if (engine) {
    sendResponse({ status: "already_initialized" });
    return;
  }

  try {
    // Use CreateMLCEngine instead of MLCEngine.Create
    engine = await CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        // Optional: Send progress back to background -> popup
        chrome.runtime.sendMessage({ 
            type: "init-progress", 
            data: report 
        });
      }
    });
    sendResponse({ status: "success" });
  } catch (err) {
    sendResponse({ status: "error", error: String(err) });
  }
}

async function runCompletion(messages: any[], params: any, sendResponse: (res: any) => void) {
  if (!engine) {
    sendResponse({ error: "Engine not initialized" });
    return;
  }

  try {
    const completion = await engine.chat.completions.create({
      messages,
      ...params
    });
    sendResponse({ result: completion });
  } catch (err) {
    sendResponse({ error: String(err) });
  }
}
