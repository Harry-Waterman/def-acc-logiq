/* eslint-disable @typescript-eslint/no-non-null-assertion */
"use strict";

import "./popup.css";

import {
  MLCEngineInterface,
  InitProgressReport,
  ChatCompletionMessageParam,
} from "@mlc-ai/web-llm";

// modified setLabel to not throw error
function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label != null) {
    label.innerText = text;
  }
}

function getElementAndCheck(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element == null) {
    throw Error("Cannot find element " + id);
  }
  return element;
}

const modelName = getElementAndCheck("model-name");

// SYSTEM_PROMPT adapted for email phishing classification with detailed multi-line guidelines
const SYSTEM_PROMPT = `
SYSTEM:
You are an email-security classifier.
You will receive the following fields of an email:
from_address, subject, recipients, attachment_names, urls, body.

Your job is to output a single JSON object containing:

{
  "score": "0-100",
  "reasons": []
}

Your rules:
	1.	The score must be a number from 0 to 100.
	2.	Only include items in "reasons" if the score is 50 or higher.
	3.	If the score is below 50, "reasons" must be an empty array ([]).
	4.	When reasons are included, they must come only from this list:
	•	"Sender address doesn't match display name”
	•	“Generic Greetings”
	•	“Urgent or threatening language”
	•	“Suspicious urls”
	•	“suspicious attachment names unrelated to the email subject or body”
	•	“spelling and grammar mistakes”
	•	“too good to be true offers”
	•	“requests for personal information”
	5.	Do not invent new reasons.
	6.	Only include reasons that actually appear in the email content.
	7.	Do not output explanations outside the JSON.
	8.	Do not output placeholder text.
	9.	Output only the JSON. No extra text.

Few-shot examples (to anchor behaviour)

Example A (score below 50 → empty reasons array)

Email content summary: harmless internal update
Output format to copy:

{
  "score": "12",
  "reasons": []
}

Example B (score above 50 → reasons list required)

Email content summary: classic phishing asking for bank login
Output format to copy:

{
  "score": "87",
  "reasons": [
    "Suspicious urls",
    "requests for personal information"
  ]
}
Now classify this email:

`;

let context = "";

// throws runtime.lastError if you refresh extension AND try to access a webpage that is already open
fetchPageContents();

function updateDebugStatus(msg: string, data?: string) {
  const statusDiv = document.getElementById("debug-status");
  if (statusDiv) {
    statusDiv.style.display = "block";
    statusDiv.innerText = msg;
    if (data) {
       // Show full data, but in a scrollable way if needed via CSS
       statusDiv.innerText += "\n\nData:\n" + data;
       console.log("Full Context Data:", data);
    }
  }
}

// Modified to trigger classification automatically
function fetchPageContents() {
  updateDebugStatus("Attempting to connect to page...");
  chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
    if (!tabs[0].id) return;
    
    try {
      const port = chrome.tabs.connect(tabs[0].id, { name: "channelName" });
      port.postMessage({});
      
      port.onMessage.addListener(function (msg) {
        console.log("Page contents received:", msg.contents);
        context = msg.contents;
        updateDebugStatus("Success! Metadata extracted.");
        
        // If engine is ready, start classification immediately
        if (engine) {
          classifyEmail();
        } else {
          modelName.innerText = "Content received. Waiting for model...";
        }
      });
    } catch (e) {
       updateDebugStatus("Connection Failed: " + e);
    }
  });
}

let initProgressCallback = (report: InitProgressReport) => {
  // Just update the text
  setLabel("init-label", report.text);
  
  // Check for completion
  if (report.progress == 1.0) {
    setLabel("init-label", "Model Loaded ✅");
  }
};

// initially selected model
const selectedModel = "Llama-3.1-8B-Instruct-q4f16_1-MLC";

let engine: MLCEngineInterface;

// Define a Custom Client
class OffscreenLLMClient {
  modelId: string;
  
  constructor(modelId: string) {
    this.modelId = modelId;
  }

  async init(callback?: (report: any) => void) {
    // Listen for progress updates
    if (callback) {
      chrome.runtime.onMessage.addListener((msg) => {
        if (msg.type === "init-progress") callback(msg.data);
      });
    }
    
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "init-engine", modelId: this.modelId }, resolve);
    });
  }

  chat = {
    completions: {
      create: async (params: any) => {
        return new Promise((resolve, reject) => {
          chrome.runtime.sendMessage({ 
            type: "chat-completion", 
            messages: params.messages, 
            params: { ...params, messages: undefined } // separate messages
          }, (response) => {
            if (response.error) reject(response.error);
            else resolve(response.result);
          });
        });
      }
    }
  }
}

(async () => {
  modelName.innerText = "Loading classifier model...";
  
  // Initialize CUSTOM client
  const client = new OffscreenLLMClient(selectedModel);
  await client.init(initProgressCallback);
  
  // Replace 'engine' usage with 'client'
  // The interface is mocked to match what you used before
  engine = client as any; 
  
  modelName.innerText = "Model loaded. Waiting for email content...";
  
  // Check if we already have context (page might have loaded faster than model)
  if (context) {
    classifyEmail();
  }
})();


// New function to handle classification
async function classifyEmail() {
  if (!context || !engine) return;

  modelName.innerText = "Classifying email...";
  document.getElementById("loading-indicator")!.style.display = "block";
  document.getElementById("resultWrapper")!.style.display = "none";

  // OPTIMIZATION: Parse and truncate context to save tokens
  let optimizedContext = context;
  try {
    const data = JSON.parse(context);
    
    // 1. Limit URLs to top 2
    if (data.urls && Array.isArray(data.urls) && data.urls.length > 2) {
      data.urls = data.urls.slice(0, 2);
      data.urls.push(`...and ${context.length - 2} more`);
    }

    // 2. Limit Body to ~2000 words (approx 8000 chars)
    // This is safe for 4k context (leaves ~2k tokens for system prompt + output)
    if (data.body && typeof data.body === 'string' && data.body.length > 8000) {
      data.body = data.body.substring(0, 8000) + "\n...[TRUNCATED]...";
    }

    optimizedContext = JSON.stringify(data, null, 2);
  } catch (e) {
    console.warn("Failed to optimize context, sending raw:", e);
  }

  const messages: ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Analyze this email:\n${optimizedContext}` }
  ];

  console.log("Messages:", messages);
  
  // Read the disable-thinking checkbox
  const disableThinkingCheckbox = document.getElementById("disable-thinking") as HTMLInputElement;
  const shouldDisableThinking = disableThinkingCheckbox ? disableThinkingCheckbox.checked : false;

  try {
    const completion = await engine.chat.completions.create({
      stream: false, 
      messages: messages,
      temperature: 0.1,
      response_format: { type: "json_object" },
      extra_body: {
        enable_thinking: !shouldDisableThinking,
      },
    });

    let resultText = completion.choices[0].message.content || "{}";
    console.log("Raw LLM Response:", resultText);
    
    // Remove <think> blocks
    resultText = resultText.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    try {
      const result = JSON.parse(resultText);
      displayResult(result, resultText);
    } catch (e) {
      console.error("JSON Parse Error", e);
      displayResult({ score: "Error", reasons: ["Failed to parse LLM response"] }, resultText);
    }

  } catch (err) {
    console.error("Classification Error:", err);
    modelName.innerText = "Error classifying email.";
  } finally {
    document.getElementById("loading-indicator")!.style.display = "none";
    modelName.innerText = "Classification complete.";
  }
}

function displayResult(data: any, rawText: string) {
  const wrapper = document.getElementById("resultWrapper")!;
  const scoreDisplay = document.getElementById("score-display")!;
  const reasonsList = document.getElementById("reasons-list")!;
  const rawJson = document.getElementById("raw-json")!;

  wrapper.style.display = "block";
  
  // Basic styling based on score
  const score = parseInt(data.score) || 0;
  if (score > 70) {
    wrapper.style.backgroundColor = "#ffebee"; // Red-ish
    wrapper.style.border = "1px solid #ef9a9a";
    scoreDisplay.style.color = "#c62828";
    scoreDisplay.innerText = `⚠️ Malicious (Score: ${score})`;
  } else if (score > 30) {
    wrapper.style.backgroundColor = "#fff3e0"; // Orange-ish
    wrapper.style.border = "1px solid #ffe0b2";
    scoreDisplay.style.color = "#ef6c00";
    scoreDisplay.innerText = `⚠️ Suspicious (Score: ${score})`;
  } else {
    wrapper.style.backgroundColor = "#e8f5e9"; // Green-ish
    wrapper.style.border = "1px solid #c8e6c9";
    scoreDisplay.style.color = "#2e7d32";
    scoreDisplay.innerText = `✅ Benign (Score: ${score})`;
  }

  // Render reasons
  reasonsList.innerHTML = "";
  if (data.reasons && Array.isArray(data.reasons)) {
    data.reasons.forEach((r: string) => {
      const li = document.createElement("li");
      li.innerText = r;
      reasonsList.appendChild(li);
    });
  }
}
