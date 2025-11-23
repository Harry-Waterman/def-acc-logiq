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

// Removed modelName element retrieval

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

// Modified to trigger classification automatically
function fetchPageContents() {
  chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
    if (!tabs[0].id) return;
    
    try {
      const port = chrome.tabs.connect(tabs[0].id, { name: "channelName" });
      port.postMessage({});
      
      port.onMessage.addListener(function (msg) {
        console.log("Page contents received:", msg.contents);
        context = msg.contents;
        
        // If engine is ready, start classification immediately
        if (engine) {
          classifyEmail();
        } else {
          // REMOVED text update: modelName.innerText = "Content received. Waiting for model...";
          // Show loading dots instead if not already showing
          document.getElementById("loading-indicator")!.style.display = "block";
        }
      });
    } catch (e) {
       console.error("Connection Failed: " + e);
    }
  });
}

let initProgressCallback = (report: InitProgressReport) => {
  // REMOVED text update: setLabel("init-label", report.text);
  
  // Check for completion
  if (report.progress == 1.0) {
     // REMOVED text update: setLabel("init-label", "Model Loaded ✅");
  }
};

// initially selected model
const selectedModel = "Qwen3-0.6B-q4f16_1-MLC";

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

class ExternalLLMClient {
  apiUrl: string;
  apiKey: string;
  model: string;

  constructor(apiUrl: string, apiKey: string, model: string) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.model = model;
  }

  chat = {
    completions: {
      create: async (params: any) => {
        let url = this.apiUrl;
        if (!url.endsWith('/chat/completions')) {
            // remove trailing slash if present before appending
            if (url.endsWith('/')) {
                url = url.slice(0, -1);
            }
            url += '/chat/completions';
        }

        // Construct body
        const body: any = {
            model: this.model,
            messages: params.messages,
            temperature: params.temperature,
            response_format: params.response_format
        };
        
        // If enable_thinking is passed, we might need to adapt it or ignore it depending on provider.
        // Standard OpenAI doesn't use extra_body for thinking usually, but DeepSeek might.
        // We will pass extra_body if present, assuming the user knows what they are doing with the model they selected.
        if (params.extra_body) {
            Object.assign(body, params.extra_body);
        }

        console.log("External API Request:", url, body);

        let response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`
          },
          body: JSON.stringify(body)
        });

        if (!response.ok) {
            const errorText = await response.text();
            
            // Auto-retry logic for response_format error (common with some local servers)
            if (response.status === 400 && errorText.includes("response_format")) {
                console.warn("External API rejected response_format, retrying with type='text'...");
                body.response_format = { type: "text" };
                response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.apiKey}`
                    },
                    body: JSON.stringify(body)
                });
                
                if (!response.ok) {
                    const errorTextRetry = await response.text();
                     throw new Error(`External API Error (after retry): ${response.status} ${response.statusText} - ${errorTextRetry}`);
                }
            } else {
                 throw new Error(`External API Error: ${response.status} ${response.statusText} - ${errorText}`);
            }
        }

        const data = await response.json();
        return data;
      }
    }
  }
}

(async () => {
  // REMOVED text update: modelName.innerText = "Loading classifier model...";
  document.getElementById("loading-indicator")!.style.display = "block";
  
  // Load settings
  const settings = await new Promise<any>((resolve) => {
    chrome.storage.sync.get({
        useExternal: false,
        apiUrl: "https://api.openai.com/v1",
        apiKey: "",
        model: "gpt-4o"
    }, resolve);
  });

  if (settings.useExternal) {
      console.log("Using External API with model:", settings.model);
      const client = new ExternalLLMClient(settings.apiUrl, settings.apiKey, settings.model);
      engine = client as any;
      // Simulate init completion
      if (initProgressCallback) {
          initProgressCallback({ progress: 1.0, text: "External Model Ready", timeElapsed: 0 } as any);
      }
  } else {
      console.log("Using Local WebLLM");
      // Initialize CUSTOM client
      const client = new OffscreenLLMClient(selectedModel);
      await client.init(initProgressCallback);
      
      // Replace 'engine' usage with 'client'
      // The interface is mocked to match what you used before
      engine = client as any; 
  }
  
  // REMOVED text update: modelName.innerText = "Model loaded. Waiting for email content...";
  
  // Check if we already have context (page might have loaded faster than model)
  if (context) {
    classifyEmail();
  }
})();


// New function to handle classification
async function classifyEmail() {
  if (!context || !engine) return;

  // REMOVED text update: modelName.innerText = "Classifying email...";
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
  
  // Removed disable-thinking checkbox logic
  const shouldDisableThinking = true; // Default to true (disabled) or false based on your preference.
  // Based on your previous HTML "checked" state, it was checked by default, so disable thinking = true.

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
      // Send to API if enabled
      await sendToApi(result, context);
    } catch (e) {
      console.error("JSON Parse Error", e);
      displayResult({ score: "Error", reasons: ["Failed to parse LLM response"] }, resultText);
    }

  } catch (err) {
    console.error("Classification Error:", err);
  } finally {
    document.getElementById("loading-indicator")!.style.display = "none";
  }
}

// Generate or retrieve persistent installation ID
async function getInstallationId(): Promise<string> {
  const result = await chrome.storage.local.get(['installationId']);
  
  if (result.installationId) {
    return result.installationId;
  }
  
  // Generate a new installation ID (UUID v4 format)
  const installationId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
  
  // Store it for persistence
  await chrome.storage.local.set({ installationId });
  
  return installationId;
}

// Function to send classification results to external API
async function sendToApi(classificationResult: any, emailContext: string) {
  try {
    // Load API settings from storage
    const settings = await chrome.storage.sync.get(['apiEnabled', 'apiEndpoint', 'apiKey']);
    
    if (!settings.apiEnabled || !settings.apiEndpoint) {
      console.log("API output disabled or not configured");
      return;
    }
    
    // Get or generate installation ID
    const installationId = await getInstallationId();

    // Parse email context to include in payload
    let emailData: any = {};
    try {
      emailData = JSON.parse(emailContext);
      // Remove body and subject from email data
      const { body, subject, ...emailDataWithoutBodyAndSubject } = emailData;
      emailData = emailDataWithoutBodyAndSubject;
    } catch (e) {
      // If parsing fails, don't include body
      emailData = {};
    }

    // Prepare payload
    const payload = {
      installationId: installationId,
      classification: {
        score: classificationResult.score,
        flags: classificationResult.reasons || []
      },
      email: emailData
    };

    // Prepare headers
    const headers: HeadersInit = {
      'Content-Type': 'application/json'
    };
    
    if (settings.apiKey) {
      headers['Authorization'] = `Bearer ${settings.apiKey}`;
    }

    // Show API status indicator
    updateApiStatus('sending');

    // Send to API
    const response = await fetch(settings.apiEndpoint, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    const responseData = await response.json().catch(() => ({}));
    console.log("API response:", responseData);
    updateApiStatus('success');

  } catch (error) {
    console.error("API Error:", error);
    updateApiStatus('error', error instanceof Error ? error.message : String(error));
  }
}

// Update API status indicator in UI
function updateApiStatus(status: 'sending' | 'success' | 'error', message?: string) {
  const apiStatusEl = document.getElementById("api-status");
  if (!apiStatusEl) return;

  apiStatusEl.style.display = "block";
  apiStatusEl.className = `api-status ${status}`;

  switch (status) {
    case 'sending':
      apiStatusEl.textContent = '📤 Sending to API...';
      apiStatusEl.style.color = '#666';
      break;
    case 'success':
      apiStatusEl.textContent = '✅ Sent to API successfully';
      apiStatusEl.style.color = '#2e7d32';
      setTimeout(() => {
        apiStatusEl.style.display = 'none';
      }, 3000);
      break;
    case 'error':
      apiStatusEl.textContent = `❌ API Error: ${message || 'Unknown error'}`;
      apiStatusEl.style.color = '#c62828';
      setTimeout(() => {
        apiStatusEl.style.display = 'none';
      }, 5000);
      break;
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

  // FORCE RESIZE: Send message immediately after rendering
  setTimeout(() => {
      const height = document.documentElement.scrollHeight;
      window.parent.postMessage({ type: "webllm-resize", height }, "*");
  }, 50); // Small 50ms delay to let DOM reflow
}

// Monitor content size changes and notify the parent iframe
const resizeObserver = new ResizeObserver(() => {
  // Calculate total height (documentElement usually covers margins/padding best)
  const height = document.documentElement.scrollHeight;
  // Post message to parent (content.js)
  window.parent.postMessage({ type: "webllm-resize", height }, "*");
});

// Start observing the body for size changes
resizeObserver.observe(document.body);
