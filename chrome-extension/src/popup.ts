/* eslint-disable @typescript-eslint/no-non-null-assertion */
"use strict";

import "./popup.css";

import {
  MLCEngineInterface,
  InitProgressReport,
  CreateMLCEngine,
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
You are an email security classifier.
You receive:
from address
subject
recipients
attachment name
urls
body

Your task: rate the email on a confidence score scale of 0-100 based on how many malicious indicators it has with 0 being benign and 100 being very confident the email is malicious

Define  higher scores as emails that clearly try to:
get money, bank details, card details, crypto, or account credentials
promise large unexpected sums (lotteries, inheritances, business deals, investments, 419-style stories)
get the user to click links or open files to fix, verify, unlock, or secure something
impersonate banks, governments, large companies, or senior executives to pressure the user

You must output ONLY valid JSON, no extra text:
{
  "score": "0-100",
  "reasons": ["reason 1", "reason 2"]
}

Rules for "reasons":
Select from the following list of reasons, pick as many that are relevant based on the context of the email; 
["Suspicious Sender Address","Generic Greetings","Urgent or threatening language","Suspicious urls","suspicious attachment names","spelling and grammar mistakes", "too good to be true offers", "requests for personal information"]
Do NOT output placeholders like "reason1" or "reason2".
Do NOT copy reasons from the example; adapt them to the current email.
IF the confidence score you have given the email is below 50 then you do not have to provide reasons 

END_EXAMPLE
NOW CLASSIFY THIS EMAIL:
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
const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";

let engine: MLCEngineInterface;

(async () => {
  modelName.innerText = "Loading classifier model...";
  
  // Initialize engine
  engine = await CreateMLCEngine(selectedModel, {
    initProgressCallback: initProgressCallback,
  });
  
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

  const messages: ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: `Analyze this email:\n${context}` }
  ];


  console.log("Messages:", messages);
  // Read the disable-thinking checkbox
  const disableThinkingCheckbox = document.getElementById("disable-thinking") as HTMLInputElement;
  const shouldDisableThinking = disableThinkingCheckbox ? disableThinkingCheckbox.checked : false;

  try {
    const completion = await engine.chat.completions.create({
      stream: false, 
      messages: messages,
      temperature: 0.1, // <--- Add this line to reduce randomness
      response_format: { type: "json_object" },
      extra_body: {
        enable_thinking: !shouldDisableThinking,
      },
    });

    let resultText = completion.choices[0].message.content || "{}"; // changed const to let
    console.log("Raw LLM Response:", resultText);
    
    // Remove <think>...</think> blocks if they exist
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

  // Debug raw output
  rawJson.innerText = "Raw Output:\n" + rawText;
}
