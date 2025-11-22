/* eslint-disable @typescript-eslint/no-non-null-assertion */
"use strict";

// This code is partially adapted from the openai-chatgpt-chrome-extension repo:
// https://github.com/jessedi0n/openai-chatgpt-chrome-extension

import "./popup.css";
import "./cache-polyfill";

import {
  MLCEngineInterface,
  InitProgressReport,
  CreateMLCEngine,
  ChatCompletionMessageParam,
  prebuiltAppConfig,
  AppConfig,
} from "@mlc-ai/web-llm";
import modelConfig from "../model-config.json";
import { Line } from "progressbar.js";

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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queryInput = getElementAndCheck("query-input")!;
const submitButton = getElementAndCheck("submit-button")!;
const modelName = getElementAndCheck("model-name");

let context = "";
let modelDisplayName = "";

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

function fetchPageContents() {
  updateDebugStatus("Attempting to connect to page...");
  chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
    if (!tabs[0].id) {
      updateDebugStatus("Error: No active tab found.");
      return;
    }
    
    try {
      const port = chrome.tabs.connect(tabs[0].id, { name: "channelName" });
      updateDebugStatus("Connected to page. Requesting content...");
      
      port.postMessage({});
      
      port.onMessage.addListener(function (msg) {
        console.log("Page contents received:", msg.contents);
        context = msg.contents;
        updateDebugStatus("Success! Metadata extracted.", msg.contents);
      });
      
      // Add a fallback timeout in case content.js is dead
      setTimeout(() => {
        if (!context) {
           updateDebugStatus("Timeout: No response from page. Try refreshing the Outlook tab.");
        }
      }, 2000);
      
    } catch (e) {
       updateDebugStatus("Connection Failed: " + e);
    }
  });
}

(<HTMLButtonElement>submitButton).disabled = true;

let progressBar: InstanceType<typeof Line> = new Line("#loadingContainer", {
  strokeWidth: 4,
  easing: "easeInOut",
  duration: 1400,
  color: "#ffd166",
  trailColor: "#eee",
  trailWidth: 1,
  svgStyle: { width: "100%", height: "100%" },
});

let isLoadingParams = true;

let initProgressCallback = (report: InitProgressReport) => {
  setLabel("init-label", report.text);
  progressBar.animate(report.progress, {
    duration: 50,
  });
  if (report.progress == 1.0) {
    enableInputs();
  }
};

// initially selected model
let selectedModel = modelConfig.modelName;

// Configure app to use local model files
// Model files should be in src/models/<modelName>/ and will be bundled in dist/models/
const localModelPath = chrome.runtime.getURL(`models/${modelConfig.modelName}/`);
// WebLLM expects HuggingFace-style URLs ending with /resolve/<branch>/
// Append a fake resolve/main/ segment followed by ../../ so the final resolved URL
// still points at our local folder but satisfies their validation logic.
const localModelBaseForWebLLM = `${localModelPath}resolve/main/../../`;

// Find the prebuilt model entry to get the correct model_lib URL
const prebuiltModel = prebuiltAppConfig.model_list.find(
  m => m.model_id === modelConfig.modelName
);

// Use the prebuilt model_lib URL (WASM file) but override model_url to use local files
const appConfig: AppConfig = {
  ...prebuiltAppConfig,
  model_list: [
    {
      model_id: modelConfig.modelName,
      model: localModelBaseForWebLLM,
      model_lib: prebuiltModel?.model_lib || modelConfig.modelLibUrl,
      // Include other properties from prebuilt model if they exist
      ...(prebuiltModel ? {
        vram_required_MB: prebuiltModel.vram_required_MB,
        low_resource_required: prebuiltModel.low_resource_required,
        overrides: prebuiltModel.overrides,
        required_features: prebuiltModel.required_features,
      } : {}),
    },
    // Keep other prebuilt models as fallback
    ...prebuiltAppConfig.model_list.filter(m => m.model_id !== modelConfig.modelName),
  ],
};

// populate model-selection
const modelSelector = getElementAndCheck(
  "model-selection",
) as HTMLSelectElement;
// Use appConfig instead of prebuiltAppConfig to include local model
for (let i = 0; i < appConfig.model_list.length; ++i) {
  const model = appConfig.model_list[i];
  const opt = document.createElement("option");
  opt.value = model.model_id;
  opt.innerHTML = model.model_id;
  opt.selected = false;

  // set initial selection as the initially selected model
  if (model.model_id == selectedModel) {
    opt.selected = true;
  }

  modelSelector.appendChild(opt);
}

let engine: MLCEngineInterface;

(async () => {
  modelName.innerText = "Loading initial model...";
  engine = await CreateMLCEngine(selectedModel, {
    initProgressCallback: initProgressCallback,
    appConfig: appConfig,
  });
  modelName.innerText = "Now chatting with " + modelDisplayName;
})();

let chatHistory: ChatCompletionMessageParam[] = [];

function enableInputs() {
  if (isLoadingParams) {
    sleep(500);
    isLoadingParams = false;
  }

  // remove loading bar and loading bar descriptors, if exists
  const initLabel = document.getElementById("init-label");
  initLabel?.remove();
  const loadingBarContainer = document.getElementById("loadingContainer")!;
  loadingBarContainer?.remove();
  queryInput.focus();

  const modelNameArray = selectedModel.split("-");
  modelDisplayName = modelNameArray[0];
  let j = 1;
  while (j < modelNameArray.length && modelNameArray[j][0] != "q") {
    modelDisplayName = modelDisplayName + "-" + modelNameArray[j];
    j++;
  }
}

let requestInProgress = false;

// Disable submit button if input field is empty
queryInput.addEventListener("keyup", () => {
  if (
    (<HTMLInputElement>queryInput).value === "" ||
    requestInProgress ||
    isLoadingParams
  ) {
    (<HTMLButtonElement>submitButton).disabled = true;
  } else {
    (<HTMLButtonElement>submitButton).disabled = false;
  }
});

// If user presses enter, click submit button
queryInput.addEventListener("keyup", (event) => {
  if (event.code === "Enter") {
    event.preventDefault();
    submitButton.click();
  }
});

// Listen for clicks on submit button
async function handleClick() {
  requestInProgress = true;
  (<HTMLButtonElement>submitButton).disabled = true;

  // Get the message from the input field
  const message = (<HTMLInputElement>queryInput).value;
  console.log("message", message);
  // Clear the answer
  document.getElementById("answer")!.innerHTML = "";
  // Hide the answer
  document.getElementById("answerWrapper")!.style.display = "none";
  // Show the loading indicator
  document.getElementById("loading-indicator")!.style.display = "block";

  // Generate response
  let inp = message;
  if (context.length > 0) {
    inp =
      "Use only the following context when answering the question at the end. Don't use any other knowledge.\n" +
      context +
      "\n\nQuestion: " +
      message +
      "\n\nHelpful Answer: ";
  }
  console.log("Input:", inp);
  chatHistory.push({ role: "user", content: inp });

  let curMessage = "";
  // Check if the user wants to disable thinking
  const disableThinkingCheckbox = document.getElementById(
    "disable-thinking",
  ) as HTMLInputElement;
  const shouldDisableThinking = disableThinkingCheckbox
    ? disableThinkingCheckbox.checked
    : false;

  const completion = await engine.chat.completions.create({
    stream: true,
    messages: chatHistory,
    extra_body: {
      enable_thinking: !shouldDisableThinking,
    },
  });
  for await (const chunk of completion) {
    const curDelta = chunk.choices[0].delta.content;
    if (curDelta) {
      curMessage += curDelta;
    }
    updateAnswer(curMessage);
  }
  const response = await engine.getMessage();
  chatHistory.push({ role: "assistant", content: await engine.getMessage() });
  console.log("response", response);

  requestInProgress = false;
  (<HTMLButtonElement>submitButton).disabled = false;
}
submitButton.addEventListener("click", handleClick);

// listen for changes in modelSelector
async function handleSelectChange() {
  if (isLoadingParams) {
    return;
  }

  modelName.innerText = "";

  const initLabel = document.createElement("p");
  initLabel.id = "init-label";
  initLabel.innerText = "Initializing model...";
  const loadingContainer = document.createElement("div");
  loadingContainer.id = "loadingContainer";

  const loadingBox = getElementAndCheck("loadingBox");
  loadingBox.appendChild(initLabel);
  loadingBox.appendChild(loadingContainer);

  isLoadingParams = true;
  (<HTMLButtonElement>submitButton).disabled = true;

  if (requestInProgress) {
    engine.interruptGenerate();
  }
  engine.resetChat();
  chatHistory = [];
  await engine.unload();

  selectedModel = modelSelector.value;

  progressBar = new Line("#loadingContainer", {
    strokeWidth: 4,
    easing: "easeInOut",
    duration: 1400,
    color: "#ffd166",
    trailColor: "#eee",
    trailWidth: 1,
    svgStyle: { width: "100%", height: "100%" },
  });

  initProgressCallback = (report: InitProgressReport) => {
    setLabel("init-label", report.text);
    progressBar.animate(report.progress, {
      duration: 50,
    });
    if (report.progress == 1.0) {
      enableInputs();
    }
  };

  engine.setInitProgressCallback(initProgressCallback);

  requestInProgress = true;
  modelName.innerText = "Reloading with new model...";
  await engine.reload(selectedModel, {
    appConfig: appConfig,
  });
  requestInProgress = false;
  modelName.innerText = "Now chatting with " + modelDisplayName;
}
modelSelector.addEventListener("change", handleSelectChange);

// Listen for messages from the background script
chrome.runtime.onMessage.addListener(({ answer, error }) => {
  if (answer) {
    updateAnswer(answer);
  }
});

function updateAnswer(answer: string) {
  // Show answer
  document.getElementById("answerWrapper")!.style.display = "block";
  const answerWithBreaks = answer.replace(/\n/g, "<br>");
  document.getElementById("answer")!.innerHTML = answerWithBreaks;
  // Add event listener to copy button
  document.getElementById("copyAnswer")!.addEventListener("click", () => {
    // Get the answer text
    const answerText = answer;
    // Copy the answer text to the clipboard
    navigator.clipboard
      .writeText(answerText)
      .then(() => console.log("Answer text copied to clipboard"))
      .catch((err) => console.error("Could not copy text: ", err));
  });
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  };
  const time = new Date().toLocaleString("en-US", options);
  // Update timestamp
  document.getElementById("timestamp")!.innerText = time;
  // Hide loading indicator
  document.getElementById("loading-indicator")!.style.display = "none";
}
