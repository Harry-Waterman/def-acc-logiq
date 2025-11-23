# TinyRod - Chrome Extension for Phishing Detection

TinyRod is a client-side phishing detection extension for Outlook webmail (OWA) and Office 365. It leverages **WebLLM** and **WebGPU** to run a powerful Transformer model (Llama-3.1-8B) directly in the browser, ensuring email data never leaves the user's device unless configured otherwise.

## 🚀 Features

-   **Local Inference**: Runs `Llama-3.1-8B-Instruct` entirely in the browser using WebGPU. No data is sent to the cloud by default.
-   **Outlook Integration**: Seamlessly injects a "Scan" button into the Outlook message toolbar.
-   **Smart Extraction**: Automatically parses email metadata (Sender, Subject, Body, URLs, Attachments) from the Outlook DOM.
-   **Real-time Analysis**: Provides a maliciousness score (0-100) and specific reasons for the verdict (e.g., "Sender address mismatch", "Urgent language").
-   **Enterprise Ready**: Optional configuration to offload inference to a centralized API (e.g., Harbour) for aggregation and deeper analysis.

## 🛠 Architecture

The extension uses a modern Manifest V3 architecture with a focus on performance and security.

### Components

1.  **Content Script (`content.js`)**:
    *   **Role**: DOM interaction and UI injection.
    *   **Function**:
        *   Detects when the user is on Outlook/Office 365.
        *   Injects the "Scan" button into the email toolbar.
        *   On click, creates an iframe to host the Popup UI.
        *   Extracts structured data (From, To, Body, URLs) from the active reading pane when requested.

2.  **Popup UI (`popup.html` / `popup.ts`)**:
    *   **Role**: User Interface and Orchestration.
    *   **Function**:
        *   Displays the scanning status and results.
        *   Connects to the Content Script to retrieve email data.
        *   Pre-processes data (truncates long bodies, limits URLs) to fit context windows.
        *   Constructs the prompt and sends it to the inference engine.

3.  **Offscreen Document (`offscreen.ts` / `offscreen.html`)**:
    *   **Role**: Dedicated WebLLM Host.
    *   **Function**:
        *   Hosts the `CreateMLCEngine` from `@mlc-ai/web-llm`.
        *   This isolates the heavy WebGPU processing from the UI thread and Service Worker, ensuring stability and persistence during inference.

4.  **Service Worker (`background.ts`)**:
    *   **Role**: Message Bridge.
    *   **Function**:
        *   Routes messages between the Popup (UI) and the Offscreen Document (Engine).
        *   Manages the lifecycle of the Offscreen document.

### Data Flow (Local Mode)

1.  **User** opens an email and clicks "Scan".
2.  **Content Script** creates the Popup iframe.
3.  **Popup** requests email content from the Content Script.
4.  **Content Script** scrapes the DOM and returns a JSON object (Subject, Body, etc.).
5.  **Popup** sends a classification request to the **Service Worker**.
6.  **Service Worker** forwards the request to the **Offscreen Document**.
7.  **Offscreen Document** runs the Llama-3.1 model via WebGPU and returns the JSON classification.
8.  **Popup** displays the Score and Reasons to the user.

## 📦 Installation

### Prerequisites
-   Google Chrome or a Chromium-based browser (Edge, Brave).
-   A GPU capable of running WebGPU (most modern integrated or dedicated GPUs).

### Setup
1.  Navigate to the `def-acc-logiq/chrome-extension` directory.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Build the extension:
    ```bash
    npm run build
    ```
4.  Open Chrome and go to `chrome://extensions/`.
5.  Enable **Developer mode** (top right).
6.  Click **Load unpacked** and select the `def-acc-logiq/chrome-extension/dist` folder.

## ⚙️ Configuration

### Local vs. Remote Inference
By default, TinyRod runs locally. However, for enterprise deployment or testing, you can configure it to use an external API.

1.  Right-click the extension icon and select **Options**.
2.  **Use External API**: Check this box to disable local WebLLM.
3.  **API URL**: Endpoint for the LLM (e.g., `https://api.openai.com/v1` or your internal Harbour instance).
4.  **API Key**: Your access token.
5.  **Model**: The model name string (e.g., `gpt-4o` or a custom model ID).

### System Prompt
The extension uses a strict system prompt to ensure consistent JSON output. It evaluates emails based on:
-   Sender/Display name mismatches
-   Generic greetings
-   Urgent/Threatening language
-   Suspicious URLs or attachments
-   Requests for personal info

## 🧩 Development

-   **`src/manifest.json`**: Extension configuration.
-   **`src/offscreen.ts`**: WebLLM engine initialization and chat completion logic.
-   **`src/content.js`**: Logic for finding the Outlook toolbar and scraping specific DOM elements (e.g., `aria-label="Reading Pane"`).
