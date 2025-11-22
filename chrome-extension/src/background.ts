
// Ensure the offscreen document exists
async function ensureOffscreenDocument() {
  try {
    await chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: "Hosting WebLLM Engine for WebGPU access",
    });
  } catch (err: any) {
    // If it already exists, chrome throws an error starting with "Only a single..."
    if (err.message.indexOf("Only a single offscreen") === -1) {
      throw err;
    }
  }
}

// Bridge: Listen for standard messages from Popup/Content
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // If the message is an "init-progress" update from the offscreen doc,
    // just let it bubble up to the popup (which also listens to runtime messages).
    // We don't need to forward it back to offscreen.
    if (msg.type === "init-progress") {
      return; 
    }

    ensureOffscreenDocument().then(() => {
        // Forward message to the offscreen document
        chrome.runtime.sendMessage(msg, (response) => {
            // Forward the response back to the original sender (Popup)
            sendResponse(response);
        });
    });
    
    return true; // keep channel open for async response
});
