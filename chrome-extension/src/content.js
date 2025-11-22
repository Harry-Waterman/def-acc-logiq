// Only the content script is able to access the DOM
const SCAN_BUTTON_ID = "webllm-scan-button";

const isOutlookHost = () => {
  const host = window.location.hostname || "";
  return /outlook|office/i.test(host);
};

const openExtensionPopup = () => {
  const IFRAME_ID = "webllm-popup-iframe";
  const existingIframe = document.getElementById(IFRAME_ID);
  
  if (existingIframe) {
    existingIframe.remove();
    return;
  }

  // Try to find either the ConversationContainer or ItemContainer (for single message view)
  const conversationContainer = document.querySelector(
    '[data-app-section="ConversationContainer"], [data-app-section="ItemContainer"]'
  );
  
  if (conversationContainer && conversationContainer.parentElement) {
    const iframe = document.createElement("iframe");
    iframe.id = IFRAME_ID;
    iframe.src = chrome.runtime.getURL("popup.html");
    iframe.style.width = "100%";
    iframe.style.boxSizing = "border-box";
    iframe.style.height = "250px"; // Start small or with a default loading height
    iframe.style.transition = "height 0.2s ease-out"; // Smooth animation when resizing
    iframe.style.overflow = "hidden"; // Hide scrollbars while resizing
    iframe.style.border = "none"; 
    iframe.style.marginBottom = "16px";
    iframe.style.borderRadius = "4px";
    iframe.style.paddingRight = "12px";
    
    conversationContainer.parentElement.insertBefore(iframe, conversationContainer);
  } else {
    // Fallback if container not found
    const popupUrl = chrome.runtime.getURL("popup.html");
    window.open(
      popupUrl,
      "_blank",
      "noopener,noreferrer,width=480,height=720",
    );
  }
};

const getBestToolbar = () => {
  // Strict selector for the specific toolbar type
  const selector = '[aria-label="Message actions"][role="toolbar"]';
  
  // Get ALL candidates
  const candidates = Array.from(document.querySelectorAll(selector));
  
  if (candidates.length === 0) return null;

  // Filter out invisible elements (width/height 0) as they might be un-rendered templates
  const visibleCandidates = candidates.filter(el => {
    return el.offsetWidth > 0 && el.offsetHeight > 0;
  });

  if (visibleCandidates.length === 0) return null;

  // Sort by vertical position (Top to Bottom)
  visibleCandidates.sort((a, b) => {
    return a.getBoundingClientRect().top - b.getBoundingClientRect().top;
  });

  // The first one is the top-most one
  return visibleCandidates[0];
};

const ensureScanButton = () => {
  if (!isOutlookHost()) return;

  const bestToolbar = getBestToolbar();
  const existingBtn = document.getElementById(SCAN_BUTTON_ID);

  // SCENARIO 1: We haven't found the right toolbar yet.
  if (!bestToolbar) {
    return;
  }

  // SCENARIO 2: Button exists, but it's in the WRONG place.
  // (e.g., it was injected into the bottom toolbar previously, or the DOM re-rendered).
  if (existingBtn && existingBtn.parentElement !== bestToolbar) {
    // Remove it so we can move it to the right place
    existingBtn.remove();
  }

  // SCENARIO 3: Button exists and is in the RIGHT place.
  if (document.getElementById(SCAN_BUTTON_ID)) {
    // Verify it's still the first child (in case Outlook prepended something else)
    if (bestToolbar.firstChild !== document.getElementById(SCAN_BUTTON_ID)) {
      bestToolbar.insertBefore(document.getElementById(SCAN_BUTTON_ID), bestToolbar.firstChild);
    }
    return; // All good!
  }

  // SCENARIO 4: Create and Inject
  const scanBtn = document.createElement("button");
  scanBtn.id = SCAN_BUTTON_ID;
  scanBtn.type = "button";
  scanBtn.textContent = "Scan";
  
  // Copy class from a sibling button for styling consistency
  // We look for a button inside the toolbar to copy styles from
  const siblingBtn = bestToolbar.querySelector('button');
  const referenceClass = (siblingBtn && siblingBtn.className) || "";
  scanBtn.className = `${referenceClass} webllm-scan-button`.trim();
  
  // Custom styling to position at far left
  scanBtn.style.marginRight = "8px";
  scanBtn.style.marginLeft = "0px";
  scanBtn.style.marginInlineStart = "0px"; 
  
  scanBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    openExtensionPopup();
  });

  // Insert as First Child
  if (bestToolbar.firstChild) {
    bestToolbar.insertBefore(scanBtn, bestToolbar.firstChild);
  } else {
    bestToolbar.appendChild(scanBtn);
  }
};

// Loop Strategy: Check often.
// This is more robust than MutationObserver for race conditions in complex SPAs
// because it self-corrects every 500ms regardless of what events fired.
const startMainLoop = () => {
  if (!isOutlookHost()) return;

  // 1. Run immediately
  ensureScanButton();

  // 2. Run on mutation (reactivity)
  const observer = new MutationObserver(() => {
    ensureScanButton();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // 3. Run on interval (resilience)
  // Catches cases where attributes change but don't trigger childList, 
  // or layout shifts that change which toolbar is "top".
  setInterval(ensureScanButton, 500);
};

startMainLoop();

chrome.runtime.onConnect.addListener(function (port) {
  port.onMessage.addListener(function (msg) {
    console.log("WebLLM Extension: Starting smart content extraction...");

    // Helper: Check if element is visible
    const isVisible = (elem) => {
      if (!elem) return false;
      const style = window.getComputedStyle(elem);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        elem.offsetWidth > 0 &&
        elem.offsetHeight > 0
      );
    };

    // Helper: Clean text for LLM consumption
    const cleanText = (text) => {
      if (!text) return "";
      return text
        .replace(/\t/g, " ") // Replace tabs with space
        .replace(/ /g, " ") // Replace non-breaking spaces
        .replace(/[\uE000-\uF8FF]/g, "") // Remove Private Use Area (icons)
        .replace(//g, "") // Remove specific artifact mentioned
        .replace(/ +/g, " ") // Collapse multiple spaces
        .replace(/\n\s*\n/g, "\n\n") // Collapse multiple empty lines
        .replace(/^ +| +$/gm, "") // Trim start/end of lines
        .replace(/_{3,}/g, "") // Remove long underscores
        .trim();
    };

    // Helper: Extract URLs from a container
    const extractUrls = (container) => {
      const urls = new Set();
      
      // 1. Get all anchor tags
      const links = container.querySelectorAll('a[href]');
      for (const link of links) {
        const href = link.href;
        // Filter out javascript:, mailto:, and internal anchors #
        if (href && href.startsWith('http')) {
          urls.add(href);
        }
      }

      // 2. Scan text for raw URLs (fallback if not linked)
      const text = container.innerText;
      const urlRegex = /(https?:\/\/[^\s]+)/g;
      const matches = text.match(urlRegex);
      if (matches) {
        matches.forEach(url => urls.add(url));
      }

      return Array.from(urls);
    };

    const normalizeAttachmentLabel = (value) => {
      if (!value) return "";
      let normalized = cleanText(value);
      normalized = normalized
        .replace(/\bMore actions\b/gi, "")
        .replace(/\bOpen\b/gi, "")
        .replace(/\s{2,}/g, " ")
        .trim();
      return normalized;
    };

    // Helper: Extract Attachments
    const extractAttachments = (container) => {
      const attachments = new Set();

      const addAttachmentText = (node) => {
        if (!node) return;
        const aria = node.getAttribute?.("aria-label");
        const text = node.innerText;
        const title = node.getAttribute?.("title");
        const candidates = [aria, text, title];
        for (const raw of candidates) {
          const lowerRaw = raw?.toLowerCase();
          if (
            lowerRaw &&
            (lowerRaw.includes("more actions") ||
              lowerRaw.includes("open menu") ||
              lowerRaw.includes("menu options"))
          ) {
            return;
          }
          const normalized = normalizeAttachmentLabel(raw);
          if (!normalized) continue;
          attachments.add(normalized);
          return; // prefer first useful descriptor (usually aria-label)
        }
      };

      // 1. Look for attachment list containers (handle variations like "file attachments")
      const listSelectors = [
        '[aria-label="Attachments"]',
        '[aria-label="Attachment"]',
        '[aria-label*="attachment" i]',
        '[data-testid*="attachment" i]',
        '[data-test-id*="attachment" i]',
      ];
      const attachmentLists = new Set();
      for (const sel of listSelectors) {
        const matches = container.querySelectorAll(sel);
        matches.forEach((node) => attachmentLists.add(node));
      }

      if (attachmentLists.size > 0) {
        for (const list of attachmentLists) {
          // Outlook/Gmail style attachments are often options or buttons
          const candidateSelectors = [
            '[role="button"]',
            '[role="option"]',
            'button',
            'li',
            '[data-testid="attachment-card"]',
            '[data-test-id="attachment-card"]',
            'a[download]',
          ];
          let foundChild = false;
          for (const candidateSel of candidateSelectors) {
            const candidates = list.querySelectorAll(candidateSel);
            if (candidates.length === 0) continue;
            foundChild = true;
            for (const item of candidates) {
              addAttachmentText(item);
            }
          }
          if (!foundChild) {
            // If we didn't find a child entry, treat the container itself as the attachment
            addAttachmentText(list);
          }
        }
      }

      // 2. Fallback: Look for individual items labeled "Attachment: ..."
      // This catches cases where the list container might be missed
      const labeledItems = container.querySelectorAll('[aria-label^="Attachment:" i]');
      for (const item of labeledItems) {
        let label = item.getAttribute("aria-label");
        if (label) {
          // Clean up "Attachment: " prefix
          label = label.replace(/^Attachment:\s*/i, "");
          attachments.add(cleanText(label));
        }
      }

      return Array.from(attachments).filter(Boolean);
    };

    // Helper: Extract Outlook Metadata
    const getOutlookMetadata = (bodyElem) => {
      // Get raw body text first
      let rawBody = bodyElem.innerText;

      const result = {
        subject: null,
        sender: { displayName: null, email: null },
        recipients: null,
        sentTime: null,
        urls: extractUrls(bodyElem),
        attachments: extractAttachments(bodyElem), // New Attachments property
        body: cleanText(rawBody),
      };

      // Try to find the Reading Pane container
      const readingPane = bodyElem.closest(
        '[aria-label="Reading Pane"], [aria-label="Content pane"], [role="main"]',
      );

      if (readingPane) {
        console.log(
          "WebLLM Extension: Found Reading Pane container",
          readingPane,
        );

        // Also try to extract attachments from the whole reading pane (headers often outside body)
        // We merge them to be safe
        const headerAttachments = extractAttachments(readingPane);
        headerAttachments.forEach(att => {
             if (!result.attachments.includes(att)) {
                 result.attachments.push(att);
             }
        });

        // 1. Subject
        const subjectElem = readingPane.querySelector('[role="heading"]');
        if (subjectElem) {
          result.subject = cleanText(subjectElem.innerText);
        }

        // 2. Sender (From)
        const fromElem = readingPane.querySelector('[aria-label^="From"]');
        let rawSender = null;
        let emailFound = null;
        let nameFound = null;

        if (fromElem) {
          rawSender = fromElem.getAttribute("aria-label");
          if (rawSender && rawSender.toLowerCase().startsWith("from:")) {
             nameFound = rawSender.substring(5).trim();
             const match = nameFound.match(/(.*)<(.+@.+)>/);
             if (match) {
               nameFound = match[1].trim();
               emailFound = match[2].trim();
             }
          }
        }

        // AGGRESSIVE EMAIL SEARCH
        if (!emailFound && fromElem) {
             const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/;
             
             const textMatch = fromElem.innerText.match(emailRegex);
             if (textMatch) {
                 emailFound = textMatch[0];
             }

             if (!emailFound) {
                 const descendants = fromElem.querySelectorAll('*');
                 for (const node of descendants) {
                     if (node.title && node.title.match(emailRegex)) {
                         emailFound = node.title.match(emailRegex)[0];
                         break;
                     }
                     const aria = node.getAttribute('aria-label');
                     if (aria && aria.match(emailRegex)) {
                         emailFound = aria.match(emailRegex)[0];
                         break;
                     }
                 }
             }
        }
        
        if (!emailFound) {
           const persona = readingPane.querySelector('.persona, [data-persona-id]');
           if (persona) {
               const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/;
               if (persona.innerText.match(emailRegex)) {
                   emailFound = persona.innerText.match(emailRegex)[0];
               }
           }
        }

        result.sender = {
          displayName: cleanText(nameFound) || "Unknown",
          email: emailFound || "Unknown (Hidden by Outlook)", 
        };

        // 3. Recipients (To)
        const toElem = readingPane.querySelector('[aria-label^="To"]');
        if (toElem) {
          let rawTo = toElem.getAttribute("aria-label");
          if (rawTo.toLowerCase().startsWith("to:")) {
            rawTo = rawTo.substring(3).trim();
          }
          result.recipients = cleanText(rawTo);
        }

        // 4. Sent Time
        const sentTimeElem = readingPane.querySelector('[data-testid="SentReceivedSavedTime"]');
        if (sentTimeElem) {
          result.sentTime = cleanText(sentTimeElem.innerText);
        }

      }

      return result;
    };

    // Find Content Body
    let bestCandidate = null;

    const selectors = [
      'div[aria-label="Message body"]',
      'div[role="main"]',
      'div[aria-label="Reading Pane"]',
    ];

    for (const sel of selectors) {
      const candidates = document.querySelectorAll(sel);
      for (const cand of candidates) {
        if (isVisible(cand)) {
          bestCandidate = cand;
          break;
        }
      }
      if (bestCandidate) break;
    }

    if (!bestCandidate) {
      const allDivs = document.body.querySelectorAll("div, section, main");
      let maxScore = 0;
      for (const div of allDivs) {
        if (!isVisible(div)) continue;
        if (div.offsetHeight < 100 || div.offsetWidth < 200) continue;
        const len = div.innerText.length;
        if (len > maxScore) {
          maxScore = len;
          bestCandidate = div;
        }
      }
    }

    let finalPayload;
    if (bestCandidate) {
      const metadata = getOutlookMetadata(bestCandidate);

      if (metadata.subject || (metadata.sender && metadata.sender.displayName)) {
        finalPayload = JSON.stringify(metadata, null, 2);
      } else {
        finalPayload = JSON.stringify(
          { 
              urls: extractUrls(bestCandidate),
              attachments: extractAttachments(bestCandidate),
              body: cleanText(bestCandidate.innerText) 
          },
          null,
          2,
        );
      }
    } else {
      finalPayload = JSON.stringify(
        { 
            urls: extractUrls(document.body),
            attachments: [],
            body: cleanText(document.body.innerText) 
        },
        null,
        2,
      );
    }

    console.log("*** FINAL EXTRACTED METADATA (JSON) ***");
    console.log(finalPayload);
    console.log("***************************************");

    port.postMessage({ contents: finalPayload });
  });
});

// Listen for resize messages from the iframe (popup.ts)
window.addEventListener("message", (event) => {
  // Security check: ensure the message is what we expect
  if (event.data && event.data.type === "webllm-resize" && typeof event.data.height === "number") {
    const iframe = document.getElementById("webllm-popup-iframe");
    if (iframe) {
      // Ensure we don't shrink below a reasonable minimum (e.g. 50px)
      // Add a larger buffer (20px) to account for body margins/padding that scrollHeight might miss
      const newHeight = Math.max(50, event.data.height + 20);
      iframe.style.height = `${newHeight}px`;
    }
  }
});
