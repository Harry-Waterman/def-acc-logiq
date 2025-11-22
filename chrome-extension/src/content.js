// Only the content script is able to access the DOM
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

    // Helper: Extract Attachments
    const extractAttachments = (container) => {
      const attachments = new Set();

      // 1. Look for the main Attachment list container
      // Outlook typically groups them in a div with aria-label="Attachments"
      const attachmentLists = container.querySelectorAll('[aria-label="Attachments"]');
      
      if (attachmentLists.length > 0) {
          for (const list of attachmentLists) {
             // Check buttons or list items inside
             // Outlook attachments are often buttons or have role="button"
             const candidates = list.querySelectorAll('div[role="button"], button, li, [data-testid="attachment-card"]');
             for (const item of candidates) {
                 const text = cleanText(item.innerText);
                 if (text) {
                     // Sometimes text includes size like "File.pdf 2MB", which is fine context
                     attachments.add(text);
                 } else {
                     // Try aria-label if text is empty
                     const label = item.getAttribute('aria-label');
                     if (label) attachments.add(cleanText(label));
                 }
             }
          }
      }

      // 2. Fallback: Look for individual items labeled "Attachment: ..."
      // This catches cases where the list container might be missed
      const labeledItems = container.querySelectorAll('[aria-label^="Attachment:"]');
      for (const item of labeledItems) {
          let label = item.getAttribute('aria-label');
          if (label) {
            // Clean up "Attachment: " prefix
            label = label.replace(/^Attachment:\s*/i, '');
            attachments.add(cleanText(label));
          }
      }

      return Array.from(attachments);
    };

    // Helper: Extract Outlook Metadata
    const getOutlookMetadata = (bodyElem) => {
      // Get raw body text first
      let rawBody = bodyElem.innerText;

      const result = {
        subject: null,
        sender: { displayName: null, email: null },
        recipients: null,
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
      }

      return result;
    };

    // STRATEGY 1: User Selection
    const selection = window.getSelection().toString().trim();
    if (selection.length > 0) {
      console.log("WebLLM Extension: Found user selection");
      const urlRegex = /(https?:\/\/[^\s]+)/g;
      const urls = selection.match(urlRegex) || [];
      
      const payload = JSON.stringify({ 
          urls: urls,
          attachments: [], // No attachments in pure text selection usually
          body: cleanText(selection) 
      }, null, 2);
      
      console.log("*** FINAL EXTRACTED METADATA (JSON) ***");
      console.log(payload);
      console.log("***************************************");
      port.postMessage({ contents: payload });
      return;
    }

    // STRATEGY 2: Find Content Body
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
