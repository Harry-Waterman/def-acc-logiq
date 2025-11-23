// Options page script for managing extension settings

interface ApiSettings {
  enabled: boolean;
  endpoint: string;
  apiKey: string;
}

async function loadSettings(): Promise<ApiSettings> {
  const result = await chrome.storage.sync.get(['apiEnabled', 'apiEndpoint', 'apiKey']);
  return {
    enabled: result.apiEnabled || false,
    endpoint: result.apiEndpoint || '',
    apiKey: result.apiKey || ''
  };
}

async function saveSettings(settings: ApiSettings): Promise<void> {
  await chrome.storage.sync.set({
    apiEnabled: settings.enabled,
    apiEndpoint: settings.endpoint,
    apiKey: settings.apiKey
  });
}

function showStatus(message: string, isError: boolean = false) {
  const statusEl = document.getElementById('status')!;
  statusEl.textContent = message;
  statusEl.className = status ;
  
  setTimeout(() => {
    statusEl.className = 'status';
  }, 3000);
}

async function init() {
  // Load existing settings
  const settings = await loadSettings();
  
  const apiEnabledEl = document.getElementById('apiEnabled') as HTMLInputElement;
  const apiEndpointEl = document.getElementById('apiEndpoint') as HTMLInputElement;
  const apiKeyEl = document.getElementById('apiKey') as HTMLInputElement;
  const saveBtn = document.getElementById('saveBtn') as HTMLButtonElement;
  
  // Populate form
  apiEnabledEl.checked = settings.enabled;
  apiEndpointEl.value = settings.endpoint;
  apiKeyEl.value = settings.apiKey;
  
  // Save button handler
  saveBtn.addEventListener('click', async () => {
    const endpoint = apiEndpointEl.value.trim();
    
    if (apiEnabledEl.checked && !endpoint) {
      showStatus('Please enter an API endpoint URL', true);
      return;
    }
    
    if (apiEnabledEl.checked && endpoint && !endpoint.match(/^https?:\/\/.+/)) {
      showStatus('Please enter a valid URL (must start with http:// or https://)', true);
      return;
    }
    
    try {
      await saveSettings({
        enabled: apiEnabledEl.checked,
        endpoint: endpoint,
        apiKey: apiKeyEl.value.trim()
      });
      showStatus('Settings saved successfully!');
    } catch (error) {
      showStatus('Failed to save settings: ' + error, true);
    }
  });
}

init();
