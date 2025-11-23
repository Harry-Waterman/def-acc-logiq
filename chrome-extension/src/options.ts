// Options page script for managing extension settings

interface ApiSettings {
  useApi: boolean;
  endpoint: string;
  apiKeyHarbour: string;
  useExternal: boolean;
  aiUrl: string;
  apiKeyAi: string;
  model: string;
}

async function loadSettings(): Promise<ApiSettings> {
  const result = await chrome.storage.sync.get(['apiEnabled', 'apiEndpoint', 'apiKey', 'useExternal', 'aiUrl', 'apiKeyAi', 'model']);
  return {
    useApi: result.apiEnabled || false,
    endpoint: result.apiEndpoint || '',
    apiKeyHarbour: result.apiKey || '',
    useExternal: result.useExternal || false,
    aiUrl: result.aiUrl || '',
    apiKeyAi: result.apiKeyAi || '',
    model: result.model || ''
  };
}

async function saveSettings(settings: ApiSettings): Promise<void> {
  await chrome.storage.sync.set({
    apiEnabled: settings.useApi,
    apiEndpoint: settings.endpoint,
    apiKey: settings.apiKeyHarbour,
    useExternal: settings.useExternal,
    aiUrl: settings.aiUrl,
    apiKeyAi: settings.apiKeyAi,
    model: settings.model
  });
}

function showStatus(message: string, isError: boolean = false) {
  const statusEl = document.getElementById('status')!;
  statusEl.textContent = message;
  statusEl.className = isError ? 'status error' : 'status success';
  
  setTimeout(() => {
    statusEl.className = 'status';
  }, 3000);
}

function toggleApiFields(show: boolean) {
  const apiFields = document.getElementById('apiFields')!;
  apiFields.style.display = show ? 'block' : 'none';
}

function toggleExternalFields(show: boolean) {
  const externalFields = document.getElementById('externalFields')!;
  externalFields.style.display = show ? 'block' : 'none';
}

async function init() {
  // Load existing settings
  const settings = await loadSettings();
  
  const apiEnabledEl = document.getElementById('apiEnabled') as HTMLInputElement;
  const apiEndpointEl = document.getElementById('apiEndpoint') as HTMLInputElement;
  const apiKeyEl = document.getElementById('apiKey') as HTMLInputElement;
  const useExternalEl = document.getElementById('useExternal') as HTMLInputElement;
  const aiUrlEl = document.getElementById('aiUrl') as HTMLInputElement;
  const apiKeyAiEl = document.getElementById('apiKeyAi') as HTMLInputElement;
  const modelEl = document.getElementById('model') as HTMLInputElement;
  const saveBtn = document.getElementById('saveBtn') as HTMLButtonElement;
  
  // Populate form
  apiEnabledEl.checked = settings.useApi;
  apiEndpointEl.value = settings.endpoint;
  apiKeyEl.value = settings.apiKeyHarbour;
  useExternalEl.checked = settings.useExternal;
  aiUrlEl.value = settings.aiUrl;
  apiKeyAiEl.value = settings.apiKeyAi;
  modelEl.value = settings.model;
  
  // Set initial visibility
  toggleApiFields(settings.useApi);
  toggleExternalFields(settings.useExternal);
  
  // Add change listeners for checkboxes
  apiEnabledEl.addEventListener('change', () => {
    toggleApiFields(apiEnabledEl.checked);
  });
  
  useExternalEl.addEventListener('change', () => {
    toggleExternalFields(useExternalEl.checked);
  });
  
  // Save button handler
  saveBtn.addEventListener('click', async () => {
    const endpoint = apiEndpointEl.value.trim();
    const aiUrl = aiUrlEl.value.trim();
    
    if (apiEnabledEl.checked && !endpoint) {
      showStatus('Please enter an API endpoint URL', true);
      return;
    }
    
    if (apiEnabledEl.checked && endpoint && !endpoint.match(/^https?:\/\/.+/)) {
      showStatus('Please enter a valid URL (must start with http:// or https://)', true);
      return;
    }
    
    if (useExternalEl.checked && !aiUrl) {
      showStatus('Please enter an API endpoint URL', true);
      return;
    }
    
    if (useExternalEl.checked && aiUrl && !aiUrl.match(/^https?:\/\/.+/)) {
      showStatus('Please enter a valid URL (must start with http:// or https://)', true);
      return;
    }
    
    try {
      await saveSettings({
        useApi: apiEnabledEl.checked,
        endpoint: endpoint,
        apiKeyHarbour: apiKeyEl.value.trim(),
        useExternal: useExternalEl.checked,
        aiUrl: aiUrl,
        apiKeyAi: apiKeyAiEl.value.trim(),
        model: modelEl.value.trim()
      });
      showStatus('Settings saved successfully!');
    } catch (error) {
      showStatus('Failed to save settings: ' + error, true);
    }
  });
}

init();