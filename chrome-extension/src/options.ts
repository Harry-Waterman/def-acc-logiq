interface Settings {
  useExternal: boolean;
  apiUrl: string;
  apiKey: string;
  model: string;
}

const defaultSettings: Settings = {
  useExternal: false,
  apiUrl: "https://api.openai.com/v1",
  apiKey: "",
  model: "gpt-4o"
};

function saveOptions() {
  const useExternal = (document.getElementById('useExternal') as HTMLInputElement).checked;
  const apiUrl = (document.getElementById('apiUrl') as HTMLInputElement).value;
  const apiKey = (document.getElementById('apiKey') as HTMLInputElement).value;
  const model = (document.getElementById('model') as HTMLInputElement).value;

  chrome.storage.sync.set(
    {
      useExternal,
      apiUrl,
      apiKey,
      model
    },
    () => {
      const status = document.getElementById('status');
      if (status) {
        status.textContent = 'Options saved.';
        setTimeout(() => {
          status.textContent = '';
        }, 2000);
      }
    }
  );
}

function restoreOptions() {
  chrome.storage.sync.get(
    defaultSettings,
    (items) => {
      (document.getElementById('useExternal') as HTMLInputElement).checked = items.useExternal;
      (document.getElementById('apiUrl') as HTMLInputElement).value = items.apiUrl;
      (document.getElementById('apiKey') as HTMLInputElement).value = items.apiKey;
      (document.getElementById('model') as HTMLInputElement).value = items.model;
      
      toggleExternalSettings(items.useExternal);
    }
  );
}

function toggleExternalSettings(show: boolean) {
    const settingsDiv = document.getElementById('externalSettings');
    if (settingsDiv) {
        if (show) {
            settingsDiv.classList.remove('hidden');
        } else {
            settingsDiv.classList.add('hidden');
        }
    }
}

document.addEventListener('DOMContentLoaded', restoreOptions);
document.getElementById('save')?.addEventListener('click', saveOptions);
document.getElementById('useExternal')?.addEventListener('change', (e) => {
    toggleExternalSettings((e.target as HTMLInputElement).checked);
});

