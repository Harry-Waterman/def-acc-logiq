/**
 * Script to download the configured model files (see model-config.json)
 * 
 * Uses: Python's huggingface_hub library (recommended) or manual download
 * 
 * Usage: node download-model.js
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const modelConfig = require('./model-config.json');

const MODEL_NAME = modelConfig.modelName;
const MODEL_ID = modelConfig.huggingFaceRepo;
const TARGET_DIR = path.join(__dirname, 'src', 'models', MODEL_NAME);

console.log('Downloading model:', MODEL_ID);
console.log('Target directory:', TARGET_DIR);

// Create target directory if it doesn't exist
if (!fs.existsSync(TARGET_DIR)) {
  fs.mkdirSync(TARGET_DIR, { recursive: true });
  console.log('Created directory:', TARGET_DIR);
}

// Helper function to find Python executable
function findPython() {
  const pythonCommands = ['python3', 'python'];
  for (const cmd of pythonCommands) {
    try {
      execSync(`${cmd} --version`, { stdio: 'ignore' });
      return cmd;
    } catch (e) {
      // Continue to next command
    }
  }
  return null;
}

// Helper function to check if huggingface_hub is installed
function checkHuggingfaceHub(pythonCmd) {
  try {
    execSync(`${pythonCmd} -c "import huggingface_hub"`, { stdio: 'ignore' });
    return true;
  } catch (e) {
    return false;
  }
}

// Helper function to run Python download script
function runDownloadScript(pythonCmd) {
  // Normalize path for Python (Windows needs special handling)
  const normalizedPath = TARGET_DIR.replace(/\\/g, '/');
  
  const pythonScript = `from huggingface_hub import snapshot_download
import os

model_id = "${MODEL_ID}"
local_dir = r"${normalizedPath}"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete!")
`;
  
  const tempScript = path.join(__dirname, 'temp_download.py');
  fs.writeFileSync(tempScript, pythonScript);
  
  try {
    execSync(`"${pythonCmd}" "${tempScript}"`, {
      stdio: 'inherit',
      cwd: __dirname,
      shell: true,
    });
    
    // Clean up temp script
    if (fs.existsSync(tempScript)) {
      fs.unlinkSync(tempScript);
    }
    
    return true;
  } catch (error) {
    // Clean up temp script
    if (fs.existsSync(tempScript)) {
      fs.unlinkSync(tempScript);
    }
    throw error;
  }
}

// Main download logic
const pythonCmd = findPython();

if (!pythonCmd) {
  console.error('\n❌ Python not found!');
  console.error('Please install Python from https://www.python.org/');
  console.error('\n📥 Manual Download Instructions:');
  console.error('1. Visit: https://huggingface.co/' + MODEL_ID);
  console.error('2. Click "Files and versions" tab');
  console.error('3. Download all files (or use the "Download repository" button)');
  console.error('4. Extract all files to:', TARGET_DIR);
  process.exit(1);
}

console.log(`Found Python: ${pythonCmd}`);

// Try to use existing huggingface_hub installation
if (checkHuggingfaceHub(pythonCmd)) {
  console.log('✅ huggingface_hub is already installed');
  console.log('Attempting to download using Python huggingface_hub...');
  console.log('(This may take a few minutes depending on your internet connection)');
  
  try {
    runDownloadScript(pythonCmd);
    console.log('\n✅ Model downloaded successfully!');
    console.log('Model files are now in:', TARGET_DIR);
    console.log('\nNext steps:');
    console.log('1. Run: npm run build');
    console.log('2. Load the extension from chrome-extension/dist/');
    process.exit(0);
  } catch (error) {
    console.error('\n⚠️  Download failed:', error.message);
    console.error('Trying to install huggingface_hub...\n');
  }
} else {
  console.log('⚠️  huggingface_hub not found. Installing...');
}

// Try to install huggingface_hub and then download
try {
  console.log('Installing huggingface_hub Python package...');
  console.log('(You may be prompted for permission or see installation progress)');
  
  execSync(`"${pythonCmd}" -m pip install huggingface_hub`, {
    stdio: 'inherit',
    shell: true,
  });
  
  console.log('\n✅ huggingface_hub installed successfully!');
  console.log('Attempting to download model...');
  console.log('(This may take a few minutes depending on your internet connection)');
  
  runDownloadScript(pythonCmd);
  
  console.log('\n✅ Model downloaded successfully!');
  console.log('Model files are now in:', TARGET_DIR);
  console.log('\nNext steps:');
  console.log('1. Run: npm run build');
  console.log('2. Load the extension from chrome-extension/dist/');
  process.exit(0);
  
} catch (error) {
  console.error('\n❌ Automatic download failed.');
  console.error('Error:', error.message);
  console.error('\n📥 Manual Download Instructions:');
  console.error('1. Install Python and huggingface_hub:');
  console.error(`   "${pythonCmd}" -m pip install huggingface_hub`);
  console.error('2. Then run this script again:');
  console.error('   npm run download-model');
  console.error('\nOr manually download:');
  console.error('1. Visit: https://huggingface.co/' + MODEL_ID);
  console.error('2. Click "Files and versions" tab');
  console.error('3. Download all files (or use the "Download repository" button)');
  console.error('4. Extract all files to:', TARGET_DIR);
  process.exit(1);
}

