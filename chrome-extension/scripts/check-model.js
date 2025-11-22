/**
 * Pre-build script to check if model files exist
 */

const fs = require('fs');
const path = require('path');
const modelConfig = require('../model-config.json');

const modelDir = path.join(__dirname, '..', 'src', 'models', modelConfig.modelName);
const huggingFaceUrl = `https://huggingface.co/${modelConfig.huggingFaceRepo}`;

if (!fs.existsSync(modelDir)) {
  console.error('\n❌ Model directory not found!');
  console.error('Expected location:', modelDir);
  console.error('\nPlease run: npm run download-model');
  console.error('Or manually download from:', huggingFaceUrl);
  process.exit(1);
}

const files = fs.readdirSync(modelDir);
if (files.length === 0 || files.length === 1 && files[0] === '.gitkeep') {
  console.error('\n❌ Model files not found!');
  console.error('Model directory exists but is empty.');
  console.error('\nPlease run: npm run download-model');
  console.error('Or manually download from:', huggingFaceUrl);
  process.exit(1);
}

// Check for essential files
const essentialFiles = ['mlc-chat-config.json', 'tokenizer.json'];
const hasEssentialFiles = essentialFiles.some(file => files.includes(file));

if (!hasEssentialFiles) {
  console.warn('\n⚠️  Warning: Some essential model files may be missing.');
  console.warn('Expected files:', essentialFiles.join(', '));
  console.warn('Found files:', files.slice(0, 10).join(', '), files.length > 10 ? '...' : '');
}

console.log('✅ Model files found. Proceeding with build...');

