/**
 * Post-build script to ensure model files are copied to dist/
 * Parcel should handle this automatically, but this ensures it happens
 */

const fs = require('fs');
const path = require('path');

const srcModelDir = path.join(__dirname, '..', 'src', 'models');
const distModelDir = path.join(__dirname, '..', 'dist', 'models');

// Check if models were already copied by Parcel
if (fs.existsSync(distModelDir)) {
  const distFiles = fs.readdirSync(distModelDir).filter(f => f !== '.gitkeep');
  if (distFiles.length > 0) {
    console.log('✅ Model files already copied to dist/');
    return;
  }
}

// If not, copy them manually
if (!fs.existsSync(srcModelDir)) {
  console.warn('⚠️  Source model directory not found. Skipping copy.');
  return;
}

// Check if source has actual model files (not just .gitkeep)
const srcFiles = fs.readdirSync(srcModelDir).filter(f => f !== '.gitkeep');
if (srcFiles.length === 0) {
  console.warn('⚠️  No model files found in source directory. Skipping copy.');
  console.warn('   Run: npm run download-model');
  return;
}

console.log('Copying model files to dist/...');

function copyRecursive(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    // Skip .gitkeep files
    if (entry.name === '.gitkeep') {
      continue;
    }

    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

try {
  copyRecursive(srcModelDir, distModelDir);
  console.log('✅ Model files copied to dist/models/');
} catch (error) {
  console.error('❌ Error copying model files:', error.message);
  console.error('The extension may not work correctly without model files in dist/');
}

