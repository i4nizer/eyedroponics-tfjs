const fs = require('fs-extra');
const path = require('path');

// Paths
const DATASET_PATH = 'C:/Users/ADMIN/Downloads/datasets'; // Original 'datasets' folder
const DATA_PATH = './data'; // New data folder structure
const SPLIT_RATIO = { train: 0.8, val: 0.1, test: 0.1 }; // Split ratios for train, val, and test



// Ensure the target directory structure exists
async function createDirectories() {
    const subfolders = ['train', 'val', 'test'];
    const classes = ['whitefly', 'snail', 'slug', 'leaf_miner', 'aphids', 'armyworm', 'cutworm'];

    // Create 'data/train', 'data/val', 'data/test' directories if they don't exist
    for (const subfolder of subfolders) {
        const subfolderPath = path.join(DATA_PATH, subfolder);
        await fs.ensureDir(subfolderPath);

        // Create class subfolders inside train, val, and test
        for (const className of classes) {
            const classPath = path.join(subfolderPath, className);
            await fs.ensureDir(classPath);
        }
    }
}

// Shuffle array function to randomize image order
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]]; // Swap elements
    }
    return arr;
}

// Distribute images to train, val, and test directories
async function distributeImages() {
    const classes = await fs.readdir(DATASET_PATH);
    
    for (const className of classes) {
        const classPath = path.join(DATASET_PATH, className, 'train');
        const images = await fs.readdir(classPath);

        // Shuffle the images for randomness
        const shuffledImages = shuffleArray(images);

        // Split the images based on ratios
        const totalImages = shuffledImages.length;
        const trainCount = Math.floor(totalImages * SPLIT_RATIO.train);
        const valCount = Math.floor(totalImages * SPLIT_RATIO.val);
        const testCount = totalImages - trainCount - valCount; // Remainder goes to test

        const trainImages = shuffledImages.slice(0, trainCount);
        const valImages = shuffledImages.slice(trainCount, trainCount + valCount);
        const testImages = shuffledImages.slice(trainCount + valCount);

        // Copy images to respective directories
        await copyImagesToFolder(trainImages, className, 'train');
        await copyImagesToFolder(valImages, className, 'val');
        await copyImagesToFolder(testImages, className, 'test');
    }
}

// Copy images to the appropriate folder
async function copyImagesToFolder(images, className, subfolder) {
    const targetFolder = path.join(DATA_PATH, subfolder, className);

    for (const image of images) {
        const sourceImagePath = path.join(DATASET_PATH, className, 'train', image);
        const targetImagePath = path.join(targetFolder, image);

        await fs.copy(sourceImagePath, targetImagePath);
        console.log(`Copied ${image} to ${targetImagePath}`);
    }
}

// Main function to run the script
async function main() {
    await createDirectories();
    await distributeImages();
    console.log('Image distribution complete!');
}



// Run the script
main().catch(console.error);
