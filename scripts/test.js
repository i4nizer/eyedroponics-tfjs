const tf = require('@tensorflow/tfjs-node');
const fs = require('fs-extra');
const path = require('path');

const MODEL_PATH = './models/pest-classifier/model.json';
const TEST_PATH = './data/test';




// Helper function to load a single image
async function loadImage(imagePath) {
    const imageBuffer = await fs.readFile(imagePath);
    return tf.node.decodeImage(imageBuffer)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims(0);
}

// Predict pest
async function predict(imagePath) {
    const model = await tf.loadLayersModel(`file://${MODEL_PATH}`);
    const imageTensor = await loadImage(imagePath);

    const predictions = model.predict(imageTensor);
    const predictionIndex = predictions.argMax(-1).dataSync()[0];

    console.log('Predicted category index:', predictionIndex);
}

// Run prediction on test images
async function test() {
    const categories = await fs.readdir(TEST_PATH);
    for (const category of categories) {
        const categoryPath = path.join(TEST_PATH, category);
        const files = await fs.readdir(categoryPath);

        for (const file of files) {
            const filePath = path.join(categoryPath, file);
            console.log(`Testing ${filePath}...`);
            await predict(filePath);
        }
    }
}



// Start testing
test().catch(console.error);
