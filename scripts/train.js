const tf = require('@tensorflow/tfjs-node-gpu'); // Use the GPU-accelerated TensorFlow.js backend
const fs = require('fs-extra');
const path = require('path');

const TRAIN_PATH = './data/train';
const VAL_PATH = './data/val';
const MODEL_SAVE_PATH = './models/pest-classifier';



// Helper function to load images and labels from a given directory
async function loadImages(dirPath) {
    const categories = await fs.readdir(dirPath);
    const imageData = [];
    const labels = [];

    for (const [label, category] of categories.entries()) {
        const categoryPath = path.join(dirPath, category);
        const files = await fs.readdir(categoryPath);

        for (const file of files) {
            const filePath = path.join(categoryPath, file);
            const imageBuffer = await fs.readFile(filePath);
            const imageTensor = tf.node.decodeImage(imageBuffer)
                .resizeBilinear([224, 224]) // Resize images
                .toFloat()
                .div(255.0) // Normalize pixel values
                .expandDims(0);

            imageData.push(imageTensor);
            labels.push(label); // Numeric label corresponding to category
        }
    }

    return {
        images: tf.concat(imageData),
        labels: tf.tensor1d(labels, 'int32'),
    };
}

// Build a simple CNN model
function createModel(numClasses) {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [224, 224, 3],
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

// Train the model
async function train() {
    console.log('Checking GPU availability...');
    await tf.setBackend('tensorflow'); // Set backend to GPU if available
    const backend = tf.getBackend();

    if (backend !== 'tensorflow') {
        console.error('GPU backend not available. Exiting...');
        process.exit(1); // Exit if GPU is not available
    }

    console.log('GPU backend is available. Using:', backend);

    console.log('Loading training data...');
    const trainData = await loadImages(TRAIN_PATH);
    console.log('Loading validation data...');
    const valData = await loadImages(VAL_PATH);

    const numClasses = trainData.labels.max().dataSync()[0] + 1;
    const model = createModel(numClasses);

    console.log('Starting training...');
    const BATCH_SIZE = 16;
    const EPOCHS = 5;

    await model.fit(trainData.images, trainData.labels, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        validationData: [valData.images, valData.labels],
        shuffle: true,
    });

    console.log('Saving the model...');
    await model.save(`file://${MODEL_SAVE_PATH}`);
    console.log('Model saved to', MODEL_SAVE_PATH);
}




// Start training
train().catch(console.error);
