import { MnistData } from './data.js';

async function showExamples(data) {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

function getModel() {

    const model = tf.sequential();

    const INPUT_SHAPE = [28, 28, 1];
    const NUM_OUTPUT_CLASSES = 10;

    // Step 1) Add some layers
    model.add(tf.layers.conv2d({
        inputShape: INPUT_SHAPE,
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        activation: 'softmax'
    }));

    // Step 2) Add optimier, loss and metrics
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    })

    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    //Step 1) Prepare the data
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [X_train, y_train] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ]
    })

    const [X_test, y_test] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ]
    })

    // Step 2 : Fit the model
    return model.fit(X_train, y_train, {
        batchSize: BATCH_SIZE,
        validationData: [X_test, y_test],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks,
    });
}

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);

    await train(model, data);

    await showAccuracy(model, data);
    await showConfusion(model, data);
}

const classNames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

function doPrediction(model, data, testDataSize = 500) {

    const testData = data.nextTestBatch(testDataSize);
    const textxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(textxs).argMax(-1);

    textxs.dispose();
    return [preds, labels];
}

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });
}

document.addEventListener('DOMContentLoaded', run);