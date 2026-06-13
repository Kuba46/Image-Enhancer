/**
 * worker.js — Web Worker для анализа и обработки изображений.
 *
 * Запускается как type:'module', не блокирует UI.
 *
 * Протокол сообщений:
 *   IN  { type: 'process', taskId, imageData, modelData, scalerData }
 *   IN  { type: 'cancel',  taskId }
 *   OUT { type: 'progress', taskId, status, progress }
 *   OUT { type: 'done',     taskId, result: ImageData, params }
 *   OUT { type: 'error',    taskId, message }
 */

'use strict';

const activeTasks = new Map();

function report(taskId, status, progress) {
    self.postMessage({ type: 'progress', taskId, status, progress });
}

function isCancelled(taskId) {
    return !activeTasks.has(taskId);
}

function extractFeatures(imageData) {
    const { data, width, height } = imageData;
    const N = width * height;

    const luminance  = new Float32Array(N);
    const saturation = new Float32Array(N);
    let sumR = 0, sumG = 0, sumB = 0;

    for (let i = 0; i < N; i++) {
        const r = data[i * 4]     / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;

        sumR += r; sumG += g; sumB += b;

        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        luminance[i] = lum;

        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        saturation[i] = maxC === 0 ? 0 : (maxC - minC) / maxC;
    }

    const BINS = 16;
    const hist = new Float32Array(BINS);
    for (let i = 0; i < N; i++) {
        const bin = Math.min(BINS - 1, Math.floor(luminance[i] * BINS));
        hist[bin]++;
    }
    for (let b = 0; b < BINS; b++) hist[b] /= N;

    const meanLum    = luminance.reduce((a, v) => a + v, 0) / N;
    const stdLum     = Math.sqrt(luminance.reduce((a, v) => a + (v - meanLum) ** 2, 0) / N);
    const sortedLum  = Float32Array.from(luminance).sort();
    const medLum     = sortedLum[Math.floor(N / 2)];
    const darkFrac   = luminance.filter(v => v < 0.25).length / N;
    const brightFrac = luminance.filter(v => v > 0.75).length / N;

    const meanSat = saturation.reduce((a, v) => a + v, 0) / N;
    const stdSat  = Math.sqrt(saturation.reduce((a, v) => a + (v - meanSat) ** 2, 0) / N);
    const meanVal = meanLum;

    const meanR = sumR / N;
    const meanG = sumG / N;
    const meanB = sumB / N;

    // Перцентили яркости
    const sorted = Float32Array.from(luminance).sort();
    const p5  = sorted[Math.floor(N * 0.05)];
    const p25 = sorted[Math.floor(N * 0.25)];
    const p75 = sorted[Math.floor(N * 0.75)];
    const p95 = sorted[Math.floor(N * 0.95)];

    // Энтропия гистограммы
    let entropy = 0;
    for (let b = 0; b < BINS; b++) {
        if (hist[b] > 0) entropy -= hist[b] * Math.log2(hist[b]);
    }

    return new Float32Array([
        ...hist,
        meanLum, stdLum, medLum,
        darkFrac, brightFrac,
        meanSat, stdSat, meanVal,
        meanR, meanG, meanB,
        meanR - meanB, meanR - meanG, meanG - meanB,
        p5, p25, p75, p95, entropy,   // +5 признаков = итого 35
    ]);
}

function normalizeFeatures(features, scalerData) {
    const mean  = scalerData.mean;
    const scale = scalerData.scale;
    const out   = new Float32Array(features.length);
    for (let i = 0; i < features.length; i++) {
        out[i] = (features[i] - mean[i]) / scale[i];
    }
    return out;
}

function mlpForward(input, modelData) {
    let x = input;
    for (const layer of modelData.layers) {
        const { weights, bias, activation } = layer;
        const outSize = bias.length;
        const inSize  = x.length;
        const out     = new Float32Array(outSize);
        for (let o = 0; o < outSize; o++) {
            let sum = bias[o];
            for (let i = 0; i < inSize; i++) {
                sum += x[i] * weights[i * outSize + o];
            }
            out[o] = activation === 'relu' ? Math.max(0, sum)
                   : activation === 'tanh' ? Math.tanh(sum)
                   : sum;
        }
        x = out;
    }
    return x; // [brightness_delta, contrast_delta, saturation_delta]
}

function applyCorrections(imageData, brightness, contrast, saturation) {
    const data    = new Uint8ClampedArray(imageData.data);
    const N       = imageData.width * imageData.height;
    const bDelta  = brightness * 0.3;
    const cFactor = 1.0 + contrast   * 0.5;
    const sFactor = 1.0 + saturation * 0.5;

    for (let i = 0; i < N; i++) {
        const idx = i * 4;
        let r = data[idx]     / 255;
        let g = data[idx + 1] / 255;
        let b = data[idx + 2] / 255;

        r += bDelta; g += bDelta; b += bDelta;

        r = (r - 0.5) * cFactor + 0.5;
        g = (g - 0.5) * cFactor + 0.5;
        b = (b - 0.5) * cFactor + 0.5;

        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        r = lum + (r - lum) * sFactor;
        g = lum + (g - lum) * sFactor;
        b = lum + (b - lum) * sFactor;

        data[idx]     = Math.round(Math.max(0, Math.min(255, r * 255)));
        data[idx + 1] = Math.round(Math.max(0, Math.min(255, g * 255)));
        data[idx + 2] = Math.round(Math.max(0, Math.min(255, b * 255)));
    }

    return new ImageData(data, imageData.width, imageData.height);
}

async function processTask(taskId, imageData, modelData, scalerData) {
    try {
        report(taskId, 'processing', 5);
        if (isCancelled(taskId)) return;

        report(taskId, 'processing', 20);
        const features = extractFeatures(imageData);
        if (isCancelled(taskId)) return;

        report(taskId, 'processing', 35);
        const normalized = normalizeFeatures(features, scalerData);
        if (isCancelled(taskId)) return;

        report(taskId, 'processing', 50);
        const [brightness, contrast, saturation] = mlpForward(normalized, modelData);
        if (isCancelled(taskId)) return;

        report(taskId, 'processing', 70);
        const result = applyCorrections(imageData, brightness, contrast, saturation);
        if (isCancelled(taskId)) return;

        report(taskId, 'processing', 95);
        await new Promise(r => setTimeout(r, 50));
        if (isCancelled(taskId)) return;

        activeTasks.delete(taskId);
        self.postMessage(
            { type: 'done', taskId, result, params: { brightness, contrast, saturation } },
            [result.data.buffer]
        );

    } catch (err) {
        activeTasks.delete(taskId);
        self.postMessage({ type: 'error', taskId, message: err.message });
    }
}

self.onmessage = function(e) {
    const { type, taskId } = e.data;

    if (type === 'process') {
        const { imageData, modelData, scalerData } = e.data;
        activeTasks.set(taskId, true);
        processTask(taskId, imageData, modelData, scalerData);
    }

    if (type === 'cancel') {
        activeTasks.delete(taskId);
        self.postMessage({ type: 'progress', taskId, status: 'cancelled', progress: 0 });
    }
};