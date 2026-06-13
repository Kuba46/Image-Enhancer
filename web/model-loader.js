/**
 * Загрузка модели и конвертация весов
 * в формат для mlpForward в Web Worker.
 *
 * Поддерживает формат экспортированный train.py::export_model()
 * { layersMeta, weightsManifest }
 */

'use strict';

/**
 * Загружает модель.
 * @param {string} modelUrl  — путь к model.json
 * @returns {Promise<Object>} modelData для worker.js::mlpForward
 */
export async function loadModel(modelUrl) {
    const modelJson = await fetch(modelUrl).then(r => {
        if (!r.ok) throw new Error(`Не удалось загрузить модель: ${modelUrl}`);
        return r.json();
    });

    // Загружаем бинарный шард весов
    const baseUrl         = modelUrl.replace(/\/model\.json$/, '/');
    const weightsManifest = modelJson.weightsManifest ?? [];
    const weightBuffers   = await Promise.all(
        weightsManifest.flatMap(group =>
        group.paths.map(p =>
            fetch(baseUrl + p).then(r => {
                if (!r.ok) throw new Error(`Не удалось загрузить веса: ${p}`);
                return r.arrayBuffer();
            })
        )
        )
    );

    // Объединяем шарды в один буфер
    const totalBytes = weightBuffers.reduce((sum, b) => sum + b.byteLength, 0);
    const combined   = new Uint8Array(totalBytes);
    let offset = 0;
    for (const buf of weightBuffers) {
        combined.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
    }

    // Строим weightMap: name → Float32Array
    const weightSpecs = weightsManifest.flatMap(g => g.weights);
    const weightMap   = {};
    let byteOffset    = 0;
    for (const spec of weightSpecs) {
        const size = spec.shape.reduce((a, b) => a * b, 1);
        weightMap[spec.name] = new Float32Array(combined.buffer, byteOffset, size);
        byteOffset += size * 4;
    }

    // Используем layersMeta из нашего кастомного экспортера
    const layersMeta = modelJson.layersMeta;
    if (!layersMeta || layersMeta.length === 0) {
        throw new Error('model.json не содержит layersMeta — переэкспортируйте модель через train.py');
    }

    const layers = layersMeta.filter(meta => meta.name.startsWith('dense'))
    .map(meta => {
        const kernelKey = Object.keys(weightMap).find(k => k.includes(meta.name) && k.endsWith('kernel'));
        const biasKey   = Object.keys(weightMap).find(k => k.includes(meta.name) && k.endsWith('bias'));

        if (!kernelKey || !biasKey) {
            throw new Error(`Веса не найдены для слоя: ${meta.name}`);
        }

        return {
            weights:    Array.from(weightMap[kernelKey]),
            bias:       Array.from(weightMap[biasKey]),
            activation: meta.activation === 'relu' ? 'relu' : 
            meta.activation === 'tanh' ? 'tanh' : 'linear',
        };
    });

    console.log(`[ModelLoader] Загружено слоёв: ${layers.length}`);
    return { layers };
}

/**
 * Загружает параметры нормализации.
 * @param {string} scalerUrl — путь к scaler.json
 * @returns {Promise<{ mean: number[], scale: number[] }>}
 */
export async function loadScaler(scalerUrl) {
    const res = await fetch(scalerUrl);
    if (!res.ok) throw new Error(`Не удалось загрузить scaler: ${scalerUrl}`);
    return res.json();
}