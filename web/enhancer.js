/**
 * enhancer.js — публичный API модуля улучшения изображений.
 *
 * Использование:
 *   import { ImageEnhancer } from './enhancer.js';
 *   const enhancer = new ImageEnhancer();
 *   await enhancer.init();
 *
 *   const taskId = await enhancer.enqueue(file);
 *   enhancer.addEventListener('taskchange', e => console.log(e.detail));
 *   const blob = await enhancer.getResult(taskId);
 *
 * API:
 *   enhancer.enqueue(file | Blob | HTMLImageElement | ImageData) → Promise<taskId>
 *   enhancer.getStatus(taskId) → { status, progress, params? }
 *   enhancer.cancel(taskId)   → Promise<{ success }>
 *   enhancer.getResult(taskId) → Promise<Blob>
 *   enhancer.addEventListener('taskchange', handler)
 */

'use strict';

// Статусы задачи
export const TaskStatus = Object.freeze({
    QUEUED:     'queued',
    PROCESSING: 'processing',
    DONE:       'done',
    ERROR:      'error',
    CANCELLED:  'cancelled',
});

// Поддерживаемые MIME типы
const SUPPORTED_TYPES = new Set([
    'image/jpeg', 
    'image/png', 
    'image/bmp', 
    'image/heic', 
    'image/heif',
    'image/webp',
]);

let heic2any = null;

async function loadHeicDecoder() {
    if (heic2any) return heic2any;
    // CDN — не влияет на 10 МБ бюджет основного кода
    const mod = await import('https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js');
    heic2any = mod.default ?? mod;
    return heic2any;
}

function generateId() {
    return `task_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

async function fileToImageData(source) {
    let blob;

    if (source instanceof File || source instanceof Blob) {
        blob = source;
    } else if (source instanceof HTMLImageElement) {
        const canvas = new OffscreenCanvas(source.naturalWidth, source.naturalHeight);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(source, 0, 0);
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    } else if (source instanceof ImageData) {
        return source;
    } else {
        throw new Error('Неподдерживаемый тип источника изображения');
    }

    // HEIC → PNG
    const type = blob.type.toLowerCase();
    if (type === 'image/heic' || type === 'image/heif' ||
        blob.name?.toLowerCase().endsWith('.heic') ||
        blob.name?.toLowerCase().endsWith('.heif')) {
        const decoder = await loadHeicDecoder();
        blob = await decoder({ blob, toType: 'image/png' });
    }

    // Blob → ImageData через OffscreenCanvas
    const bmp  = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx  = canvas.getContext('2d');
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function imagedataToBlob(imageData) {
    const canvas = new OffscreenCanvas(imageData.width, imageData.height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.93 });
}

export class ImageEnhancer extends EventTarget {
    #worker       = null;
    #modelData    = null;
    #scalerData   = null;
    #tasks        = new Map();   // taskId → { status, progress, result?, error?, params? }
    #resultStore  = new Map();   // taskId → Blob (готовые результаты)

    /**
     * Инициализирует воркер и загружает модель.
     * @param {Object} options
     * @param {string} options.workerUrl  — путь к worker.js
     * @param {string} options.modelUrl   — путь к model.json (TF.js формат)
     * @param {string} options.scalerUrl  — путь к scaler.json
     */
    async init({ workerUrl = './worker.js', modelUrl = './model/model.json', scalerUrl = './model/scaler.json' } = {}) {
        // Загружаем модель только если URL передан
        if (modelUrl && scalerUrl) {
        const { loadModel, loadScaler } = await import('./model-loader.js');
        const [modelData, scalerData] = await Promise.all([
            loadModel(modelUrl),
            loadScaler(scalerUrl),
        ]);
        this.#modelData  = modelData;
        this.#scalerData = scalerData;
        }

        // Запуск воркера как module чтобы поддерживать ESM если нужно
        this.#worker = new Worker(workerUrl, { type: 'classic' });
        this.#worker.onmessage = this.#handleWorkerMessage.bind(this);
        this.#worker.onerror   = (e) => console.error('[Enhancer] Worker error:', e);

        return this;
    }

    /**
     * Ставит изображение в очередь на обработку.
     * @param {File|Blob|HTMLImageElement|ImageData} source
     * @returns {Promise<string>} taskId
     */
    async enqueue(source) {
        if (!this.#worker) throw new Error('Enhancer не инициализирован. Вызовите init() сначала.');

        const taskId = generateId();
        this.#tasks.set(taskId, { status: TaskStatus.QUEUED, progress: 0 });
        this.#dispatchChange(taskId);

        // Конвертация в ImageData (в main thread, т.к. нужен DOM/OffscreenCanvas)
        let imageData;
        try {
            imageData = await fileToImageData(source);
        } catch (err) {
            this.#setTask(taskId, { status: TaskStatus.ERROR, progress: 0, error: err.message });
            throw err;
        }

        // Проверка размера (15 Мпк)
        const pixels = imageData.width * imageData.height;
        if (pixels > 15_000_000) {
            // Масштабируем до 15 Мпк
            const scale  = Math.sqrt(15_000_000 / pixels);
            const w = Math.round(imageData.width  * scale);
            const h = Math.round(imageData.height * scale);
            const canvas = new OffscreenCanvas(w, h);
            const ctx    = canvas.getContext('2d');
            const bmp    = await createImageBitmap(imageData, { resizeWidth: w, resizeHeight: h });
            ctx.drawImage(bmp, 0, 0);
            bmp.close();
            imageData = ctx.getImageData(0, 0, w, h);
        }

        // Отправка в воркер (transferable)
        this.#worker.postMessage(
            { type: 'process', taskId, imageData, modelData: this.#modelData, scalerData: this.#scalerData },
            [imageData.data.buffer]
        );

        return taskId;
    }

    /**
     * Возвращает текущий статус задачи.
     * @param {string} taskId
     * @returns {{ status: string, progress: number, params?: object }}
     */
    getStatus(taskId) {
        const task = this.#tasks.get(taskId);
        if (!task) throw new Error(`Задача не найдена: ${taskId}`);
        return { ...task };
    }

    /**
     * Отменяет задачу.
     * @param {string} taskId
     * @returns {Promise<{ success: boolean }>}
     */
    async cancel(taskId) {
        const task = this.#tasks.get(taskId);
        if (!task) return { success: false };
        if (task.status === TaskStatus.DONE || task.status === TaskStatus.ERROR) {
            return { success: false };
        }
        this.#worker.postMessage({ type: 'cancel', taskId });
        return { success: true };
    }

    /**
     * Возвращает готовое изображение в виде Blob.
     * Если задача ещё не завершена — ждёт завершения.
     * @param {string} taskId
     * @returns {Promise<Blob>}
     */
    async getResult(taskId) {
        const task = this.#tasks.get(taskId);
        if (!task) throw new Error(`Задача не найдена: ${taskId}`);

        // Уже готово
        if (this.#resultStore.has(taskId)) {
            return this.#resultStore.get(taskId);
        }

        if (task.status === TaskStatus.ERROR)     throw new Error(task.error);
        if (task.status === TaskStatus.CANCELLED) throw new Error('Задача отменена');

        // Ждём события завершения
        return new Promise((resolve, reject) => {
            const handler = (e) => {
                if (e.detail.taskId !== taskId) return;
                this.removeEventListener('taskchange', handler);
                if (e.detail.status === TaskStatus.DONE) {
                    resolve(this.#resultStore.get(taskId));
                } 
                else if (e.detail.status === TaskStatus.ERROR) {
                    reject(new Error(e.detail.error));
                } 
                else if (e.detail.status === TaskStatus.CANCELLED) {
                    reject(new Error('Задача отменена'));
                }
            };
            this.addEventListener('taskchange', handler);
        });
    }

    #handleWorkerMessage(e) {
        const { type, taskId } = e.data;

        if (type === 'progress') {
            const { status, progress } = e.data;
            this.#setTask(taskId, { status, progress });
        }

        if (type === 'done') {
            const { result, params } = e.data;
            imagedataToBlob(result).then(blob => {
                this.#resultStore.set(taskId, blob);
                this.#setTask(taskId, { status: TaskStatus.DONE, progress: 100, params });
            });
            console.log('params:', e.data.params);
        }

        if (type === 'error') {
            this.#setTask(taskId, { status: TaskStatus.ERROR, progress: 0, error: e.data.message });
        }
    }

    #setTask(taskId, updates) {
        const current = this.#tasks.get(taskId) ?? {};
        this.#tasks.set(taskId, { ...current, ...updates });
        this.#dispatchChange(taskId);
    }

    #dispatchChange(taskId) {
        const task = this.#tasks.get(taskId);
        this.dispatchEvent(new CustomEvent('taskchange', { detail: { taskId, ...task } }));
    }
}