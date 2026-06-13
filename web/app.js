/**
 * app.js — точка входа.
 * Инициализирует enhancer, связывает UI-модули с бизнес-логикой.
 */

import { ImageEnhancer, TaskStatus } from './enhancer.js';
import { initDropZone, showDropZone, isValidFile } from './ui/dropzone.js';
import { showProgress, hideProgress, updateProgress } from './ui/progress.js';
import { showOriginal, showResult, showError, hideError } from './ui/result.js';
import { btnCancel, btnNew, resultSec, demoNotice } from './ui/dom.js';

const enhancer = new ImageEnhancer();
let currentTask = null;

async function init() {
    try {
        await enhancer.init({
            workerUrl: './worker.js',
            modelUrl:  './model/model.json',
            scalerUrl: './model/scaler.json',
        });
        demoNotice.style.display = 'none';
        console.log('[App] Модель загружена');
    } catch (err) {
        console.warn('[App] Модель не загружена:', err.message);
        try {
            await enhancer.init({
                workerUrl: './worker.js',
                modelUrl:  null,
                scalerUrl: null,
            });
        } catch (e) {
            console.error('[App] Не удалось запустить воркер:', e.message);
        }
    }
}

async function handleFile(file) {
    if (!isValidFile(file)) {
        showError('Неподдерживаемый формат. Используйте JPG, PNG, HEIC или BMP.');
        return;
    }

    hideError();
    showOriginal(file);
    showProgress();

    try {
        const taskId = await enhancer.enqueue(file);
        currentTask  = taskId;

        enhancer.addEventListener('taskchange', function handler(e) {
            if (e.detail.taskId !== taskId) return;
            updateProgress(e.detail);

            if (e.detail.status === TaskStatus.DONE) {
                enhancer.removeEventListener('taskchange', handler);
                enhancer.getResult(taskId).then(blob => showResult(blob, e.detail.params));
            }
            if (e.detail.status === TaskStatus.ERROR) {
                enhancer.removeEventListener('taskchange', handler);
                showError('Ошибка обработки: ' + (e.detail.error ?? 'неизвестная ошибка'));
                hideProgress();
            }
            if (e.detail.status === TaskStatus.CANCELLED) {
                enhancer.removeEventListener('taskchange', handler);
                hideProgress();
                showDropZone();
            }
        });
    } catch (err) {
        showError(err.message);
        hideProgress();
    }
}

btnCancel.addEventListener('click', () => {
    if (currentTask) enhancer.cancel(currentTask);
});

btnNew.addEventListener('click', () => {
    resultSec.style.display = 'none';
    showDropZone();
    currentTask = null;
});

initDropZone(handleFile);
init();