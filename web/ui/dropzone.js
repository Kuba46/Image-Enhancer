/**
 * dropzone.js — drag & drop, валидация формата, сброс зоны загрузки.
 */

import { dropZone, fileInput } from './dom.js';

const ALLOWED_MIME = new Set([
    'image/jpeg', 'image/png', 'image/bmp',
    'image/heic', 'image/heif', 'image/webp',
]);

const ALLOWED_EXT = new Set([
    'jpg', 'jpeg', 'png', 'bmp', 'heic', 'heif',
]);

/**
 * Инициализирует drag & drop и выбор файла.
 * @param {function(File): void} onFile — колбэк при получении файла
 */
export function initDropZone(onFile) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) onFile(file);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) onFile(fileInput.files[0]);
    });
}

/**
 * Показывает зону загрузки, сбрасывает input.
 */
export function showDropZone() {
    dropZone.style.display = '';
    fileInput.value = '';
}

/**
 * Проверяет, поддерживается ли формат файла.
 * @param {File} file
 * @returns {boolean}
 */
export function isValidFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    return ALLOWED_MIME.has(file.type) || ALLOWED_EXT.has(ext);
}