/**
 * progress.js — управление прогресс-баром и статусными надписями.
 */

import { dropZone, progressSec, progressBar, progressPct, progressStat, resultSec } from './dom.js';

const STATUS_LABELS = {
    queued:     'В очереди...',
    processing: 'Обработка изображения...',
    done:       'Готово',
};

export function showProgress() {
    dropZone.style.display    = 'none';
    progressSec.style.display = 'block';
    resultSec.style.display   = 'none';
    setProgress(0, 'Подготовка...');
}

export function hideProgress() {
    progressSec.style.display = 'none';
}

export function updateProgress({ status, progress }) {
    setProgress(progress, STATUS_LABELS[status] ?? status);
}

function setProgress(pct, label) {
    progressBar.style.width  = pct + '%';
    progressPct.textContent  = pct + '%';
    progressStat.textContent = label;
}