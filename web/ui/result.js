/**
 * result.js — отображение результата: сравнение до/после,
 * параметры коррекции, кнопка скачивания.
 */

import { imgOriginal, imgResult, paramsBar, btnDownload, resultSec } from './dom.js';
import { hideProgress } from './progress.js';

export function showOriginal(file) {
    const url = URL.createObjectURL(file);
    imgOriginal.src = url;
    imgOriginal.onload = () => URL.revokeObjectURL(url);
}

export function showResult(blob, params) {
    const url = URL.createObjectURL(blob);
    imgResult.src = url;
    imgResult.onload = () => {
        btnDownload.href     = url;
        btnDownload.download = buildDownloadName(imgOriginal);
    };
    renderParams(params);
    hideProgress();
    resultSec.style.display = 'block';
}

export function showError(msg) {
    const el = document.getElementById('errorMsg');
    el.textContent   = msg;
    el.style.display = 'block';
}

export function hideError() {
    document.getElementById('errorMsg').style.display = 'none';
}

function buildDownloadName(inputImageName) {
    if (!inputImageName) return 'enhanced.jpg';
    const dot = inputImageName.lastIndexOf('.');
    if (dot === -1) return inputImageName + '_enhanced';
    const name = inputImageName.slice(0, dot);
    const ext  = inputImageName.slice(dot);
    return `${name}_enhanced${ext}`;
}

function renderParams(params) {
    if (!params) { paramsBar.style.display = 'none'; return; }

    const fmt = v => (v >= 0 ? '+' : '') + (v * 100).toFixed(0) + '%';
    const cls = v => v > 0.05 ? 'positive' : v < -0.05 ? 'negative' : '';

    paramsBar.innerHTML = `
        <div class="param-item">
            <span class="param-name">Яркость</span>
            <span class="param-val ${cls(params.brightness)}">${fmt(params.brightness)}</span>
        </div>
        <div class="param-item">
            <span class="param-name">Контраст</span>
            <span class="param-val ${cls(params.contrast)}">${fmt(params.contrast)}</span>
        </div>
        <div class="param-item">
            <span class="param-name">Насыщенность</span>
            <span class="param-val ${cls(params.saturation)}">${fmt(params.saturation)}</span>
        </div>
    `;
}