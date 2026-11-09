'use strict';
const demo = document.querySelector('#demo');
const base = demo.dataset.base.replace(/\/?$/, '/');
const play = document.querySelector('#play');
const replay = document.querySelector('#replay');
const stages = [...document.querySelectorAll('[data-stage]')];
const captions = [
  'Draw a lasso around the target in each of four support scans.',
  'nnInteractive turns the outlines into four full-volume support masks.',
  'Reuse those supports to label new scans automatically, then refine with AutoZoom.'
];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const duration = 18000;
let time = 0, previous = 0, frame = null, running = false, activeStage = -1;
let supports = [], queries = [], dataset;
function card(record, index, support) {
  const figure = document.createElement('figure');
  figure.className = 'scan-card';
  const name = `${support ? 'Support' : 'Scan'} ${String(index + 1).padStart(2, '0')}`;
  figure.innerHTML = `<div class="scan-view"><img src="${base}${record.image}" alt="${name}: axial abdominal CT" width="${record.width}" height="${record.height}"><img class="mask" src="${base}${record.mask}" alt="" width="${record.width}" height="${record.height}">${support ? `<svg viewBox="0 0 ${record.width} ${record.height}" aria-hidden="true"><path class="lasso" pathLength="1" d="${record.lasso}"/></svg>` : ''}</div><figcaption><span>${name}</span><span class="scan-state">${support ? 'Awaiting prompt' : 'Unlabeled'}</span></figcaption>`;
  const slice = figure.querySelector('.scan-view');
  const pair = document.createElement('div'); pair.className = 'scan-pair';
  slice.replaceWith(pair); pair.append(slice);
  const surface = document.createElement('img'); surface.className = 'surface-preview'; surface.src = base + record.surface; surface.alt = '3D segmentation preview'; pair.append(surface);
  return figure;
}
function render() {
  const stage = time < 6000 ? 0 : time < 9000 ? 1 : 2;
  if (stage !== activeStage) {
    stages.forEach((button, i) => i === stage ? button.setAttribute('aria-current', 'step') : button.removeAttribute('aria-current'));
    demo.dataset.phase = stage;
    document.querySelector('#query-section').hidden = stage < 2;
    document.querySelector('#stage-caption').textContent = `Step ${stage + 1} of 3 · ${captions[stage]}`;
    activeStage = stage;
  }
  let supportCount = 0, promptCount = 0, queryCount = 0;
  supports.forEach((node, i) => {
    const draw = Math.max(0, Math.min(1, (time - i * 1400) / 950));
    const ready = time >= 6500 + i * 650;
    const active = stage === 0 ? draw < 1 && time >= i * 1400 : stage === 1 && !ready && time >= 6000 + i * 650;
    node.style.setProperty('--draw', draw);
    node.style.setProperty('--drawing', draw > 0 && !ready ? 1 : 0);
    node.classList.toggle('active', active);
    node.style.setProperty('--visible', ready ? 1 : 0);
    node.classList.toggle('ready', ready);
    node.querySelector('.scan-state').textContent = ready ? 'Support ready ✓' : stage === 1 && active ? 'Building mask…' : draw === 1 ? 'Prompt ready ✓' : draw > 0 ? 'Lasso prompt' : 'Awaiting prompt';
    promptCount += Number(draw === 1);
    supportCount += Number(ready);
  });
  queries.forEach((node, i) => {
    const ready = time >= 9500 + i * 900;
    const active = !ready && time >= 9000 + i * 900;
    node.classList.toggle('active', active);
    node.style.setProperty('--visible', ready ? 1 : 0);
    node.classList.toggle('ready', ready);
    node.querySelector('.scan-state').textContent = ready ? 'Labeled ✓' : active ? 'Labeling…' : 'Unlabeled';
    queryCount += Number(ready);
  });
  demo.style.setProperty('--transfer', time >= 6000 ? 1 : 0);
  document.querySelector('#support-count').textContent = stage === 0 ? `${promptCount} / ${supports.length} outlined` : `${supportCount} / ${supports.length} ready`;
  document.querySelector('#query-count').textContent = `${queryCount} / ${queries.length} labeled`;
  document.querySelector('#progress-fill').style.width = `${100 * time / duration}%`;
  document.querySelector('#playback-time').textContent = `00:${String(Math.floor(time / 1000)).padStart(2, '0')} / 00:18`;
}
function pause() {
  running = false;
  cancelAnimationFrame(frame);
  demo.classList.remove('running');
  play.innerHTML = time >= duration ? '↺ Play again' : '▶ Play workflow';
  play.setAttribute('aria-label', time >= duration ? 'Replay recorded workflow' : 'Play recorded workflow');
}
function tick(now) {
  time = Math.min(duration, time + now - previous);
  previous = now;
  render();
  if (time >= duration) pause();
  else frame = requestAnimationFrame(tick);
}
function start() {
  if (time >= duration) time = 0;
  if (reducedMotion.matches) { time = duration; render(); pause(); return; }
  running = true;
  previous = performance.now();
  demo.classList.add('running');
  play.textContent = 'Ⅱ Pause workflow';
  play.setAttribute('aria-label', 'Pause recorded workflow');
  frame = requestAnimationFrame(tick);
}
play.addEventListener('click', () => running ? pause() : start());
replay.addEventListener('click', () => { pause(); time = 0; render(); start(); });
stages.forEach((button, i) => button.addEventListener('click', () => { pause(); time = [5500, 8800, duration][i]; render(); pause(); }));
document.querySelector('#mask-toggle').addEventListener('click', event => {
  const hidden = demo.classList.toggle('hide-masks');
  event.currentTarget.setAttribute('aria-pressed', String(!hidden));
});
document.addEventListener('visibilitychange', () => { if (document.hidden && running) pause(); });
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) { pause(); time = duration; render(); } });
function selectOrgan(index) {
  pause(); time = reducedMotion.matches ? duration : 0;
  const selected = dataset.organs[index];
  supports = selected.supports.map((item, i) => card(item, i, true));
  queries = selected.queries.map((item, i) => card(item, i, false));
  document.querySelector('#supports').replaceChildren(...supports);
  document.querySelector('#queries').replaceChildren(...queries);
  render();
}
async function init() {
  try {
    const response = await fetch(`${base}assets/demo.json`);
    if (!response.ok) throw new Error('Example unavailable');
    dataset = await response.json();
    const picker = document.querySelector('#organ');
    dataset.organs.forEach((organ, i) => picker.add(new Option(organ.target, i)));
    picker.addEventListener('change', () => selectOrgan(Number(picker.value)));
    selectOrgan(0);
    document.querySelector('#load-status').textContent = '';
    demo.setAttribute('aria-busy', 'false');
    [play, replay, ...stages, document.querySelector('#mask-toggle'), picker].forEach(button => button.disabled = false);
    if (reducedMotion.matches) time = duration;
    render();
  } catch (error) {
    demo.setAttribute('aria-busy', 'false');
    document.querySelector('#load-status').textContent = 'The recorded images could not be loaded. Reload this page to try again.';
  }
}
init();
