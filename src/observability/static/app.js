const frame = document.querySelector("#perfetto");
const origin = location.origin;
let trace;

const ready = new Promise(resolve => {
  const ping = setInterval(() => frame.contentWindow.postMessage("PING", origin), 100);
  window.addEventListener("message", function receive(event) {
    if (event.source === frame.contentWindow && event.data === "PONG") {
      clearInterval(ping);
      window.removeEventListener("message", receive);
      resolve();
    }
  });
});

async function update() {
  trace = await (await fetch("/rl/trace.json")).json();
  const run = trace.otherData;
  const step = run.metrics.at(-1)?.step ?? 0;
  document.querySelector("#steps").textContent = `${step} / ${run.total_steps} steps`;
  document.querySelector("#progress").max = run.total_steps;
  document.querySelector("#progress").value = step;
  document.querySelector("#iteration").textContent = `第 ${run.iteration} 轮`;
  document.querySelector("#status").textContent = run.finished ? "已完成" : "训练中";
  renderLoss(run.metrics);
}

const number = new Intl.NumberFormat("en", {maximumSignificantDigits: 4});

function renderLoss(metrics) {
  const svg = document.querySelector("#loss-chart");
  if (metrics.length === 0) {
    svg.replaceChildren();
    return;
  }
  const first = metrics[0], last = metrics.at(-1);
  const values = metrics.map(point => point.loss);
  const low = Math.min(...values), high = Math.max(...values);
  const padding = Math.max((high - low) * .15, Math.abs(high) * .05, 1e-6);
  const width = svg.clientWidth;
  svg.setAttribute("viewBox", `0 0 ${width} 100`);
  const x = step => 66 + (step - first.step) / Math.max(1, last.step - first.step) * (width - 90);
  const y = value => 10 + (high + padding - value) / (high - low + 2 * padding) * 60;
  const path = metrics.map((point, index) => `${index ? "L" : "M"}${x(point.step)},${y(point.loss)}`).join(" ");
  const ticks = [high + padding, (high + low) / 2, low - padding];
  const steps = [...new Set([first.step, Math.round((first.step + last.step) / 2), last.step])];
  svg.innerHTML = `
    <defs><linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#b7a197" stop-opacity=".22"/><stop offset="1" stop-color="#b7a197" stop-opacity="0"/></linearGradient></defs>
    ${ticks.map(value => `<path class="grid" d="M66,${y(value)} H${width - 24}"/><text x="56" y="${y(value) + 4}" text-anchor="end">${number.format(value)}</text>`).join("")}
    ${steps.map(step => `<text x="${x(step)}" y="92" text-anchor="middle">Step ${step}</text>`).join("")}
    <path d="${path} L${x(last.step)},70 L${x(first.step)},70 Z" fill="url(#loss-fill)"/>
    <path class="curve" d="${path}"/>
    ${metrics.map(point => `<circle cx="${x(point.step)}" cy="${y(point.loss)}" r="3.5" tabindex="0" role="button" aria-label="Step ${point.step}, loss ${number.format(point.loss)}" data-step="${point.step}"/>`).join("")}`;
  document.querySelector("#loss-value").textContent = number.format(last.loss);
  document.querySelector("#loss-step").textContent = `STEP ${last.step}`;
  document.querySelector("#loss-detail").textContent = `Step ${last.step} · Loss ${number.format(last.loss)} · 第 ${last.iteration} 轮`;
  for (const dot of svg.querySelectorAll("circle")) {
    const point = metrics.find(value => value.step === Number(dot.dataset.step));
    dot.onmouseenter = () => {
      document.querySelector("#loss-detail").textContent = `Step ${point.step} · Loss ${number.format(point.loss)} · 第 ${point.iteration} 轮`;
    };
    dot.onkeydown = event => { if (event.key === "Enter") focusStep(point.step); };
  }
  svg.onclick = event => {
    const pointer = svg.createSVGPoint();
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    const position = pointer.matrixTransform(svg.getScreenCTM().inverse());
    const nearest = metrics.reduce((a, b) => Math.abs(x(a.step) - position.x) < Math.abs(x(b.step) - position.x) ? a : b);
    focusStep(nearest.step);
  };
}

async function focusStep(step) {
  await openTrace();
  const event = trace.traceEvents.find(event => event.name === "train.minibatch" && event.args.optimizer_step === step);
  frame.contentWindow.postMessage({perfetto: {
    timeStart: event.ts / 1e6, timeEnd: (event.ts + event.dur) / 1e6, viewPercentage: 1,
  }}, origin);
}

async function openTrace() {
  await ready;
  const buffer = new TextEncoder().encode(JSON.stringify(trace)).buffer;
  frame.contentWindow.postMessage({perfetto: {
    buffer, title: "RL framework events", keepApiOpen: true,
  }}, origin, [buffer]);
}

document.querySelector("#refresh").onclick = async () => {
  await update();
  await openTrace();
};
await update();
await openTrace();
setInterval(update, 2000);
