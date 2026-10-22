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
