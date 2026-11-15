// uPlot wrapper: temperature curve with current-time marker.
const AXES_STYLE = {
  stroke: "#6b7186", grid: { stroke: "#232837", width: 1 },
  ticks: { stroke: "#2a2f3d", width: 1 }, font: "10px sans-serif",
};
const SERIES_COLOR = "#e8967a";

export function tempChart(el, dense, opts = {}) {
  const data = [
    dense.map((r) => r.d),
    dense.map((r) => r.T),
  ];
  let markerX = opts.markerDays ?? 0;
  let follow = opts.follow ?? false;

  const u = new uPlot({
    width: Math.max(50, el.clientWidth - 4),   // hidden tab => clientWidth 0
    height: 190,
    legend: { show: false },
    cursor: { show: false, drag: { x: false, y: false } },
    scales: { x: { time: false } },
    axes: [
      { ...AXES_STYLE, label: "距初始时间 (天)", labelSize: 18, size: 26 },
      { ...AXES_STYLE, label: "温度 (K)", labelSize: 18, size: 36, labelGap: 6 },
    ],
    series: [
      {},
      { stroke: SERIES_COLOR, width: 1.6, points: { size: 3, fill: SERIES_COLOR } },
    ],
    hooks: {
      draw: [
        (u2) => {
          const ctx = u2.ctx;
          const x = u2.valToPos(markerX, "x", true);
          if (x >= u2.bbox.left && x <= u2.bbox.left + u2.bbox.width) {
            ctx.save();
            ctx.strokeStyle = "#d8dbe4";
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, u2.bbox.top);
            ctx.lineTo(x, u2.bbox.top + u2.bbox.height);
            ctx.stroke();
            ctx.restore();
          }
        },
      ],
    },
  }, data, el);

  function setMarker(days) {
    markerX = days;
    if (follow) {
      // keep the current time inside the visible window
      const win = u.scales.x.max - u.scales.x.min;
      if (days < u.scales.x.min || days > u.scales.x.max) {
        u.setScale("x", { min: Math.max(0, days - win * 0.1), max: Math.max(0, days - win * 0.1) + win });
      }
    }
    u.redraw();
  }
  function setFollow(on) { follow = on; }
  function setSize() {
    u.setSize({ width: Math.max(50, el.clientWidth - 4), height: 190 });
  }
  return { u, setMarker, setFollow, setSize };
}
