// Left sidebar: body catalog tree.
export function buildCatalog(system, onSelect) {
  const ul = document.getElementById("catalog");
  const count = document.getElementById("catalog-count");
  const filter = document.getElementById("catalog-filter");
  const items = {};

  function kindColor(b) {
    if (b.kind === "star") {
      const t = b.extra?.teff_k ?? 5772;
      if (t > 6200) return "#fdf6e3";
      if (t > 5400) return "#ffe9b8";
      if (t > 4200) return "#ffc98a";
      return "#ff9a6a";
    }
    return b.kind === "planet" ? "#7aa2e8" : "#a8c8f0";
  }

  function render() {
    ul.innerHTML = "";
    const q = filter.value.trim().toLowerCase();
    for (const b of system.bodies) {
      if (q && !b.name.toLowerCase().includes(q)) continue;
      const li = document.createElement("li");
      li.dataset.name = b.name;
      const depth = b.parent ? 1 : 0;
      li.innerHTML = `
        <span class="child-mark">${depth ? "└" : ""}</span>
        <span class="dot" style="background:${kindColor(b)}"></span>
        <span class="bname">${b.name}</span>
        <span class="bkind">${b.kind === "star" ? "恒星" : b.kind === "planet" ? "行星" : "卫星"}</span>`;
      li.style.paddingLeft = `${10 + depth * 16}px`;
      li.addEventListener("click", () => onSelect(b.name));
      ul.appendChild(li);
      items[b.name] = li;
    }
    count.textContent = system.bodies.length;
  }

  filter.addEventListener("input", render);
  render();

  return {
    render,
    select(name) {
      for (const li of Object.values(items)) li.classList.remove("selected");
      if (items[name]) items[name].classList.add("selected");
    },
  };
}
