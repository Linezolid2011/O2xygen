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
    return "#7aa2e8";                      // flat world: one non-star kind (ADR-0016)
  }

  function render() {
    ul.innerHTML = "";
    for (const k of Object.keys(items)) delete items[k];   // drop stale detached rows
    const q = filter.value.trim().toLowerCase();
    for (const b of system.bodies) {
      if (q && !b.name.toLowerCase().includes(q)) continue;
      const li = document.createElement("li");
      li.dataset.name = b.name;
      const isOrigin = system.origin === b.name;
      // names are user-supplied and persisted: textContent, never innerHTML
      const mark = document.createElement("span");
      mark.className = "child-mark";
      mark.textContent = isOrigin ? "◎" : "";
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = kindColor(b);
      const bname = document.createElement("span");
      bname.className = "bname";
      bname.textContent = b.name;
      const bkind = document.createElement("span");
      bkind.className = "bkind";
      bkind.textContent = b.kind === "star" ? "恒星" : "天体";
      li.append(mark, dot, bname, bkind);
      li.style.paddingLeft = "10px";
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
