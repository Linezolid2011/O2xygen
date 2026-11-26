// Left sidebar: body catalog tree.
import { bodyHex } from "./color.js";

export function buildCatalog(system, onSelect) {
  const ul = document.getElementById("catalog");
  const count = document.getElementById("catalog-count");
  const filter = document.getElementById("catalog-filter");
  const items = {};
  let selectedName = null;

  function kindColor(b) {
    return bodyHex(b);   // the same source as the sphere and its trail
  }

  function render() {
    ul.innerHTML = "";
    for (const k of Object.keys(items)) delete items[k];   // drop stale detached rows
    const q = filter.value.trim().toLowerCase();
    for (const b of system.bodies) {
      if (q && !b.name.toLowerCase().includes(q)) continue;
      const li = document.createElement("li");
      li.dataset.name = b.name;
      if (b.name === selectedName) li.classList.add("selected");
      // no privileged origin body (ADR-0022): the mark slot stays empty;
      // names are user-supplied and persisted: textContent, never innerHTML
      const mark = document.createElement("span");
      mark.className = "child-mark";
      mark.textContent = "";
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
      selectedName = name;
      for (const li of Object.values(items)) li.classList.remove("selected");
      if (items[name]) items[name].classList.add("selected");
    },
  };
}
