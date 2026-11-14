"""Gráfica 3D interactiva (HTML5 Canvas) con rotación por cursor y líneas de altura."""

import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_interactive_3d_scatter(df_3d: pd.DataFrame, inc_cap: float) -> None:
    """Renderiza una gráfica 3D interactiva con líneas verticales de altura para `loan_int_rate`."""
    if df_3d.empty:
        st.info("No hay registros disponibles para la combinación de filtros seleccionada en la vista 3D.")
        return

    sample = df_3d.sample(n=min(4500, len(df_3d)), random_state=42)
    z_min = float(sample["loan_int_rate"].min())
    z_max = float(sample["loan_int_rate"].max())
    y_min = 0.0
    y_max = float(max(sample["loan_amnt"].max(), 1.0))

    points = [
        {
            "x": round(float(r.person_income), 0),
            "y": round(float(r.loan_amnt), 0),
            "z": round(float(r.loan_int_rate), 2),
            "c": int(r.loan_status),
            "d": str(r.previous_loan_defaults_on_file),
            "h": str(r.person_home_ownership),
        }
        for r in sample.itertuples(index=False)
    ]

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
      * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
      body {{ margin: 0; padding: 0; background: transparent; }}
      .wrap {{
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        background: #FFFFFF;
        padding: 12px 14px 10px 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
      }}
      .toolbar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
      }}
      .btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
      .btn {{
        border: 1px solid #D1D5DB;
        background: #F9FAFB;
        color: #1F2937;
        padding: 5px 10px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .btn:hover {{ background: #EFF6FF; border-color: #93C5FD; }}
      .btn.active {{ background: #1E3A8A; color: #FFFFFF; border-color: #1E3A8A; }}
      .sel-lbl {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        font-weight: 600;
        color: #1F2937;
      }}
      .sel {{
        border: 1px solid #D1D5DB;
        background: #F9FAFB;
        color: #1F2937;
        padding: 4px 8px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
      }}
      .sel:focus {{ outline: none; border-color: #1E3A8A; }}
      .legend {{ display: flex; gap: 12px; align-items: center; font-size: 12px; color: #374151; }}
      .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
      .canvas-box {{ position: relative; width: 100%; height: 520px; cursor: grab; }}
      .canvas-box:active {{ cursor: grabbing; }}
      canvas {{ width: 100%; height: 100%; display: block; border-radius: 8px; background: #FFFFFF; }}
      .tooltip {{
        position: absolute;
        pointer-events: none;
        display: none;
        background: rgba(15, 23, 42, 0.92);
        color: #F8FAFC;
        padding: 7px 10px;
        border-radius: 7px;
        font-size: 12px;
        line-height: 1.45;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10;
      }}
      .hint {{ font-size: 11.5px; color: #6B7280; margin-top: 6px; text-align: center; }}
    </style>
    </head>
    <body>
      <div class="wrap">
        <div class="toolbar">
          <div class="btn-group">
            <button id="mode-cursor" class="btn active">🖱️ Rotar con Cursor</button>
            <button id="mode-drag" class="btn">✋ Arrastrar Libre</button>
            <button id="mode-auto" class="btn">🔄 Auto-Animación</button>
            <button id="btn-stems" class="btn active">📏 Líneas Verticales</button>
            <button id="btn-reset" class="btn">↺ Reset</button>
          </div>
          <div class="btn-group">
            <button id="cnt-400" class="btn active">400 pts</button>
            <button id="cnt-800" class="btn">800 pts</button>
            <button id="cnt-1600" class="btn">1,600 pts</button>
          </div>
          <div class="btn-group">
            <button id="flt-all" class="btn active">Todos</button>
            <button id="flt-1" class="btn">Aprobado (1)</button>
            <button id="flt-0" class="btn">Denegado (0)</button>
            <label class="sel-lbl">
              Defaults:
              <select id="flt-defaults" class="sel">
                <option value="all">Todos</option>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </label>
            <label class="sel-lbl">
              Vivienda:
              <select id="flt-home" class="sel">
                <option value="all">Todas</option>
                <option value="RENT">RENT</option>
                <option value="OWN">OWN</option>
                <option value="MORTGAGE">MORTGAGE</option>
                <option value="OTHER">OTHER</option>
              </select>
            </label>
          </div>
          <div class="legend">
            <span><span class="dot" style="background:#DC2626;"></span>Denegado (0)</span>
            <span><span class="dot" style="background:#059669;"></span>Aprobado (1)</span>
            <span id="vis-cnt" style="font-weight:600;color:#475569;"></span>
          </div>
        </div>
        <div class="canvas-box" id="box">
          <canvas id="c3d"></canvas>
          <div class="tooltip" id="tip"></div>
        </div>
        <div class="hint">
          Cada línea vertical conecta el punto con el piso para resaltar la altura de <b>loan_int_rate (%)</b>. Mueve el cursor para rotar, arrastra para girar 360° y usa la rueda para zoom.
        </div>
      </div>

      <script>
        const POINTS = {json.dumps(points)};
        const X_MAX = {round(inc_cap, 2)};
        const Y_MIN = {y_min}, Y_MAX = {y_max};
        const Z_MIN = {z_min}, Z_MAX = {z_max};

        const box = document.getElementById("box");
        const canvas = document.getElementById("c3d");
        const ctx = canvas.getContext("2d");
        const tip = document.getElementById("tip");

        let W = 900, H = 520, dpr = window.devicePixelRatio || 1;
        function resize() {{
          const rect = box.getBoundingClientRect();
          W = Math.max(320, rect.width);
          H = rect.height;
          canvas.width = W * dpr;
          canvas.height = H * dpr;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}
        resize();
        window.addEventListener("resize", resize);

        let mode = "cursor";
        let classFilter = "all";
        let defaultFilter = "all";
        let homeFilter = "all";
        let showStems = true;
        let maxPoints = 400;
        let baseYaw = -0.58, basePitch = 0.34;
        let yaw = baseYaw, pitch = basePitch;
        let targetYaw = baseYaw, targetPitch = basePitch;
        let zoom = 0.98, targetZoom = 0.98;
        let dragging = false, lastX = 0, lastY = 0;
        let mouseX = -9999, mouseY = -9999;

        function setMode(m) {{
          mode = m;
          ["cursor", "drag", "auto"].forEach(k => {{
            document.getElementById("mode-" + k).classList.toggle("active", k === m);
          }});
          if (m === "drag") {{ targetYaw = yaw; targetPitch = pitch; }}
        }}
        function setFilter(f) {{
          classFilter = f;
          ["all", "1", "0"].forEach(k => {{
            document.getElementById("flt-" + k).classList.toggle("active", k === f);
          }});
        }}
        function setCount(n) {{
          maxPoints = n;
          [400, 800, 1600].forEach(k => {{
            document.getElementById("cnt-" + k).classList.toggle("active", k === n);
          }});
        }}

        document.getElementById("mode-cursor").onclick = () => setMode("cursor");
        document.getElementById("mode-drag").onclick = () => setMode("drag");
        document.getElementById("mode-auto").onclick = () => setMode("auto");
        document.getElementById("btn-stems").onclick = () => {{
          showStems = !showStems;
          document.getElementById("btn-stems").classList.toggle("active", showStems);
        }};
        document.getElementById("btn-reset").onclick = () => {{
          baseYaw = -0.58; basePitch = 0.34;
          targetYaw = baseYaw; targetPitch = basePitch;
          targetZoom = 0.98;
        }};
        document.getElementById("cnt-400").onclick = () => setCount(400);
        document.getElementById("cnt-800").onclick = () => setCount(800);
        document.getElementById("cnt-1600").onclick = () => setCount(1600);
        document.getElementById("flt-all").onclick = () => setFilter("all");
        document.getElementById("flt-1").onclick = () => setFilter("1");
        document.getElementById("flt-0").onclick = () => setFilter("0");
        document.getElementById("flt-defaults").onchange = (e) => {{ defaultFilter = e.target.value; }};
        document.getElementById("flt-home").onchange = (e) => {{ homeFilter = e.target.value; }};

        canvas.addEventListener("mousedown", (e) => {{
          dragging = true;
          lastX = e.clientX;
          lastY = e.clientY;
        }});
        window.addEventListener("mouseup", () => {{ dragging = false; }});
        window.addEventListener("mousemove", (e) => {{
          const rect = canvas.getBoundingClientRect();
          mouseX = e.clientX - rect.left;
          mouseY = e.clientY - rect.top;
          if (dragging) {{
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            baseYaw += dx * 0.008;
            basePitch = Math.max(-1.35, Math.min(1.35, basePitch + dy * 0.008));
            targetYaw = baseYaw;
            targetPitch = basePitch;
            return;
          }}
          if (mode === "cursor" && mouseX >= 0 && mouseX <= rect.width && mouseY >= 0 && mouseY <= rect.height) {{
            const nx = (mouseX / rect.width - 0.5) * 2;
            const ny = (mouseY / rect.height - 0.5) * 2;
            targetYaw = baseYaw + nx * 0.95;
            targetPitch = Math.max(-1.35, Math.min(1.35, basePitch + ny * 0.65));
          }}
        }});
        canvas.addEventListener("mouseleave", () => {{
          mouseX = -9999; mouseY = -9999; tip.style.display = "none";
          if (mode === "cursor" && !dragging) {{
            targetYaw = baseYaw; targetPitch = basePitch;
          }}
        }});
        canvas.addEventListener("wheel", (e) => {{
          e.preventDefault();
          targetZoom = Math.max(0.55, Math.min(2.6, targetZoom * (e.deltaY < 0 ? 1.09 : 0.92)));
        }}, {{ passive: false }});

        function project(nx, ny, nz) {{
          const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
          const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
          const x1 = nx * cosY - ny * sinY;
          const y1 = nx * sinY + ny * cosY;
          const z1 = nz;
          const y2 = y1 * cosP - z1 * sinP;
          const z2 = y1 * sinP + z1 * cosP;
          const persp = 3.6 / (3.6 - y2);
          const scale = zoom * (Math.min(W, H) * 0.33) * persp;
          return {{ sx: W * 0.5 + x1 * scale, sy: H * 0.51 - z2 * scale, depth: y2, persp }};
        }}

        const NORM_PTS = POINTS.map(p => ({{
          nx: (p.x / X_MAX) * 2 - 1,
          ny: ((p.y - Y_MIN) / (Y_MAX - Y_MIN)) * 2 - 1,
          nz: ((p.z - Z_MIN) / Math.max(Z_MAX - Z_MIN, 0.01)) * 2 - 1,
          raw: p
        }}));

        const CUBE_VERTS = [
          [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
        ];
        const CUBE_EDGES = [
          [0,1],[1,2],[2,3],[3,0],
          [4,5],[5,6],[6,7],[7,4],
          [0,4],[1,5],[2,6],[3,7]
        ];

        function drawTickLabel(px, py, txt, dx, dy, align) {{
          ctx.fillStyle = "#334155";
          ctx.font = "500 10.5px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.textAlign = align || "center";
          ctx.fillText(txt, px + dx, py + dy);
        }}

        function draw() {{
          if (mode === "auto" && !dragging) {{
            baseYaw += 0.0055;
            targetYaw = baseYaw;
          }}
          yaw += (targetYaw - yaw) * 0.10;
          pitch += (targetPitch - pitch) * 0.10;
          zoom += (targetZoom - zoom) * 0.14;

          ctx.clearRect(0, 0, W, H);

          ctx.strokeStyle = "#CBD5E1";
          ctx.lineWidth = 0.75;
          for (let t = -1; t <= 1.001; t += 0.5) {{
            const a = project(t, -1, -1), b = project(t, 1, -1);
            ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
            const c = project(-1, t, -1), d = project(1, t, -1);
            ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(d.sx, d.sy); ctx.stroke();
          }}

          const pv = CUBE_VERTS.map(v => project(v[0], v[1], v[2]));
          ctx.strokeStyle = "#334155";
          ctx.lineWidth = 1.15;
          CUBE_EDGES.forEach(([i, j]) => {{
            ctx.beginPath();
            ctx.moveTo(pv[i].sx, pv[i].sy);
            ctx.lineTo(pv[j].sx, pv[j].sy);
            ctx.stroke();
          }});

          for (let i = 0; i <= 4; i++) {{
            const f = i / 4;
            const n = f * 2 - 1;

            const tx = project(n, -1, -1);
            ctx.fillStyle = "#334155";
            ctx.fillRect(tx.sx - 1, tx.sy - 2, 2, 5);
            drawTickLabel(tx.sx, tx.sy, "$" + Math.round((f * X_MAX) / 1000) + "k", 0, 14, "center");

            const ty = project(1, n, -1);
            ctx.fillRect(ty.sx - 2, ty.sy - 1, 5, 2);
            drawTickLabel(ty.sx, ty.sy, "$" + Math.round((Y_MIN + f * (Y_MAX - Y_MIN)) / 1000) + "k", 14, 4, "left");

            const tz = project(-1, -1, n);
            ctx.fillRect(tz.sx - 3, tz.sy - 1, 5, 2);
            drawTickLabel(tz.sx, tz.sy, (Z_MIN + f * (Z_MAX - Z_MIN)).toFixed(1) + "%", -8, 4, "right");
          }}

          ctx.font = "700 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.textAlign = "center";
          const midX = project(0, -1, -1);
          ctx.fillStyle = "#0F172A";
          ctx.fillText("Ingreso Anual (USD)", midX.sx, midX.sy + 32);

          const midY = project(1, 0, -1);
          ctx.textAlign = "left";
          ctx.fillText("Monto Préstamo (USD)", midY.sx + 48, midY.sy + 4);

          const midZ = project(-1, -1, 0);
          ctx.save();
          ctx.translate(midZ.sx - 56, midZ.sy);
          ctx.rotate(-Math.PI / 2);
          ctx.textAlign = "center";
          ctx.fillStyle = "#6D28D9";
          ctx.fillText("Tasa de Interés (% — loan_int_rate)", 0, 0);
          ctx.restore();

          const projected = [];
          let totalMatched = 0;
          for (let i = 0; i < NORM_PTS.length; i++) {{
            const item = NORM_PTS[i];
            if (classFilter !== "all" && String(item.raw.c) !== classFilter) continue;
            if (defaultFilter !== "all" && item.raw.d !== defaultFilter) continue;
            if (homeFilter !== "all" && item.raw.h !== homeFilter) continue;
            totalMatched++;
            if (projected.length >= maxPoints) continue;
            const top = project(item.nx, item.ny, item.nz);
            const base = showStems ? project(item.nx, item.ny, -1) : null;
            projected.push({{ top, base, raw: item.raw }});
          }}
          document.getElementById("vis-cnt").textContent = "(" + projected.length + " / " + totalMatched + " pts)";
          projected.sort((a, b) => a.top.depth - b.top.depth);

          let bestDist = 11, bestPt = null;
          for (let i = 0; i < projected.length; i++) {{
            const pt = projected[i];
            const top = pt.top;
            const alpha = Math.max(0.32, Math.min(0.95, 0.55 + top.depth * 0.22));
            const r = Math.max(2.6, Math.min(4.8, 3.2 * top.persp));

            if (showStems && pt.base) {{
              ctx.strokeStyle = pt.raw.c === 1
                ? "rgba(5, 150, 105, " + (alpha * 0.58).toFixed(2) + ")"
                : "rgba(220, 38, 38, " + (alpha * 0.58).toFixed(2) + ")";
              ctx.lineWidth = 1.1;
              ctx.beginPath();
              ctx.moveTo(pt.base.sx, pt.base.sy);
              ctx.lineTo(top.sx, top.sy);
              ctx.stroke();
            }}

            ctx.fillStyle = pt.raw.c === 1
              ? "rgba(5, 150, 105, " + alpha.toFixed(2) + ")"
              : "rgba(220, 38, 38, " + alpha.toFixed(2) + ")";
            ctx.beginPath();
            ctx.arc(top.sx, top.sy, r, 0, Math.PI * 2);
            ctx.fill();

            const dx = top.sx - mouseX, dy = top.sy - mouseY;
            const d = Math.hypot(dx, dy);
            if (d < bestDist) {{ bestDist = d; bestPt = pt; }}
          }}

          if (bestPt) {{
            const top = bestPt.top;
            if (showStems && bestPt.base) {{
              ctx.strokeStyle = "#0F172A";
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(bestPt.base.sx, bestPt.base.sy);
              ctx.lineTo(top.sx, top.sy);
              ctx.stroke();
            }}
            ctx.strokeStyle = "#0F172A";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(top.sx, top.sy, 6.5, 0, Math.PI * 2);
            ctx.stroke();

            tip.style.display = "block";
            tip.style.left = Math.min(W - 190, Math.max(8, top.sx + 12)) + "px";
            tip.style.top = Math.min(H - 95, Math.max(8, top.sy - 14)) + "px";
            const statusTxt = bestPt.raw.c === 1
              ? "<span style='color:#34D399;font-weight:700;'>Aprobado (1)</span>"
              : "<span style='color:#F87171;font-weight:700;'>Denegado (0)</span>";
            tip.innerHTML =
              statusTxt + "<br>" +
              "Ingreso: <b>$" + bestPt.raw.x.toLocaleString() + "</b><br>" +
              "Préstamo: <b>$" + bestPt.raw.y.toLocaleString() + "</b><br>" +
              "Tasa Interés: <b>" + bestPt.raw.z.toFixed(2) + "%</b><br>" +
              "Vivienda: <b>" + bestPt.raw.h + "</b> | Mora previa: <b>" + bestPt.raw.d + "</b>";
          }} else {{
            tip.style.display = "none";
          }}

          requestAnimationFrame(draw);
        }}
        requestAnimationFrame(draw);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=645)
