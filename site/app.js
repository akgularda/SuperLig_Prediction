const formatNumber = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
};

const formatInteger = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US").format(value);
};

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${formatNumber(value, 1)}%`;
};

const formatDuration = (milliseconds) => `${(milliseconds / 1000).toFixed(1)}s`;

const safeText = (value) => (value ? String(value) : "--");

const createTag = (text) => {
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = text;
  return tag;
};

const renderContenders = (table) => {
  const container = document.getElementById("contenders-list");
  container.innerHTML = "";
  const top = [...table]
    .sort((a, b) => b.championship_probability - a.championship_probability)
    .slice(0, 6);

  top.forEach((team) => {
    const row = document.createElement("div");
    row.className = "contender-row";

    const left = document.createElement("div");
    left.innerHTML = `<strong>${team.team}</strong><div class="contender-bar" style="width:${Math.min(
      team.championship_probability,
      100
    )}%;"></div>`;

    const right = document.createElement("div");
    right.textContent = formatPercent(team.championship_probability);

    row.appendChild(left);
    row.appendChild(right);
    container.appendChild(row);
  });
};

const renderChampion = (table, teams) => {
  const teamEl = document.getElementById("champion-team");
  const probEl = document.getElementById("champion-prob");
  const metaEl = document.getElementById("champion-meta");

  if (!teamEl || !probEl || !metaEl || table.length === 0) {
    return;
  }

  const contenders = [...table].sort((a, b) => {
    if (b.championship_probability !== a.championship_probability) {
      return b.championship_probability - a.championship_probability;
    }
    if (b.points !== a.points) {
      return b.points - a.points;
    }
    if (b.goal_difference !== a.goal_difference) {
      return b.goal_difference - a.goal_difference;
    }
    return b.goals_for - a.goals_for;
  });

  const champion = contenders[0];
  const team = teams[champion.team] || {};
  teamEl.textContent = champion.team;
  probEl.textContent = `Championship probability ${formatPercent(
    champion.championship_probability
  )}`;

  metaEl.innerHTML = "";
  const metaItems = [
    `Manager: ${safeText(team.manager)}`,
    `Market value: EUR ${formatNumber(team.market_value_eur_m, 1)}m`,
    `Projected points: ${formatNumber(champion.points, 1)}`,
  ];
  metaItems.forEach((text) => {
    const span = document.createElement("span");
    span.textContent = text;
    metaEl.appendChild(span);
  });
};

const renderTable = (table) => {
  const tbody = document.getElementById("forecast-table");
  tbody.innerHTML = "";

  table.slice(0, 18).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.team}</td>
      <td>${formatNumber(row.points, 1)}</td>
      <td>${formatPercent(row.championship_probability)}</td>
      <td>${formatPercent(row.europe_probability)}</td>
      <td>${formatPercent(row.relegation_probability)}</td>
    `;
    tbody.appendChild(tr);
  });
};

const renderTeamCards = (table, teams) => {
  const container = document.getElementById("team-cards");
  container.innerHTML = "";
  const top = [...table].slice(0, 6);

  top.forEach((row, index) => {
    const team = teams[row.team] || {};
    const card = document.createElement("div");
    card.className = "team-card reveal";
    card.style.setProperty("--delay", `${0.05 * index}s`);

    const signings = (team.key_signings || []).slice(0, 3);
    card.innerHTML = `
      <h3>${row.team}</h3>
      <p class="team-meta">Manager: ${safeText(team.manager)}</p>
      <p class="team-meta">Market value: EUR ${formatNumber(team.market_value_eur_m, 1)}m</p>
      <p class="team-meta">Stadium: ${formatInteger(team.stadium_capacity)} seats</p>
      <p class="team-meta">Projected points: ${formatNumber(row.points, 1)}</p>
    `;

    if (signings.length) {
      const tagList = document.createElement("div");
      tagList.className = "tag-list";
      signings.forEach((player) => tagList.appendChild(createTag(player)));
      card.appendChild(tagList);
    }

    container.appendChild(card);
  });
};

const renderPlayerRadar = (table, teams) => {
  const container = document.getElementById("player-radar");
  container.innerHTML = "";
  const ranked = [...table]
    .sort((a, b) => b.championship_probability - a.championship_probability)
    .slice(0, 8);

  ranked.forEach((row, index) => {
    const team = teams[row.team] || {};
    const card = document.createElement("div");
    card.className = "player-card reveal";
    card.style.setProperty("--delay", `${0.04 * index}s`);

    card.innerHTML = `
      <h3>${row.team}</h3>
      <p class="team-meta">Signal strength: ${formatPercent(row.championship_probability)}</p>
    `;

    const tagList = document.createElement("div");
    tagList.className = "tag-list";
    (team.key_signings || []).forEach((player) => tagList.appendChild(createTag(player)));
    if (!team.key_signings || team.key_signings.length === 0) {
      tagList.appendChild(createTag("No signings listed"));
    }
    card.appendChild(tagList);

    container.appendChild(card);
  });
};

const applyMetadata = (metadata) => {
  const lastUpdated = document.getElementById("last-updated");
  const sims = document.getElementById("simulations");
  const range = document.getElementById("historical-range");

  if (metadata) {
    lastUpdated.textContent = metadata.generated_at || "--";
    sims.textContent = formatInteger(metadata.simulations);
    range.textContent = metadata.historical_range || "--";
  }
};

const renderMethodMetrics = (metadata) => {
  const matches = document.getElementById("method-matches");
  const range = document.getElementById("method-range");
  const sims = document.getElementById("method-sims");
  const teams = document.getElementById("method-teams");

  if (!metadata) {
    return;
  }

  if (matches) {
    matches.textContent = formatInteger(metadata.historical_matches);
  }
  if (range) {
    range.textContent = metadata.historical_range || "--";
  }
  if (sims) {
    sims.textContent = formatInteger(metadata.simulations);
  }
  if (teams) {
    teams.textContent = formatInteger(metadata.teams_total);
  }
};

const simState = {
  initialized: false,
  running: false,
  table: [],
  counts: {},
  done: 0,
  target: 0,
  start: 0,
};

const buildSampler = (table) => {
  const entries = table.map((row) => ({
    team: row.team,
    weight: Math.max(0, Number(row.championship_probability) || 0),
  }));
  let total = entries.reduce((sum, entry) => sum + entry.weight, 0);
  if (total <= 0) {
    entries.forEach((entry) => {
      entry.weight = 1;
    });
    total = entries.length;
  }

  const cumulative = [];
  let running = 0;
  entries.forEach((entry) => {
    running += entry.weight;
    cumulative.push({ team: entry.team, threshold: running });
  });

  return () => {
    const roll = Math.random() * total;
    for (let i = 0; i < cumulative.length; i += 1) {
      if (roll <= cumulative[i].threshold) {
        return cumulative[i].team;
      }
    }
    return cumulative[cumulative.length - 1]?.team || "--";
  };
};

const updateSimulationUI = () => {
  const iterationsEl = document.getElementById("sim-iterations");
  const leaderEl = document.getElementById("sim-leader");
  const elapsedEl = document.getElementById("sim-elapsed");
  const resultsEl = document.getElementById("sim-results");

  if (!iterationsEl || !leaderEl || !elapsedEl || !resultsEl) {
    return;
  }

  const total = simState.done;
  const elapsed = simState.start ? performance.now() - simState.start : 0;
  const iterationsText = simState.target
    ? `${formatInteger(total)} / ${formatInteger(simState.target)}`
    : formatInteger(total);
  iterationsEl.textContent = iterationsText;
  elapsedEl.textContent = formatDuration(elapsed);

  const entries = Object.entries(simState.counts).map(([team, count]) => ({
    team,
    count,
    share: total ? (count / total) * 100 : 0,
  }));
  entries.sort((a, b) => b.count - a.count);

  if (entries.length > 0) {
    leaderEl.textContent = `${entries[0].team} (${formatPercent(entries[0].share)})`;
  } else {
    leaderEl.textContent = "--";
  }

  resultsEl.innerHTML = "";
  if (!total) {
    const row = document.createElement("div");
    row.className = "sim-row";
    row.innerHTML = "<span>No runs yet</span><span>--</span><span>--</span>";
    resultsEl.appendChild(row);
    return;
  }

  entries.slice(0, 6).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "sim-row";
    row.innerHTML = `
      <span>${entry.team}</span>
      <span>${formatInteger(entry.count)}</span>
      <span>${formatPercent(entry.share)}</span>
    `;
    resultsEl.appendChild(row);
  });
};

const initLiveSimulation = (table) => {
  const runButton = document.getElementById("sim-run");
  const resetButton = document.getElementById("sim-reset");
  if (!runButton || !resetButton) {
    return;
  }

  simState.table = table || [];
  if (simState.initialized) {
    return;
  }
  simState.initialized = true;
  updateSimulationUI();

  const resetSimulation = () => {
    simState.running = false;
    simState.counts = {};
    simState.done = 0;
    simState.target = 0;
    simState.start = 0;
    runButton.disabled = false;
    resetButton.disabled = false;
    updateSimulationUI();
  };

  const runSimulation = (iterations) => {
    if (simState.running || simState.table.length === 0) {
      return;
    }
    simState.running = true;
    simState.counts = {};
    simState.done = 0;
    simState.target = iterations;
    simState.start = performance.now();
    runButton.disabled = true;
    resetButton.disabled = true;

    const sample = buildSampler(simState.table);
    const chunkSize = 250;

    const runChunk = () => {
      const remaining = simState.target - simState.done;
      const batch = Math.min(chunkSize, remaining);
      for (let i = 0; i < batch; i += 1) {
        const team = sample();
        simState.counts[team] = (simState.counts[team] || 0) + 1;
      }
      simState.done += batch;
      updateSimulationUI();

      if (simState.done < simState.target) {
        requestAnimationFrame(runChunk);
      } else {
        simState.running = false;
        runButton.disabled = false;
        resetButton.disabled = false;
        updateSimulationUI();
      }
    };

    requestAnimationFrame(runChunk);
  };

  runButton.addEventListener("click", () => {
    const iterations = Number.parseInt(runButton.dataset.iterations, 10) || 10000;
    runSimulation(iterations);
  });
  resetButton.addEventListener("click", () => {
    if (!simState.running) {
      resetSimulation();
    }
  });
};

const loadData = async () => {
  const response = await fetch(`data/latest.json?ts=${Date.now()}`);
  if (!response.ok) {
    throw new Error("Failed to load latest data");
  }
  return response.json();
};

const render = async () => {
  try {
    const payload = await loadData();
    const table = payload?.predictions?.table || [];
    const teams = payload?.teams || {};

    applyMetadata(payload?.metadata);
    renderMethodMetrics(payload?.metadata);
    renderChampion(table, teams);
    renderContenders(table);
    renderTable(table);
    renderTeamCards(table, teams);
    renderPlayerRadar(table, teams);
    initLiveSimulation(table);
  } catch (error) {
    const fallback = document.getElementById("forecast-table");
    if (fallback) {
      fallback.innerHTML = `<tr><td colspan="6">Data feed unavailable. Run scripts/build_site.py.</td></tr>`;
    }
  }
};

render();
