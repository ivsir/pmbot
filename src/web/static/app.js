/* ── PM Arbitrage Dashboard — Frontend Logic ── */

// ── State ──

const state = {
  connected: false,
  activeTab: 'overview',
  snapshot: null,
  events: [],
  equityCurve: [],
  spreadHistory: [],
};

// ── Overview Animation State ──

const overviewState = {
  dotGridInitialized: false,
  currentPortfolioValue: 0,
  currentWalletBalance: 0,
  currentPnl: 0,
};

// ── Chart.js Global Config ──

Chart.defaults.color = '#a1a1a1';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation = false;

const charts = {};

// ── Formatters ──

function fmtCurrency(n) {
  if (n == null) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  return '$' + n.toFixed(2);
}

function fmtPnl(n) {
  if (n == null) return '—';
  const prefix = n >= 0 ? '+' : '';
  return prefix + fmtCurrency(n);
}

function fmtPct(n, decimals = 1) {
  if (n == null) return '—';
  return (n * 100).toFixed(decimals) + '%';
}

function fmtPctRaw(n, decimals = 1) {
  if (n == null) return '—';
  return n.toFixed(decimals) + '%';
}

function fmtNum(n, decimals = 2) {
  if (n == null) return '—';
  return n.toFixed(decimals);
}

function fmtTime(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtTimeShort(ms) {
  if (!ms) return '';
  const d = new Date(ms);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function relTime(ms) {
  if (!ms) return '';
  const diff = Date.now() - ms;
  if (diff < 1000) return 'just now';
  if (diff < 60000) return Math.floor(diff / 1000) + 's ago';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  return Math.floor(diff / 3600000) + 'h ago';
}

function pnlClass(n) {
  if (n > 0) return 'text-green';
  if (n < 0) return 'text-red';
  return 'text-muted';
}

function pnlPill(n) {
  if (n > 0) return 'pill--green';
  if (n < 0) return 'pill--red';
  return 'pill--gray';
}

// ── Event type display config ──

const EVENT_CONFIG = {
  price_tick:       { label: 'PRICE TICK',    pill: 'pill--gray' },
  orderbook_update: { label: 'ORDERBOOK',     pill: 'pill--gray' },
  oracle_price:     { label: 'ORACLE',        pill: 'pill--gray' },
  spread_detected:  { label: 'SPREAD',        pill: 'pill--blue' },
  latency_arb:      { label: 'LATENCY ARB',   pill: 'pill--blue' },
  liquidity_update: { label: 'LIQUIDITY',     pill: 'pill--gray' },
  research_signal:  { label: 'RESEARCH',      pill: 'pill--blue' },
  alpha_signal:     { label: 'ALPHA',         pill: 'pill--amber' },
  trade_signal:     { label: 'TRADE SIGNAL',  pill: 'pill--green' },
  order_placed:     { label: 'ORDER',         pill: 'pill--amber' },
  order_filled:     { label: 'FILLED',        pill: 'pill--green' },
  position_update:  { label: 'POSITION',      pill: 'pill--blue' },
  risk_alert:       { label: 'RISK ALERT',    pill: 'pill--red' },
  system_health:    { label: 'SYSTEM',        pill: 'pill--gray' },
};


// ── WebSocket ──

class DashboardWS {
  constructor() {
    this.ws = null;
    this.reconnectDelay = 1000;
    this.connect();
  }

  connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    this.ws.onopen = () => {
      state.connected = true;
      this.reconnectDelay = 1000;
      updateConnectionUI();
    };

    this.ws.onclose = () => {
      state.connected = false;
      updateConnectionUI();
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    };

    this.ws.onerror = () => {};

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch (e) {}
    };
  }

  handleMessage(msg) {
    if (msg.type === 'snapshot') {
      state.snapshot = msg.data;
      this.trackEquity(msg.data);
      this.trackSpreads(msg.data);
      renderActiveTab();
    } else if (msg.type === 'event') {
      state.events.unshift(msg);
      if (state.events.length > 200) state.events.length = 200;
      this.handleEvent(msg);
    }
  }

  trackEquity(snap) {
    // Track portfolio value (wallet + positions) in live mode, equity otherwise
    const val = snap.portfolio_value_usd || (snap.portfolio ? snap.portfolio.current_equity : null);
    if (val == null) return;
    const point = { t: snap.timestamp_ms, v: val };
    state.equityCurve.push(point);
    if (state.equityCurve.length > 3600) state.equityCurve.shift();
  }

  trackSpreads(snap) {
    if (!snap.recent_spreads) return;
    state.spreadHistory = snap.recent_spreads.slice(0, 100);
  }

  handleEvent(msg) {
    const type = msg.event_type;
    if (type === 'trade_signal' && msg.data) {
      const action = msg.data.action || 'SIGNAL';
      showToast(action === 'TRADE' ? 'success' : 'info', `${action}: ${msg.data.market_id || 'signal'}`);
    } else if (type === 'risk_alert' && msg.data) {
      showToast('warning', msg.data.message || 'Risk alert triggered');
    } else if (type === 'order_filled' && msg.data) {
      showToast('success', `Order filled: ${msg.data.position_id || ''}`);
    }

    // Refresh event stream if on live tab
    if (state.activeTab === 'live') {
      renderEventStream();
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}


// ── Tab Router ──

function showTab(name) {
  state.activeTab = name;

  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

  const tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');

  const nav = document.querySelector(`.nav-item[data-tab="${name}"]`);
  if (nav) nav.classList.add('active');

  renderActiveTab();
}

function renderActiveTab() {
  const snap = state.snapshot;
  if (!snap) return;

  try {
    switch (state.activeTab) {
      case 'overview':    renderOverview(snap); break;
      case 'live':        renderLive(snap); break;
      case 'positions':   renderPositions(snap); break;
      case 'risk':        renderRisk(snap); break;
      case 'performance': renderPerformance(snap); break;
    }
    renderSystemStatus(snap);
  } catch (e) {
    console.warn('Render error:', e);
  }
}


// ── Connection UI ──

function updateConnectionUI() {
  const dot = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  if (state.connected) {
    dot.className = 'status-dot status-dot--connected';
    label.textContent = 'Connected';
  } else {
    dot.className = 'status-dot status-dot--disconnected';
    label.textContent = 'Reconnecting…';
  }
}

function renderSystemStatus(snap) {
  const dot = document.getElementById('system-dot');
  const label = document.getElementById('system-label');
  const detail = document.getElementById('system-detail');

  const halted = snap.portfolio?.halted;
  const healthy = snap.platform?.all_healthy;

  if (snap.standby) {
    dot.className = 'status-dot';
    dot.style.background = 'var(--accent-amber)';
    label.textContent = 'Standby';
    detail.textContent = 'Waiting for funds — $' + (snap.wallet_balance_usd || 0).toFixed(2);
  } else if (halted) {
    dot.className = 'status-dot status-dot--disconnected';
    label.textContent = 'Halted';
    detail.textContent = 'Circuit breaker active';
  } else if (snap.live_mode) {
    dot.className = 'status-dot status-dot--connected';
    label.textContent = 'Live';
    detail.textContent = `$${(snap.wallet_balance_usd || 0).toFixed(2)} USDC | ${snap.portfolio?.active_positions || 0} pos`;
  } else if (healthy) {
    dot.className = 'status-dot status-dot--connected';
    label.textContent = 'Running';
    detail.textContent = `${snap.portfolio?.active_positions || 0} positions open`;
  } else {
    dot.className = 'status-dot';
    dot.style.background = 'var(--accent-amber)';
    label.textContent = 'Degraded';
    detail.textContent = 'Platform issues detected';
  }
}


// ── Overview Tab ──

function initDotGrid() {
  if (overviewState.dotGridInitialized) return;

  const container = document.getElementById('overview-dot-grid');
  if (!container) return;

  const spacing = 32;
  const rect = container.getBoundingClientRect();
  const cols = Math.floor(rect.width / spacing) || 1;
  const rows = Math.floor(rect.height / spacing) || 1;

  container.style.gridTemplateColumns = `repeat(${cols}, ${spacing}px)`;
  container.style.gridTemplateRows = `repeat(${rows}, ${spacing}px)`;

  const totalDots = cols * rows;
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < totalDots; i++) {
    const dot = document.createElement('div');
    dot.className = 'overview-dot';
    fragment.appendChild(dot);
  }
  container.appendChild(fragment);

  overviewState.dotGridInitialized = true;
  startDotPulse();
}

function startDotPulse() {
  if (!window.Motion) return;
  const { animate, stagger } = Motion;

  function pulse() {
    animate(
      '.overview-dot',
      { opacity: [0.08, 0.35, 0.08], scale: [1, 1.8, 1] },
      { duration: 4, delay: stagger(0.015, { from: 'center' }), ease: 'ease-in-out' }
    ).then(() => pulse());
  }

  pulse();
}

function animateNumber(elementId, fromValue, toValue, duration, formatter) {
  const el = document.getElementById(elementId);
  if (!el) return toValue;

  if (window.Motion && fromValue !== toValue) {
    Motion.animate(el, { scale: [1, 1.04, 1] }, { duration: 0.3, ease: 'ease-out' });
  }

  const animKey = elementId + '_raf';
  if (overviewState[animKey]) cancelAnimationFrame(overviewState[animKey]);

  const startTime = performance.now();
  const diff = toValue - fromValue;

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = fromValue + diff * eased;
    el.textContent = formatter(current);
    if (progress < 1) {
      overviewState[animKey] = requestAnimationFrame(step);
    }
  }

  overviewState[animKey] = requestAnimationFrame(step);
  return toValue;
}

function renderOverview(snap) {
  const ls = snap.live_stats || {};

  initDotGrid();

  // Mode banner
  const banner = document.getElementById('mode-banner');
  if (banner) {
    if (snap.standby) {
      banner.className = 'mode-banner mode-banner--standby';
      banner.textContent = 'STANDBY — No funds in wallet';
      banner.style.display = 'inline-flex';
    } else if (snap.live_mode) {
      banner.className = 'mode-banner mode-banner--live';
      banner.textContent = 'LIVE TRADING';
      banner.style.display = 'inline-flex';
    } else {
      banner.className = 'mode-banner mode-banner--paper';
      banner.textContent = 'PAPER TRADING';
      banner.style.display = 'inline-flex';
    }
  }

  // Portfolio Value
  const newPortfolio = snap.portfolio_value_usd || snap.wallet_balance_usd || 0;
  overviewState.currentPortfolioValue = animateNumber(
    'overview-portfolio-value', overviewState.currentPortfolioValue, newPortfolio, 600,
    (v) => '$' + v.toFixed(2)
  );

  // Wallet Balance
  const newWallet = snap.wallet_balance_usd || 0;
  overviewState.currentWalletBalance = animateNumber(
    'overview-wallet-value', overviewState.currentWalletBalance, newWallet, 600,
    (v) => '$' + v.toFixed(2)
  );

  // P&L
  const newPnl = ls.total_pnl != null ? ls.total_pnl : (snap.portfolio?.daily_pnl || 0);
  overviewState.currentPnl = animateNumber(
    'overview-pnl-value', overviewState.currentPnl, newPnl, 600,
    (v) => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2)
  );

  const pnlEl = document.getElementById('overview-pnl-value');
  if (pnlEl) {
    pnlEl.classList.remove('pnl-positive', 'pnl-negative', 'pnl-zero');
    if (newPnl > 0) pnlEl.classList.add('pnl-positive');
    else if (newPnl < 0) pnlEl.classList.add('pnl-negative');
    else pnlEl.classList.add('pnl-zero');
  }

  // Win rate + W/L
  const wins = ls.wins || 0;
  const losses = ls.losses || 0;
  setText('overview-wl', wins + ' / ' + losses);
  const wrEl = document.getElementById('overview-win-rate');
  if (wrEl) {
    if (ls.win_rate != null) {
      wrEl.textContent = (ls.win_rate * 100).toFixed(1) + '%';
      wrEl.style.color = ls.win_rate >= 0.6 ? 'var(--accent-green)' : ls.win_rate >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)';
    } else {
      wrEl.textContent = '--';
    }
  }

  // Bayesian warmup
  const bay = snap.bayesian || {};
  const tc = bay.trade_count || 0;
  const target = bay.warmup_target || 50;
  const pct = Math.min(100, Math.round(tc / target * 100));
  setText('warmup-pct', pct + '%');
  const fillBar = document.getElementById('warmup-fill');
  if (fillBar) fillBar.style.width = pct + '%';

  // Position cards — active on top, then resolved
  renderOverviewPositions(snap.active_positions || [], snap.closed_positions || []);
}

function renderOverviewPositions(active, closed) {
  const container = document.getElementById('overview-positions');
  if (!container) return;

  // Merge: active first (newest on top), then closed (newest on top)
  const sortByTime = (a, b) => (b.filled_at_ms || b.created_at_ms) - (a.filled_at_ms || a.created_at_ms);
  const activeSorted = [...active].sort(sortByTime);
  const closedSorted = [...closed].sort(sortByTime);
  const all = [...activeSorted, ...closedSorted];

  // Scroll fade listener — attach once
  if (!container.dataset.scrollBound) {
    container.addEventListener('scroll', () => {
      container.classList.toggle('is-scrolled', container.scrollTop > 8);
    }, { passive: true });
    container.dataset.scrollBound = '1';
  }

  if (all.length === 0) {
    if (container.dataset.posFingerprint) { container.innerHTML = ''; container.dataset.posFingerprint = ''; }
    return;
  }

  // Fingerprint includes id + status so a close triggers rebuild
  const fingerprint = all.map((p) => p.id + ':' + p.status).join(',');
  const prev = container.dataset.posFingerprint || '';

  if (fingerprint === prev) {
    // Same set — update P&L values in-place
    all.forEach((p) => {
      const card = container.querySelector(`[data-pos-id="${p.id}"]`);
      if (!card) return;
      const pnl = p.pnl_usd || 0;
      const pnlClass = pnl > 0 ? 'pnl-positive' : pnl < 0 ? 'pnl-negative' : 'pnl-zero';
      const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
      const pnlVal = card.querySelector('.pos-card-pnl .pos-card-stat-value');
      if (pnlVal) {
        pnlVal.textContent = pnlStr;
        pnlVal.className = 'pos-card-stat-value ' + pnlClass;
      }
    });
    return;
  }

  // Track which IDs are new (weren't in previous render)
  const prevIds = new Set((prev || '').split(',').map((s) => s.split(':')[0]).filter(Boolean));

  // Rebuild DOM
  const html = all.map((p) => buildPosCard(p)).join('');
  container.innerHTML = html;
  container.dataset.posFingerprint = fingerprint;

  // Only animate cards that are genuinely new
  if (window.Motion) {
    const newCards = [...container.querySelectorAll('.pos-card')].filter(
      (el) => !prevIds.has(el.dataset.posId)
    );
    if (newCards.length > 0) {
      Motion.animate(newCards, { opacity: [0, 1], y: [12, 0] }, { duration: 0.35, ease: 'ease-out' });
    }
  }
}

function buildPosCard(p) {
  const resolved = p.status === 'closed';
  const isYes = p.direction === 'BUY_YES';
  const dirLabel = isYes ? 'Yes' : 'No';
  const dirClass = isYes ? 'pos-card-direction--yes' : 'pos-card-direction--no';
  const price = p.fill_price || p.entry_price || 0;
  const shares = price > 0 ? Math.round(p.size_usd / price) : 0;
  const pnl = p.pnl_usd || 0;
  const pnlClass = pnl > 0 ? 'pnl-positive' : pnl < 0 ? 'pnl-negative' : 'pnl-zero';
  const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);

  // Determine outcome: only count as won if actually filled AND has positive P&L
  const filled = p.fill_price > 0 && p.filled_at_ms > 0;
  const won = resolved && filled && pnl > 0;
  const lost = resolved && filled && pnl < 0;
  const noFill = resolved && !filled;

  let cardClass = 'pos-card';
  if (resolved) {
    cardClass += ' pos-card--resolved';
    if (won) cardClass += ' pos-card--win';
    else if (lost) cardClass += ' pos-card--loss';
  }

  let badge;
  if (!resolved) {
    badge = `<span class="pos-card-badge pos-card-badge--live">Live</span>`;
  } else if (noFill) {
    badge = `<span class="pos-card-badge pos-card-badge--nofill">No Fill</span>`;
  } else if (won) {
    badge = `<span class="pos-card-badge pos-card-badge--win">Won</span>`;
  } else if (lost) {
    badge = `<span class="pos-card-badge pos-card-badge--loss">Lost</span>`;
  } else {
    badge = `<span class="pos-card-badge pos-card-badge--nofill">Break Even</span>`;
  }

  return `<div class="${cardClass}" data-pos-id="${p.id}">
    <div class="pos-card-header">
      <span class="pos-card-market">BTC 5min Up/Down</span>
      <div class="pos-card-header-right">
        ${badge}
        <span class="pos-card-direction ${dirClass}">${dirLabel}</span>
      </div>
    </div>
    <div class="pos-card-row">
      <div class="pos-card-stat">
        <span class="pos-card-stat-label">Avg</span>
        <span class="pos-card-stat-value">${(price * 100).toFixed(0)}\u00a2</span>
      </div>
      <div class="pos-card-stat">
        <span class="pos-card-stat-label">Shares</span>
        <span class="pos-card-stat-value">${shares}</span>
      </div>
      <div class="pos-card-stat">
        <span class="pos-card-stat-label">Cost</span>
        <span class="pos-card-stat-value">$${p.size_usd.toFixed(2)}</span>
      </div>
      <div class="pos-card-stat pos-card-pnl">
        <span class="pos-card-stat-label">P&L</span>
        <span class="pos-card-stat-value ${pnlClass}">${pnlStr}</span>
      </div>
    </div>
  </div>`;
}

function timeSince(ms) {
  const seconds = Math.floor((Date.now() - ms) / 1000);
  if (seconds < 60) return seconds + 's ago';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + 'm ago';
  const hours = Math.floor(minutes / 60);
  return hours + 'h ago';
}


// ── Live Trading Tab ──

function renderLive(snap) {
  renderSpreadChart();
  renderEventStream();

  const spreadEl = document.getElementById('spread-count');
  if (spreadEl) spreadEl.textContent = `${state.spreadHistory.length} spreads`;

  const eventEl = document.getElementById('event-count');
  if (eventEl) eventEl.textContent = `${state.events.length} events`;
}

function renderSpreadChart() {
  const ctx = document.getElementById('chart-spreads');
  if (!ctx) return;

  const data = state.spreadHistory.map(s => ({
    x: s.timestamp_ms,
    y: s.spread_pct,
  }));

  if (!charts.spreads) {
    // Don't create time-scale chart until we have data
    if (data.length < 1) return;
    charts.spreads = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          data: data,
          backgroundColor: 'rgba(0, 112, 243, 0.6)',
          borderColor: '#0070f3',
          borderWidth: 1,
          pointRadius: 3,
          pointHoverRadius: 5,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            grid: { display: false },
            ticks: { maxTicksLimit: 8, font: { size: 10 } },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: {
              font: { size: 10 },
              callback: v => v.toFixed(1) + '%',
            },
            suggestedMin: 0,
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: item => `Spread: ${item.parsed.y.toFixed(2)}%`,
            },
          },
        },
      },
    });
  } else {
    charts.spreads.data.datasets[0].data = data;
    charts.spreads.update('none');
  }
}

function renderEventStream() {
  const el = document.getElementById('event-stream');
  if (!el) return;

  if (!state.events.length) {
    el.innerHTML = '<div class="empty-state"><p>Waiting for events…</p></div>';
    return;
  }

  el.innerHTML = state.events.slice(0, 100).map(evt => {
    const cfg = EVENT_CONFIG[evt.event_type] || { label: evt.event_type, pill: 'pill--gray' };
    const detail = formatEventDetail(evt);
    return `
      <div class="event-row">
        <span class="event-time">${fmtTimeShort(evt.timestamp_ms)}</span>
        <span class="event-type"><span class="pill ${cfg.pill}">${cfg.label}</span></span>
        <span class="event-detail">${detail}</span>
      </div>
    `;
  }).join('');
}

function formatEventDetail(evt) {
  const d = evt.data || {};
  switch (evt.event_type) {
    case 'spread_detected':  return `${d.market_id || ''} spread: ${fmtPctRaw(d.spread_pct)} ${d.direction || ''}`;
    case 'trade_signal':     return `${d.action || ''} ${d.market_id || ''} ${d.direction || ''}`;
    case 'order_filled':     return `pos: ${d.position_id || ''} slippage: ${d.slippage_bps || 0}bps`;
    case 'order_placed':     return `pos: ${d.position_id || ''} size: ${fmtCurrency(d.size_usd)}`;
    case 'position_update':  return `${d.action || ''} ${d.id || ''} ${d.direction || ''} ${fmtCurrency(d.size_usd)}`;
    case 'risk_alert':       return `[${d.level || ''}] ${d.message || d.alert || ''}`;
    case 'price_tick':       return `${d.exchange || d.source || ''} $${d.last || d.mid || ''}`;
    default:                 return JSON.stringify(d).substring(0, 80);
  }
}


// ── Positions Tab ──

function renderPositions(snap) {
  const active = snap.active_positions || [];
  const closed = snap.closed_positions || [];

  document.getElementById('active-pos-count').textContent = active.length;
  document.getElementById('closed-pos-count').textContent = closed.length;

  renderPositionTable('active-positions-table', active, false);
  renderPositionTable('closed-positions-table', closed, true);
}

function renderPositionTable(containerId, positions, isClosed) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (!positions.length) {
    el.innerHTML = `<div class="empty-state"><p>No ${isClosed ? 'closed' : 'active'} positions</p></div>`;
    return;
  }

  const headers = isClosed
    ? '<th>ID</th><th>Market</th><th>Direction</th><th>Entry</th><th>Exit</th><th class="text-right">Size</th><th class="text-right">P&L</th><th>Duration</th>'
    : '<th>ID</th><th>Market</th><th>Direction</th><th>Entry</th><th>Fill</th><th class="text-right">Size</th><th class="text-right">P&L</th><th>Status</th>';

  const rows = positions.map(p => {
    const dirPill = p.direction === 'BUY_YES' ? 'pill--green' : 'pill--red';
    const dur = p.closed_at_ms && p.created_at_ms
      ? Math.round((p.closed_at_ms - p.created_at_ms) / 1000) + 's'
      : relTime(p.created_at_ms);

    if (isClosed) {
      return `<tr>
        <td>${p.id}</td>
        <td>${shortMarket(p.market_id)}</td>
        <td><span class="pill ${dirPill}">${p.direction}</span></td>
        <td>${fmtNum(p.entry_price, 4)}</td>
        <td>${fmtNum(p.exit_price, 4)}</td>
        <td class="text-right">${fmtCurrency(p.size_usd)}</td>
        <td class="text-right ${pnlClass(p.pnl_usd)}">${fmtPnl(p.pnl_usd)}</td>
        <td class="text-muted">${dur}</td>
      </tr>`;
    } else {
      return `<tr>
        <td>${p.id}</td>
        <td>${shortMarket(p.market_id)}</td>
        <td><span class="pill ${dirPill}">${p.direction}</span></td>
        <td>${fmtNum(p.entry_price, 4)}</td>
        <td>${p.fill_price ? fmtNum(p.fill_price, 4) : '—'}</td>
        <td class="text-right">${fmtCurrency(p.size_usd)}</td>
        <td class="text-right ${pnlClass(p.pnl_usd)}">${fmtPnl(p.pnl_usd)}</td>
        <td><span class="pill pill--blue">${p.status}</span></td>
      </tr>`;
    }
  }).join('');

  el.innerHTML = `<table class="data-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
}

function shortMarket(id) {
  if (!id) return '—';
  return id.length > 16 ? id.substring(0, 12) + '…' : id;
}


// ── Risk Monitor Tab ──

function renderRisk(snap) {
  const p = snap.portfolio || {};
  const platform = snap.platform || {};
  const alerts = snap.alerts || [];

  // KPIs
  const hasEmergency = alerts.some(a => a.level === 'critical' || a.level === 'emergency');
  setTextWithColor('kpi-tail-risk', hasEmergency ? 'ALERT' : 'OK',
    hasEmergency ? 'text-red' : 'text-green');
  setTextWithColor('kpi-risk-drawdown', '-' + fmtPct(p.drawdown_pct || 0),
    (p.drawdown_pct || 0) > 0.03 ? 'text-red' : 'text-green');
  setText('kpi-gas', platform.gas_gwei ? fmtNum(platform.gas_gwei, 0) + ' gwei' : '—');
  setText('kpi-clob-latency', platform.clob_latency_ms ? platform.clob_latency_ms + 'ms' : '—');

  // Alert count
  document.getElementById('alert-count').textContent = alerts.length;

  // Alert list
  renderAlertList(alerts);

  // CEX ticks
  renderCexTicks(snap.ticks || {});
}

function renderAlertList(alerts) {
  const el = document.getElementById('alert-list');
  if (!el) return;

  if (!alerts.length) {
    el.innerHTML = '<div class="empty-state"><p>No risk alerts — all clear</p></div>';
    return;
  }

  el.innerHTML = alerts.map(a => `
    <div class="alert-row">
      <div class="alert-dot alert-dot--${a.level}"></div>
      <div class="alert-content">
        <div class="alert-message">${a.message}</div>
        ${a.action ? `<div class="alert-action">Action: ${a.action}</div>` : ''}
      </div>
      <div class="alert-time">${fmtTimeShort(a.timestamp_ms)}</div>
    </div>
  `).join('');
}

function renderCexTicks(ticks) {
  const el = document.getElementById('cex-ticks');
  if (!el) return;

  const exchanges = Object.keys(ticks);
  if (!exchanges.length) {
    el.innerHTML = '<div class="empty-state"><p>Waiting for ticks…</p></div>';
    return;
  }

  el.innerHTML = `
    <div class="health-grid">
      ${exchanges.map(ex => {
        const t = ticks[ex];
        return `
          <div class="health-row">
            <span class="health-name" style="text-transform:capitalize">${ex}</span>
            <span class="health-status text-mono">
              <span class="text-muted">bid</span> ${fmtNum(t.bid, 1)}
              <span class="text-muted" style="margin-left:8px">ask</span> ${fmtNum(t.ask, 1)}
              <span class="font-medium" style="margin-left:8px">$${fmtNum(t.last || t.mid, 1)}</span>
            </span>
          </div>
        `;
      }).join('')}
    </div>
  `;
}


// ── Performance Tab ──

function renderPerformance(snap) {
  const bt = snap.backtest || {};
  const fm = snap.fill_metrics || {};
  const p = snap.portfolio || {};

  // KPIs — fall back to portfolio stats when backtest hasn't run
  const totalTrades = bt.total_trades != null ? bt.total_trades : (p.closed_trades || '—');
  const winRate = bt.win_rate != null ? bt.win_rate : p.win_rate;
  setText('kpi-total-trades', totalTrades);
  setTextWithColor('kpi-bt-win-rate', winRate != null ? fmtPct(winRate) : '—',
    winRate >= 0.7 ? 'text-green' : 'text-amber');
  setText('kpi-profit-factor', bt.profit_factor != null ? fmtNum(bt.profit_factor) : '—');
  setText('kpi-expectancy', bt.expectancy != null ? fmtCurrency(bt.expectancy) : '—');
  setTextWithColor('kpi-fill-rate', fm.fill_rate != null ? fmtPct(fm.fill_rate) : '—',
    fm.fill_rate >= 0.95 ? 'text-green' : 'text-amber');
  setText('kpi-avg-slippage', fm.avg_slippage_bps != null ? fmtNum(fm.avg_slippage_bps) + ' bps' : '—');

  // Fetch and render fills table + charts
  renderPnlDistChart(snap.closed_positions || []);
  renderFillTimeChart(snap);
  renderRecentFillsTable(snap);
}

function renderPnlDistChart(positions) {
  const ctx = document.getElementById('chart-pnl-dist');
  if (!ctx || !positions.length) return;

  // Build histogram bins
  const pnls = positions.map(p => p.pnl_usd).filter(v => v != null && !isNaN(v));
  if (pnls.length < 2) return;

  const min = Math.min(...pnls);
  const max = Math.max(...pnls);
  const range = max - min || 1;
  const binCount = Math.min(20, Math.max(5, Math.ceil(pnls.length / 3)));
  const binSize = range / binCount;

  const bins = Array(binCount).fill(0);
  const labels = [];

  for (let i = 0; i < binCount; i++) {
    const lo = min + i * binSize;
    labels.push(fmtCurrency(lo));
  }

  pnls.forEach(v => {
    let idx = Math.floor((v - min) / binSize);
    if (idx >= binCount) idx = binCount - 1;
    bins[idx]++;
  });

  const colors = labels.map((_, i) => {
    const midVal = min + (i + 0.5) * binSize;
    return midVal >= 0 ? 'rgba(0, 200, 83, 0.6)' : 'rgba(255, 68, 68, 0.6)';
  });

  if (!charts.pnlDist) {
    charts.pnlDist = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{ data: bins, backgroundColor: colors, borderWidth: 0, borderRadius: 3 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 6, font: { size: 10 } } },
          y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { font: { size: 10 } }, beginAtZero: true },
        },
      },
    });
  } else {
    charts.pnlDist.data.labels = labels;
    charts.pnlDist.data.datasets[0].data = bins;
    charts.pnlDist.data.datasets[0].backgroundColor = colors;
    charts.pnlDist.update('none');
  }
}

function renderFillTimeChart(snap) {
  // Placeholder — would need fill time data from the API
  const ctx = document.getElementById('chart-fill-time');
  if (!ctx) return;

  if (!charts.fillTime) {
    charts.fillTime = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['<100ms', '100-200', '200-300', '300-500', '500ms+'],
        datasets: [{
          data: [0, 0, 0, 0, 0],
          backgroundColor: 'rgba(0, 112, 243, 0.5)',
          borderWidth: 0,
          borderRadius: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
          y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { font: { size: 10 } }, beginAtZero: true },
        },
      },
    });
  }
}

function renderRecentFillsTable(snap) {
  const el = document.getElementById('recent-fills-table');
  if (!el) return;

  // Use the fill metrics from the API; full fill data requires /api/performance/fills
  // For now, show a summary
  const fm = snap.fill_metrics || {};
  if (!fm.total_fills) {
    el.innerHTML = '<div class="empty-state"><p>No fills yet</p></div>';
    return;
  }

  el.innerHTML = `
    <div class="health-grid">
      <div class="health-row">
        <span class="health-name">Total Fills</span>
        <span class="health-status text-mono font-medium">${fm.total_fills}</span>
      </div>
      <div class="health-row">
        <span class="health-name">Pending Orders</span>
        <span class="health-status text-mono font-medium">${fm.pending_orders}</span>
      </div>
      <div class="health-row">
        <span class="health-name">Avg Fill Time</span>
        <span class="health-status text-mono font-medium">${fmtNum(fm.avg_fill_time_ms, 0)}ms</span>
      </div>
      <div class="health-row">
        <span class="health-name">Avg Slippage</span>
        <span class="health-status text-mono font-medium">${fmtNum(fm.avg_slippage_bps)} bps</span>
      </div>
      <div class="health-row">
        <span class="health-name">Fill Rate</span>
        <span class="health-status text-mono font-medium ${fm.fill_rate >= 0.95 ? 'text-green' : 'text-amber'}">${fmtPct(fm.fill_rate)}</span>
      </div>
    </div>
  `;
}


// ── Toast Notifications ──

function showToast(type, message) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 200);
  }, 4000);
}


// ── Helpers ──

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setTextWithColor(id, value, colorClass) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
    el.className = 'kpi-value ' + (colorClass || '');
  }
}

function createGradient(canvasEl, color) {
  try {
    const ctx2d = canvasEl.getContext('2d');
    if (!ctx2d) return color + '15';
    const gradient = ctx2d.createLinearGradient(0, 0, 0, 260);
    gradient.addColorStop(0, color + '33');
    gradient.addColorStop(1, color + '00');
    return gradient;
  } catch (e) {
    return color + '15';
  }
}


// ── Init ──

document.addEventListener('DOMContentLoaded', () => {
  new DashboardWS();
});
