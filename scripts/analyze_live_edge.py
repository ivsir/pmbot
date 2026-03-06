"""Analyze live trading edge: are we getting fills cheap enough to profit?

Reads the bot log and compares:
  - Fill price (what we paid)
  - Model's fair probability
  - Actual outcome (win/loss)
  - Implied edge = win_rate - fill_price
"""

import re
import sys

LOG_FILE = "/tmp/arb_bot.log"


def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)


def parse_log():
    with open(LOG_FILE, 'rb') as f:
        raw = f.read().decode('utf-8', errors='replace')

    lines = raw.split('\n')
    lines = [strip_ansi(l) for l in lines]

    # Collect fills: order_id → {price, size}
    fills = {}
    for line in lines:
        m = re.search(r'fill_confirmed\s+fill_price=([\d.]+)\s+filled_size=([\d.]+)\s+order_id=(\S+)', line)
        if m:
            fills[m.group(3)] = {
                'fill_price': float(m.group(1)),
                'filled_size': float(m.group(2)),
            }

    # Collect orders: order_id → {price, side, size}
    orders = {}
    for line in lines:
        m = re.search(r'order_placed\s+order_id=(\S+)\s+price=([\d.]+)\s+side=(\w+)\s+size=([\d.]+)', line)
        if m:
            orders[m.group(1)] = {
                'price': float(m.group(2)),
                'side': m.group(3),
                'size': float(m.group(4)),
            }

    # Collect displacement signals: market → {displacement, fair_up, pm_mid, direction}
    signals = {}
    for line in lines:
        m = re.search(r'displacement_detected\s+.*?direction=(\w+)\s+displacement=([-\d.]+)\s+fair_up=([\d.]+)\s+market=(\S+).*?pm_mid=([\d.]+)', line)
        if m:
            market = m.group(4)
            signals[market] = {
                'direction': m.group(1),
                'displacement': float(m.group(2)),
                'fair_up': float(m.group(3)),
                'pm_mid': float(m.group(5)),
            }

    # Collect alpha entries: market → {confidence, edge, entry details}
    entries = {}
    for line in lines:
        m = re.search(r'alpha_signal\.entry\s+.*?confidence=([\d.]+)\s+direction=(\w+)\s+edge=([\d.]+).*?market=(\S+).*?size_usd=([\d.]+)', line)
        if m:
            market = m.group(4)
            if market not in entries:  # first entry per market
                entries[market] = {
                    'confidence': float(m.group(1)),
                    'direction': m.group(2),
                    'edge': float(m.group(3)),
                    'size_usd': float(m.group(5)),
                }

    # Collect resolutions: market → {won, pnl, outcome, direction}
    resolutions = {}
    for line in lines:
        m = re.search(r'position_resolved\s+.*?direction=(\w+)\s+market_id=(\S+)\s+outcome=(\w+)\s+pnl=([-\d.]+).*?won=(\w+)', line)
        if m:
            market = m.group(2)
            resolutions[market] = {
                'direction': m.group(1),
                'outcome': m.group(3),
                'pnl': float(m.group(4)),
                'won': m.group(5) == 'True',
            }

    return fills, orders, signals, entries, resolutions


def main():
    fills, orders, signals, entries, resolutions = parse_log()

    print(f"\n{'='*100}")
    print(f"  LIVE EDGE ANALYSIS — Fill Price vs Fair Value vs Outcome")
    print(f"{'='*100}\n")

    print(f"  Raw counts: {len(fills)} fills, {len(orders)} orders, "
          f"{len(signals)} signals, {len(entries)} entries, {len(resolutions)} resolutions\n")

    # Match resolutions with their entry details
    trades = []
    for market, res in resolutions.items():
        # Find matching signal
        # market IDs in resolutions are truncated, match by prefix
        sig = None
        for sig_market, s in signals.items():
            if market in sig_market or sig_market.startswith(market):
                sig = s
                break

        entry = None
        for ent_market, e in entries.items():
            if market in ent_market or ent_market.startswith(market):
                entry = e
                break

        trades.append({
            'market': market[:20],
            'direction': res['direction'],
            'won': res['won'],
            'pnl': res['pnl'],
            'outcome': res['outcome'],
            'displacement': sig['displacement'] if sig else None,
            'fair_up': sig['fair_up'] if sig else None,
            'pm_mid': sig['pm_mid'] if sig else None,
            'confidence': entry['confidence'] if entry else None,
            'model_edge': entry['edge'] if entry else None,
        })

    if not trades:
        print("  No matched trades found.")
        return

    # ── Trade-by-trade breakdown ──
    print(f"  {'#':>3} {'Dir':>8} {'Won':>4} {'PnL':>7} {'Disp%':>7} {'FairUp':>7} {'PmMid':>7} {'ModelEdge':>9}")
    print(f"  {'─'*3} {'─'*8} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")

    for i, t in enumerate(trades):
        disp = f"{t['displacement']:.3f}" if t['displacement'] is not None else "?"
        fair = f"{t['fair_up']:.3f}" if t['fair_up'] is not None else "?"
        pm = f"{t['pm_mid']:.3f}" if t['pm_mid'] is not None else "?"
        edge = f"{t['model_edge']:.1f}" if t['model_edge'] is not None else "?"
        won = "W" if t['won'] else "L"
        print(f"  {i+1:>3} {t['direction']:>8} {won:>4} {t['pnl']:>+7.2f} {disp:>7} {fair:>7} {pm:>7} {edge:>9}")

    # ── Aggregate stats ──
    wins = [t for t in trades if t['won']]
    losses = [t for t in trades if not t['won']]

    print(f"\n{'='*100}")
    print(f"  AGGREGATE STATS")
    print(f"{'='*100}\n")
    print(f"  Total: {len(trades)} trades | {len(wins)}W/{len(losses)}L = {len(wins)/len(trades)*100:.1f}% WR")
    print(f"  Net PnL: ${sum(t['pnl'] for t in trades):.2f}")
    print(f"  Avg win:  ${sum(t['pnl'] for t in wins)/len(wins):.2f}" if wins else "")
    print(f"  Avg loss: ${sum(t['pnl'] for t in losses)/len(losses):.2f}" if losses else "")

    # ── Fill price analysis ──
    print(f"\n{'='*100}")
    print(f"  FILL PRICE ANALYSIS — What prices are we actually getting?")
    print(f"{'='*100}\n")

    fill_prices = sorted(set(f['fill_price'] for f in fills.values()))
    print(f"  Fill prices seen: {fill_prices}")
    print(f"  Total fills: {len(fills)}")

    for price in fill_prices:
        count = sum(1 for f in fills.values() if f['fill_price'] == price)
        print(f"    ${price:.2f}: {count} fills")

    # ── PM Mid analysis (what PM was pricing when we entered) ──
    pm_mids = [t['pm_mid'] for t in trades if t['pm_mid'] is not None]
    if pm_mids:
        print(f"\n{'='*100}")
        print(f"  PM PRICING AT ENTRY — Is PM already efficient?")
        print(f"{'='*100}\n")

        avg_pm = sum(pm_mids) / len(pm_mids)
        print(f"  Avg PM mid at entry: {avg_pm:.3f}")
        print(f"  PM mid range: {min(pm_mids):.3f} - {max(pm_mids):.3f}")

        # For BUY_YES: we want pm_mid < 0.50 (cheap YES tokens)
        # For BUY_NO: we want pm_mid > 0.50 (cheap NO tokens = 1-pm_mid)
        yes_trades = [t for t in trades if t['direction'] == 'BUY_YES' and t['pm_mid'] is not None]
        no_trades = [t for t in trades if t['direction'] == 'BUY_NO' and t['pm_mid'] is not None]

        if yes_trades:
            yes_pm = [t['pm_mid'] for t in yes_trades]
            yes_wr = sum(1 for t in yes_trades if t['won']) / len(yes_trades) * 100
            print(f"\n  BUY_YES ({len(yes_trades)} trades, {yes_wr:.0f}% WR):")
            print(f"    PM mid (YES price): avg={sum(yes_pm)/len(yes_pm):.3f}  range={min(yes_pm):.3f}-{max(yes_pm):.3f}")
            print(f"    Ideal: pm_mid < 0.50 (cheap YES). We need to buy YES below fair value.")
            for t in yes_trades:
                fair = t['fair_up'] or 0
                pm = t['pm_mid'] or 0
                edge = (fair - pm) * 100
                won = "W" if t['won'] else "L"
                print(f"      {won} pm={pm:.3f} fair={fair:.3f} edge={edge:+.1f}% disp={t['displacement']:.4f}%")

        if no_trades:
            no_pm = [t['pm_mid'] for t in no_trades]
            no_wr = sum(1 for t in no_trades if t['won']) / len(no_trades) * 100
            print(f"\n  BUY_NO ({len(no_trades)} trades, {no_wr:.0f}% WR):")
            print(f"    PM mid (YES price): avg={sum(no_pm)/len(no_pm):.3f}  range={min(no_pm):.3f}-{max(no_pm):.3f}")
            print(f"    Ideal: pm_mid > 0.50 (cheap NO = 1-pm_mid). We need to buy NO below fair value.")
            for t in no_trades:
                fair = 1.0 - (t['fair_up'] or 0.5)
                pm = 1.0 - (t['pm_mid'] or 0.5)
                edge = (fair - pm) * 100
                won = "W" if t['won'] else "L"
                print(f"      {won} no_price={pm:.3f} fair_no={fair:.3f} edge={edge:+.1f}% disp={t['displacement']:.4f}%")

    # ── Key question: are fills at fair value? ──
    print(f"\n{'='*100}")
    print(f"  KEY QUESTION: Is there exploitable edge after PM repricing?")
    print(f"{'='*100}\n")

    edges = []
    for t in trades:
        if t['fair_up'] is not None and t['pm_mid'] is not None:
            if t['direction'] == 'BUY_YES':
                # Edge = fair_up - pm_mid (positive = we think YES is underpriced)
                edge = t['fair_up'] - t['pm_mid']
            else:
                # Edge = (1-fair_up) - (1-pm_mid) = pm_mid - fair_up
                edge = t['pm_mid'] - t['fair_up']
            edges.append(edge * 100)

    if edges:
        avg_edge = sum(edges) / len(edges)
        print(f"  Model's perceived edge at entry: avg={avg_edge:.1f}%  range={min(edges):.1f}%-{max(edges):.1f}%")
        print(f"  Actual WR: {len(wins)/len(trades)*100:.1f}%")
        print(f"  WR needed to break even at avg fill price: ~50% (binary payout)")
        print()

        if avg_edge > 20:
            print(f"  ⚠ Edge looks unrealistically large ({avg_edge:.0f}%).")
            print(f"  This means PM is priced FAR from our model's fair value.")
            print(f"  Either: (a) our model is wrong, or (b) PM orderbook is stale/illiquid.")
            print(f"  Given live WR of {len(wins)/len(trades)*100:.0f}%, the model is likely overconfident.")
        elif avg_edge > 5:
            print(f"  Edge appears moderate ({avg_edge:.1f}%). Should be profitable if real.")
        else:
            print(f"  Edge is small ({avg_edge:.1f}%). Hard to profit after transaction costs.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
