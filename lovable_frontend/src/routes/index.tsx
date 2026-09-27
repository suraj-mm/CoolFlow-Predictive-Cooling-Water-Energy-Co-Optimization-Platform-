import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowDownRight, BrainCircuit, CircleGauge, CloudSun, Droplets, Eye, Fan, Gauge, Info, Layers3, LockKeyhole, Server, ShieldCheck, Sparkles, Thermometer, Waves, Zap } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Button } from '@/components/ui/button';
import { initialMonthRecords, type AuraState, type NodeState } from '@/lib/aura-data';

export const Route = createFileRoute('/')({
  head: () => ({ meta: [
    { title: 'CoolFlow — "Predictive Cooling & Water-Energy Co-Optimization Platform"' },
    { name: 'description', content: 'View-only executive monitoring console for autonomous datacenter thermal control, water compliance, and explainable cooling decisions.' },
    { property: 'og:title', content: 'CoolFlow — "Predictive Cooling & Water-Energy Co-Optimization Platform"' },
    { property: 'og:description', content: 'View-only executive monitoring console for autonomous datacenter thermal control, water compliance, and explainable cooling decisions.' },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary_large_image' },
  ] }),
  component: Dashboard,
});

const modeNames = ['Free-Air Economizer', 'Evaporative Cooling', 'Mechanical Chiller'];
const modeShort = ['FREE-AIR', 'EVAPORATIVE', 'CHILLER'];
const css = (name: string) => `var(--${name})`;
const pareto = [
  { x: 15, y: 78, sla: 14 }, { x: 23, y: 66, sla: 12 }, { x: 31, y: 56, sla: 10 },
  { x: 40, y: 47, sla: 8 }, { x: 50, y: 39, sla: 7 }, { x: 61, y: 33, sla: 5 },
  { x: 72, y: 29, sla: 4 }, { x: 82, y: 27, sla: 3 },
];

function PanelTitle({ icon: Icon, label, aside }: { icon: typeof Activity; label: string; aside?: React.ReactNode }) {
  return <div className="flex min-w-0 items-center justify-between gap-3 border-b border-border px-5 py-4">
    <div className="flex min-w-0 items-center gap-2.5"><Icon className="size-4 shrink-0 text-cyan" strokeWidth={1.8} /><h2 className="text-[12px] font-bold uppercase leading-4 text-foreground tracking-[.12em]">{label}</h2></div>{aside}
  </div>;
}
function Dot({ color = 'emerald' }: { color?: 'emerald' | 'cyan' | 'amber' | 'destructive' }) { return <span className={`inline-block size-1.5 shrink-0 rounded-full bg-${color} signal-pulse`} />; }
function TinyBar({ value, color = 'cyan' }: { value: number; color?: 'cyan' | 'sky' | 'emerald' | 'amber' | 'indigo' | 'destructive' }) { return <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary"><div className={`h-full rounded-full bg-${color} transition-all duration-500`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>; }
function Metric({ icon: Icon, label, value, unit, footer, color = 'cyan', children }: { icon: typeof Activity; label: string; value: string | number; unit?: string; footer: React.ReactNode; color?: string; children?: React.ReactNode }) {
  return <div className="panel flex min-h-[170px] min-w-0 flex-col justify-between p-4 lg:p-5">
    <div className="flex items-start justify-between gap-2"><p className="text-[10px] font-bold uppercase leading-4 tracking-[.11em] text-muted-foreground">{label}</p><Icon className={`size-4 shrink-0 text-${color}`} strokeWidth={1.7} /></div>
    <div className="mt-4"><div className="flex items-baseline gap-1.5"><span className="font-mono text-[29px] font-medium leading-none text-foreground lg:text-[32px]">{value}</span>{unit && <span className="font-mono text-xs text-muted-foreground">{unit}</span>}</div>{children && <div className="mt-3">{children}</div>}</div>
    <div className="mt-3 min-h-4 text-[10px] leading-4 text-muted-foreground">{footer}</div>
  </div>;
}
function NodeCard({ node, selected, onSelect }: { node: NodeState; selected: boolean; onSelect: () => void }) {
  return <Button variant="ghost" onClick={onSelect} aria-label={`Inspect Node ${node.id}`} className={`h-auto w-full justify-start whitespace-normal rounded-sm border px-3 py-3 text-left hover:bg-accent/40 ${selected ? 'border-cyan/55 bg-cyan/5' : 'border-border bg-surface/60'}`}>
    <div className="w-full min-w-0">
      <div className="mb-3 flex items-center justify-between gap-2"><div className="flex items-center gap-2"><span className={`flex size-7 items-center justify-center rounded-sm border ${selected ? 'border-cyan/40 bg-cyan/10 text-cyan' : 'border-border bg-secondary text-muted-foreground'}`}><Server className="size-3.5" /></span><span className="text-xs font-bold text-foreground">NODE 0{node.id}</span></div><span className="flex items-center gap-1.5 font-mono text-[9px] uppercase text-emerald"><Dot /> OPTIMAL</span></div>
      <div className="grid grid-cols-4 gap-1 text-center"><div><p className="text-[9px] text-muted-foreground">CPU</p><p className="mt-1 font-mono text-[11px] text-foreground">{node.cpu}%</p></div><div><p className="text-[9px] text-muted-foreground">POWER</p><p className="mt-1 font-mono text-[11px] text-foreground">{node.power}W</p></div><div><p className="text-[9px] text-muted-foreground">FAN</p><p className="mt-1 font-mono text-[11px] text-foreground">{node.fan}%</p></div><div><p className="text-[9px] text-muted-foreground">TEMP</p><p className="mt-1 font-mono text-[11px] text-cyan">{node.temp}°</p></div></div>
      <div className="mt-3"><TinyBar value={node.cpu} color="sky" /></div>
    </div>
  </Button>;
}
function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return <div className="rounded border border-border bg-popover px-3 py-2 text-xs shadow-xl"><p className="mb-1 font-mono text-muted-foreground">{label}</p>{payload.map(item => <p key={item.name} style={{ color: item.color }}>{item.name}: {item.value}</p>)}</div>;
}

function Dashboard() {
  const [records] = useState<AuraState[]>(initialMonthRecords);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [selectedNode, setSelectedNode] = useState(0);
  const [activePareto, setActivePareto] = useState(4);
  const [log, setLog] = useState<string[]>([
    'Tick #1001 · Free-Air Economizer active · 0 L/h · Sub-zero ambient baseline',
    '30-Day continuous dataset loaded from EQCAM_Dataset_FINAL_with_RC (March 2022)',
    'Chebyshev knee-point co-optimization active (145ms response)'
  ]);

  const state = records[currentIndex] ?? records[0];

  useEffect(() => {
    if (!isPlaying || records.length === 0) return;
    const timer = setInterval(() => {
      setCurrentIndex(prev => (prev + 1) % records.length);
    }, 1000 / speed);
    return () => clearInterval(timer);
  }, [isPlaying, speed, records.length]);

  const jumpDay = (diff: number) => {
    const step = 144; // 144 ticks = 24 hours
    setCurrentIndex(prev => Math.max(0, Math.min(records.length - 1, prev + diff * step)));
  };

  const jumpToDay = (dayNum: number) => {
    const targetIdx = Math.min(records.length - 1, Math.max(0, (dayNum - 1) * 144));
    setCurrentIndex(targetIdx);
  };

  const node: NodeState = state.nodes[selectedNode] ?? state.nodes[0];
  const status = state.peakTemp > 80 ? 'CRITICAL HOTSPOT' : state.peakTemp >= 76 ? 'PRE-HOTSPOT' : 'NORMAL RANGE';
  const statusColor = state.peakTemp > 80 ? 'destructive' : state.peakTemp >= 76 ? 'amber' : 'emerald';
  const capPercent = Math.min(100, (state.water / state.cap) * 100);
  const remaining = Math.max(0, state.cap - state.water);

  const forecast = useMemo(() => [
    { name: 'Workload intensity', value: Math.min(95, Math.round(state.workload * 0.9)), color: 'sky', detail: `${state.shap.workload > 0 ? '+' : ''}${state.shap.workload}°C` },
    { name: 'Fan duty airflow', value: Math.round(node.fan * 0.8), color: 'emerald', detail: `${state.shap.fan > 0 ? '+' : ''}${state.shap.fan}°C` },
    { name: 'Stull wet-bulb baseline', value: Math.round(Math.abs(state.wetBulb) * 2.2), color: 'amber', detail: `${state.shap.wetBulb > 0 ? '+' : ''}${state.shap.wetBulb}°C` },
    { name: 'Thermal lag inertia', value: 25, color: 'indigo', detail: `+${state.shap.lag}°C` },
  ], [state.workload, state.shap, state.wetBulb, node.fan]);

  return <div className="min-h-screen bg-background text-foreground dashboard-scroll">
    <div className="mx-auto max-w-[1720px] px-4 pb-12 pt-5 sm:px-6 lg:px-8">
      <header className="mb-4">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-5">
          <div className="flex min-w-0 items-start gap-3.5">
            <div className="relative mt-0.5 flex size-10 shrink-0 items-center justify-center rounded border border-cyan/35 bg-cyan/10 text-cyan glow-cyan">
              <BrainCircuit className="size-5" strokeWidth={1.5} />
              <span className="absolute -bottom-1 -right-1 size-2 border border-background bg-emerald" />
            </div>
            <div className="min-w-0">
              <div className="mb-1 flex items-center gap-2">
                <span className="font-mono text-[10px] font-semibold uppercase tracking-[.22em] text-cyan">COOLFLOW / COMMAND CENTER</span>
                <span className="text-muted-foreground/50">•</span>
                <span className="font-mono text-[10px] text-muted-foreground">SYSTEM OVERVIEW</span>
              </div>
              <h1 className="text-xl font-semibold leading-tight text-foreground sm:text-[25px]">
                CoolFlow <span className="font-normal text-muted-foreground">|</span> "Predictive Cooling & Water-Energy Co-Optimization Platform"
              </h1>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2 rounded border border-cyan/30 bg-cyan/8 px-3 py-2 font-mono text-[10px] uppercase tracking-[.1em] text-cyan">
            <Eye className="size-3.5" /> VIEW ONLY CONSOLE
          </div>
        </div>
        <div className="flex flex-col justify-between gap-4 border-b border-border py-3.5 xl:flex-row xl:items-center">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded border border-emerald/25 bg-emerald/8 px-2.5 py-1.5 font-mono text-[10px] text-emerald"><Dot /> CLOSED-LOOP ACTIVE</span>
            <span className="inline-flex items-center gap-1.5 rounded border border-emerald/35 bg-emerald/15 px-2.5 py-1.5 font-mono text-[10px] text-emerald font-semibold"><Dot /> LIVE DATASET STREAM · 30-DAY MONTH (MARCH 2022)</span>
            <span className="inline-flex items-center gap-1.5 rounded border border-cyan/25 bg-cyan/8 px-2.5 py-1.5 font-mono text-[10px] text-cyan"><LockKeyhole className="size-3" /> READ-ONLY TELEMETRY FEED</span>
            <span className="inline-flex items-center gap-1.5 rounded border border-cyan/25 bg-cyan/8 px-2.5 py-1.5 font-mono text-[10px] text-cyan"><ShieldCheck className="size-3" /> REGULATORY WATER CAP ENFORCED</span>
            <span className="inline-flex items-center gap-1.5 rounded border border-indigo/25 bg-indigo/8 px-2.5 py-1.5 font-mono text-[10px] text-indigo"><Activity className="size-3" /> 145ms TICK RESPONSE</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[10px]">
            <span className="flex items-center gap-1.5 text-muted-foreground"><CloudSun className="size-3.5 text-amber" /> AMBIENT <b className="font-medium text-foreground">{state.ambient.toFixed(1)}°C</b></span>
            <span className="flex items-center gap-1.5 text-muted-foreground"><Droplets className="size-3.5 text-cyan" /> RH <b className="font-medium text-foreground">{state.humidity}%</b></span>
            <span className="flex items-center gap-1.5 text-muted-foreground"><Waves className="size-3.5 text-indigo" /> STULL T<sub>WET</sub> <b className="font-medium text-foreground">{state.wetBulb.toFixed(1)}°C</b></span>
          </div>
        </div>
      </header>

      {/* 30-Day Dataset Player Control Bar */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded border border-border bg-surface-raised px-4 py-3 font-mono text-xs">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold text-sky uppercase">DATASET PLAYER (30 DAYS):</span>
          <Button variant="outline" size="sm" onClick={() => setIsPlaying(!isPlaying)} className="h-7 text-[10px]">
            {isPlaying ? '⏸ PAUSE' : '▶ PLAY'}
          </Button>
          <Button variant="outline" size="sm" onClick={() => setCurrentIndex(p => (p + 1) % records.length)} className="h-7 text-[10px]">
            ⏭ STEP
          </Button>
          <Button variant="outline" size="sm" onClick={() => jumpDay(-1)} className="h-7 text-[10px]">
            ⏮ -1D
          </Button>
          <Button variant="outline" size="sm" onClick={() => jumpDay(1)} className="h-7 text-[10px]">
            ⏭ +1D
          </Button>
        </div>

        <div className="flex flex-1 items-center gap-3 min-w-[280px]">
          <span className="text-[10px] text-muted-foreground whitespace-nowrap">TIMELINE (30d):</span>
          <input
            type="range"
            min={0}
            max={records.length - 1}
            value={currentIndex}
            onChange={e => setCurrentIndex(Number(e.target.value))}
            className="w-full accent-sky cursor-pointer"
          />
          <span className="text-[11px] font-semibold text-foreground whitespace-nowrap min-w-[210px]">
            DAY {String(state.day).padStart(2, '0')}/30 · {state.timestamp}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">JUMP:</span>
          <select
            value={state.day}
            onChange={e => jumpToDay(Number(e.target.value))}
            className="rounded border border-border bg-background px-2 py-1 text-[10px] text-foreground"
          >
            <option value={1}>Day 01 (Sub-zero Free Air -4°C)</option>
            <option value={5}>Day 05 (Diurnal Transition)</option>
            <option value={10}>Day 10 (Evaporative Peak)</option>
            <option value={15}>Day 15 (Chiller Heatwave 44°C)</option>
            <option value={20}>Day 20 (High Compute Load 99%)</option>
            <option value={25}>Day 25 (Mixed Gate Transition)</option>
            <option value={30}>Day 30 (Month-End Wrap)</option>
          </select>
          <span className="text-[10px] text-muted-foreground ml-2">SPEED:</span>
          {[1, 5, 20, 60].map(s => (
            <Button
              key={s}
              variant={speed === s ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSpeed(s)}
              className={`h-7 px-2 text-[10px] ${speed === s ? 'bg-sky text-background font-bold' : ''}`}
            >
              {s}x
            </Button>
          ))}
        </div>
      </div>

      <section className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-5" aria-label="Key performance indicators">
        <Metric icon={Thermometer} label="Peak server temperature" value={state.peakTemp.toFixed(1)} unit="°C" color={statusColor} footer={<span className={`flex items-center gap-1.5 font-mono text-${statusColor}`}><Dot color={statusColor as 'emerald' | 'amber' | 'destructive'} /> {status} <span className="text-muted-foreground">· limit 80°C</span></span>}><TinyBar value={state.peakTemp} color={statusColor as 'emerald' | 'amber' | 'destructive'} /></Metric>
        <Metric icon={Droplets} label="Water consumption" value={state.water.toLocaleString()} unit="L/h" footer={<span className="flex items-center gap-1 text-emerald"><ArrowDownRight className="size-3" /> {Math.round(state.saved).toLocaleString()} L saved vs baseline</span>}><TinyBar value={capPercent} color="sky" /></Metric>
        <Metric icon={Waves} label="Water usage effectiveness" value={state.wue.toFixed(2)} unit="L/kWh" color="indigo" footer={<span className="flex items-center gap-1.5"><span className="text-emerald">●</span> ASHRAE TC 9.9 COMPLIANT <Info className="size-3" aria-label="Compliant with data center efficiency guidelines." /></span>}><div className="flex justify-between font-mono text-[9px] text-muted-foreground"><span>TARGET ≤0.60</span><span>LIVE RATIO</span></div></Metric>
        <Metric icon={Zap} label="IT total power" value={state.power.toLocaleString()} unit="kW" color="amber" footer={<span>Load split <span className="font-mono text-foreground">{state.nodes.map(n => `${n.cpu}%`).join(' / ')}</span></span>}><div className="flex h-1.5 gap-0.5 overflow-hidden rounded-full">{state.nodes.map((n, i) => <div key={i} className={`h-full bg-${(['sky', 'indigo', 'emerald'] as const)[i]}`} style={{ width: `${n.cpu / state.nodes.reduce((a, v) => a + v.cpu, 0) * 100}%` }} />)}</div></Metric>
        <Metric icon={Fan} label="Active cooling mode" value={`0${state.mode}`} color="emerald" footer={<span className="font-mono text-emerald">{modeShort[state.mode]} ACTIVE</span>}><div className="truncate text-[11px] text-foreground">{modeNames[state.mode]}</div></Metric>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(360px,1fr)]">
        <div className="panel min-w-0 overflow-hidden"><PanelTitle icon={Server} label="Server rack digital twin" aside={<span className="flex items-center gap-1.5 font-mono text-[10px] text-emerald"><Dot /> 3 NODES ONLINE</span>} />
          <div className="p-4 lg:p-5"><div className="grid gap-4 lg:grid-cols-[180px_minmax(0,1fr)]">
            <div className="grid-field relative flex h-[295px] items-center justify-center overflow-hidden rounded border border-border bg-background/40">
              <div className="absolute left-3 top-3 font-mono text-[9px] tracking-[.12em] text-muted-foreground">RACK / A-01</div><div className="absolute right-3 top-3 flex items-center gap-1 font-mono text-[9px] text-emerald"><Dot /> ONLINE</div>
              <div className="relative w-[112px] rounded border border-muted-foreground/40 bg-surface-raised p-2 shadow-2xl">
                <div className="mb-2 flex items-center justify-between border-b border-border pb-1.5 font-mono text-[8px] text-muted-foreground"><span>COOLFLOW / RACK</span><span>01</span></div>
                {state.nodes.map(n => <Button key={n.id} variant="ghost" aria-label={`Select Node ${n.id} in rack`} onClick={() => setSelectedNode(n.id)} className={`mb-1.5 flex h-[56px] w-full items-center justify-between rounded-sm border px-2 hover:bg-accent/40 ${selectedNode === n.id ? 'border-sky/60 bg-sky/10' : 'border-border bg-background/70'}`}><div className="flex flex-col items-start gap-1.5"><span className="font-mono text-[8px] text-muted-foreground">N0{n.id} / ACTIVE</span><span className="flex gap-1"><i className="size-1 rounded-full bg-emerald glow-emerald" /><i className="size-1 rounded-full bg-sky" /><i className="size-1 rounded-full bg-muted-foreground/40" /></span></div><span className={`font-mono text-xs ${selectedNode === n.id ? 'text-sky' : 'text-foreground'}`}>{n.temp.toFixed(1)}°</span></Button>)}
                <div className="flex justify-center gap-2 py-1"><div className="h-1 w-5 rounded-full bg-muted-foreground/30" /><div className="h-1 w-5 rounded-full bg-muted-foreground/30" /></div>
              </div>
              <div className="absolute bottom-3 left-3 font-mono text-[9px] text-muted-foreground">CLUSTER 01 / 03</div>
            </div>
            <div className="min-w-0">
              <div className="mb-3 grid gap-2 sm:grid-cols-3 lg:grid-cols-1 2xl:grid-cols-3">
                {state.nodes.map(n => <NodeCard key={n.id} node={n} selected={selectedNode === n.id} onSelect={() => setSelectedNode(n.id)} />)}
              </div>
              <div className="rounded border border-border bg-background/35 p-3">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2"><CircleGauge className="size-3.5 text-indigo" /><span className="text-[10px] font-bold uppercase tracking-[.1em] text-foreground">CQR temperature envelope</span><span className="font-mono text-[9px] text-muted-foreground">/ NODE 0{node.id}</span></div>
                  <span className="rounded border border-emerald/25 bg-emerald/10 px-1.5 py-0.5 font-mono text-[9px] text-emerald">{node.confidence}% CONFIDENCE</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div><p className="font-mono text-[9px] text-muted-foreground">T_LOW</p><p className="mt-1 font-mono text-base text-indigo">{node.low.toFixed(1)}°</p></div>
                  <div><p className="font-mono text-[9px] text-muted-foreground">T_MID</p><p className="mt-1 font-mono text-base text-sky">{node.mid.toFixed(1)}°</p></div>
                  <div><p className="font-mono text-[9px] text-muted-foreground">T_HIGH · 97.5%</p><p className="mt-1 font-mono text-base text-amber">{node.high.toFixed(1)}°</p></div>
                </div>
                <div className="relative mt-3 h-1.5 rounded-full bg-secondary">
                  <div className="absolute left-[20%] right-[20%] h-full rounded-full bg-gradient-to-r from-indigo via-sky to-amber" />
                  <div className="absolute left-1/2 top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background bg-sky" />
                </div>
              </div>
            </div>
          </div></div>
        </div>

        <div className="panel min-w-0 overflow-hidden">
          <PanelTitle icon={CloudSun} label="Psychrometric mode manager" aside={<span className="font-mono text-[9px] text-muted-foreground">AUTO / CLOSED-LOOP</span>} />
          <div className="p-4 lg:p-5">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">Stull wet-bulb temperature</p>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="font-mono text-[36px] leading-none text-foreground">{state.wetBulb.toFixed(1)}<span className="text-xl text-muted-foreground">°C</span></span>
                  <span className="font-mono text-[10px] text-muted-foreground">/ 13°C GATE</span>
                </div>
              </div>
              <span className={`rounded border px-2 py-1 font-mono text-[10px] ${state.mode === 0 ? 'border-emerald/35 bg-emerald/10 text-emerald' : 'border-amber/35 bg-amber/10 text-amber'}`}>
                {state.mode === 0 ? 'GATE OPEN (FREE AIR)' : 'GATE CLOSED'}
              </span>
            </div>
            <div className="relative mb-5 mt-5 h-2 rounded-full bg-secondary">
              <div className="absolute inset-y-0 left-0 w-[43%] rounded-l-full bg-emerald/80" />
              <div className="absolute inset-y-0 left-[43%] w-[27%] bg-sky/75" />
              <div className="absolute inset-y-0 left-[70%] right-0 rounded-r-full bg-indigo/75" />
              <div className="absolute -top-1.5 size-5 -translate-x-1/2 rounded-full border-[3px] border-background bg-foreground shadow-lg transition-all duration-500" style={{ left: `${Math.min(95, Math.max(5, (state.wetBulb + 5) / 35 * 100))}%` }} />
              <div className="absolute -top-4 left-[43%] h-5 w-px bg-foreground/40" />
            </div>
            <div className="mb-5 flex justify-between font-mono text-[9px] text-muted-foreground">
              <span>0°C FREE-AIR</span><span>13°C GATE</span><span>22°C CHILLER</span>
            </div>
            <div className="space-y-1.5">
              {modeNames.map((mode, i) => (
                <div key={mode} className={`flex items-center justify-between gap-3 rounded border px-3 py-2.5 ${state.mode === i ? 'border-sky/50 bg-sky/10' : 'border-border bg-background/25'}`}>
                  <div className="flex items-center gap-2.5">
                    <span className={`flex size-5 items-center justify-center rounded-sm font-mono text-[10px] ${state.mode === i ? 'bg-sky text-background font-bold' : 'bg-secondary text-muted-foreground'}`}>0{i}</span>
                    <span className={`text-[11px] ${state.mode === i ? 'text-foreground font-semibold' : 'text-muted-foreground'}`}>{mode}</span>
                  </div>
                  <span className={`shrink-0 font-mono text-[9px] ${state.mode === i ? 'text-sky font-semibold' : 'text-muted-foreground'}`}>
                    {i === 0 ? '0 L/h' : state.mode === i ? `${state.water} L/h` : 'STANDBY'}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-5 border-t border-border pt-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[.1em] text-muted-foreground"><LockKeyhole className="size-3 text-emerald" /> WATER WITHDRAWAL CAP</span>
                <span className={`font-mono text-[10px] ${state.water <= state.cap ? 'text-emerald' : 'text-destructive'}`}>{state.water <= state.cap ? 'COMPLIANT' : 'EXCEEDED'}</span>
              </div>
              <TinyBar value={capPercent} color={state.water <= state.cap ? 'emerald' : 'destructive'} />
              <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>{state.water} / {state.cap} L/h</span>
                <span>{remaining.toFixed(0)} L/h HEADROOM</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(360px,1fr)]">
        <div className="panel min-w-0 overflow-hidden">
          <PanelTitle icon={Activity} label="Thermal & resource telemetry" aside={<span className="font-mono text-[9px] text-muted-foreground">ROLLING 4-HOUR DATASET WINDOW</span>} />
          <div className="px-3 pb-4 pt-5 sm:px-5">
            <div className="mb-3 flex flex-wrap gap-4 px-2 font-mono text-[9px] uppercase text-muted-foreground">
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-sky" /> Peak temp °C</span>
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-emerald" /> Water L/h</span>
            </div>
            <div className="h-[235px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={state.history} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="tempFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#38bdf8" stopOpacity={0.25} /><stop offset="100%" stopColor="#38bdf8" stopOpacity={0} /></linearGradient>
                    <linearGradient id="waterFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#10b981" stopOpacity={0.15} /><stop offset="100%" stopColor="#10b981" stopOpacity={0} /></linearGradient>
                  </defs>
                  <CartesianGrid stroke={css('grid')} vertical={false} />
                  <XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: css('muted-foreground'), fontSize: 9, fontFamily: 'IBM Plex Mono' }} interval={3} />
                  <YAxis yAxisId="temp" domain={[45, 90]} tickLine={false} axisLine={false} tick={{ fill: css('muted-foreground'), fontSize: 9, fontFamily: 'IBM Plex Mono' }} />
                  <YAxis yAxisId="water" orientation="right" domain={[0, 650]} hide />
                  <Tooltip content={<ChartTooltip />} />
                  <Area yAxisId="temp" type="monotone" dataKey="temp" name="Peak temp °C" stroke="#38bdf8" strokeWidth={2} fill="url(#tempFill)" activeDot={{ r: 4 }} />
                  <Area yAxisId="water" type="monotone" dataKey="water" name="Water L/h" stroke="#10b981" strokeWidth={2} fill="url(#waterFill)" activeDot={{ r: 4 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="panel min-w-0 overflow-hidden">
          <PanelTitle icon={Layers3} label="Multi-objective Pareto frontier" aside={<span className="font-mono text-[9px] text-muted-foreground">CHEBYSHEV SELECTOR</span>} />
          <div className="px-4 pb-4 pt-4 lg:px-5">
            <div className="mb-2 flex items-center gap-2"><span className="size-1.5 rounded-full bg-sky" /><span className="font-mono text-[9px] text-muted-foreground">EFFICIENT ACTIONS</span><span className="ml-2 size-1.5 rotate-45 bg-amber" /><span className="font-mono text-[9px] text-muted-foreground">KNEE POINT</span></div>
            <div className="relative h-[225px] w-full border-b border-l border-border bg-background/25 grid-field">
              <div className="pointer-events-none absolute bottom-3 left-3 font-mono text-[9px] text-muted-foreground">LOW COST</div>
              <div className="pointer-events-none absolute right-2 top-2 font-mono text-[9px] text-muted-foreground">HIGH PENALTY</div>
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full overflow-visible" aria-label="Pareto frontier chart">
                <polyline points={pareto.map(p => `${p.x},${100-p.y}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth=".55" vectorEffect="non-scaling-stroke" opacity=".75" />
                {pareto.map((p, i) => (
                  <g key={i} role="button" tabIndex={0} aria-label={`Inspect Pareto point ${i + 1}`} onClick={() => setActivePareto(i)} className="cursor-pointer">
                    <circle cx={p.x} cy={100-p.y} r="5" fill="transparent" />
                    <circle cx={p.x} cy={100-p.y} r={i === activePareto ? '1.8' : '1.1'} fill={i === activePareto ? '#f59e0b' : '#38bdf8'} stroke="#0b0f19" strokeWidth=".35" />
                  </g>
                ))}
              </svg>
              <div className="pointer-events-none absolute left-[50%] top-[60%] -translate-x-1/2 rounded border border-amber/40 bg-background/95 px-2 py-1 font-mono text-[9px] text-amber">KNEE POINT</div>
            </div>
            <div className="mt-2 flex justify-between font-mono text-[9px] uppercase text-muted-foreground"><span>Resource cost →</span><span>Thermal penalty ↓</span></div>
            <div className="mt-4 grid grid-cols-3 divide-x divide-border rounded border border-border bg-background/35 py-2 text-center">
              <div><p className="font-mono text-[9px] text-muted-foreground">RESOURCE COST</p><p className="mt-1 font-mono text-[12px] text-sky">${((pareto[activePareto] ?? pareto[0]).x * .0018).toFixed(3)}</p><p className="font-mono text-[8px] text-muted-foreground">/kWh + /L</p></div>
              <div><p className="font-mono text-[9px] text-muted-foreground">THERMAL PENALTY</p><p className="mt-1 font-mono text-[12px] text-amber">{((pareto[activePareto] ?? pareto[0]).y * .028).toFixed(2)}</p><p className="font-mono text-[8px] text-muted-foreground">excess °C²</p></div>
              <div><p className="font-mono text-[9px] text-muted-foreground">SLA LOSS</p><p className="mt-1 font-mono text-[12px] text-indigo">{(pareto[activePareto] ?? pareto[0]).sla.toFixed(1)}%</p><p className="font-mono text-[8px] text-muted-foreground">performance</p></div>
            </div>
          </div>
        </div>
      </section>

      <section className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(360px,1fr)]">
        <div className="panel min-w-0 overflow-hidden">
          <PanelTitle icon={BrainCircuit} label="Explainability engine" aside={<span className="flex items-center gap-1 font-mono text-[9px] text-indigo"><Sparkles className="size-3" /> TREESHAP INSIGHTS</span>} />
          <div className="grid gap-5 p-4 sm:grid-cols-2 lg:p-5">
            <div>
              <div className="mb-4 flex items-center justify-between"><span className="font-mono text-[10px] uppercase text-muted-foreground">Temperature forecast · Node 0{node.id}</span><span className="font-mono text-[9px] text-sky">WHY?</span></div>
              <div className="space-y-4">{forecast.map(item => <div key={item.name}><div className="mb-1.5 flex items-center justify-between gap-2 text-[10px]"><span className="text-foreground">{item.name}</span><span className={`font-mono text-${item.color}`}>{item.detail}</span></div><TinyBar value={item.value} color={item.color as 'cyan' | 'emerald' | 'amber' | 'indigo'} /></div>)}</div>
            </div>
            <div className="border-t border-border pt-5 sm:border-l sm:border-t-0 sm:pl-5 sm:pt-0">
              <p className="mb-4 font-mono text-[10px] uppercase text-muted-foreground">Workload priority classification</p>
              <div className="rounded border border-indigo/30 bg-indigo/8 p-3">
                <div className="flex items-center justify-between"><span className="text-xs font-semibold text-foreground">Priority: {state.workload >= 80 ? 'High' : 'Balanced'}</span><span className="rounded bg-indigo/15 px-1.5 py-0.5 font-mono text-[9px] text-indigo">{state.workload >= 80 ? 'P1' : 'P2'}</span></div>
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">{state.workload >= 80 ? 'Elevated compute demand raises thermal risk. Cooling response is prioritized to protect SLA.' : 'Current compute demand remains within balanced operating envelope. Cooling and water use are co-optimized.'}</p>
              </div>
              <div className="mt-4 space-y-3">
                <div className="flex justify-between text-[10px]"><span className="text-muted-foreground">Compute saturation</span><span className="font-mono text-sky">{state.workload}%</span></div>
                <TinyBar value={state.workload} color="sky" />
                <div className="flex justify-between text-[10px]"><span className="text-muted-foreground">Thermal headroom</span><span className="font-mono text-emerald">{Math.max(0, 80 - state.peakTemp).toFixed(1)}°C</span></div>
                <TinyBar value={Math.max(0, (80 - state.peakTemp) * 10)} color="emerald" />
              </div>
            </div>
          </div>
        </div>

        <div className="panel min-w-0 overflow-hidden">
          <PanelTitle icon={Eye} label="Autonomous Environmental Observer (View Only)" aside={<span className="shrink-0 font-mono text-[9px] text-sky">OBSERVATION MODE</span>} />
          <div className="p-4 lg:p-5">
            <div className="space-y-5">
              {[
                { label: 'Ambient temperature', value: state.ambient, max: 45, suffix: '°C', icon: Thermometer, decimals: 1 },
                { label: 'Relative humidity', value: state.humidity, max: 100, suffix: '%', icon: Droplets, decimals: 0 },
                { label: 'Workload demand', value: state.workload, max: 100, suffix: '%', icon: Gauge, decimals: 0 },
              ].map(item => (
                <div key={item.label}>
                  <div className="mb-2 flex items-center justify-between gap-2"><span className="flex items-center gap-2 text-[11px] text-muted-foreground"><item.icon className="size-3.5 text-sky" /> {item.label}</span><span className="font-mono text-xs text-foreground">{item.value.toFixed(item.decimals)}{item.suffix}</span></div>
                  <div role="meter" aria-label={item.label} aria-valuemin={0} aria-valuemax={item.max} aria-valuenow={item.value} className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                    <div className="h-full rounded-full transition-all duration-500 bg-sky" style={{ width: `${Math.min(100, Math.max(0, item.value / item.max * 100))}%` }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 flex items-start gap-2.5 rounded-sm border border-sky/30 bg-sky/8 px-3 py-3 text-sky">
              <LockKeyhole className="mt-0.5 size-4 shrink-0" />
              <span className="font-mono text-[10px] font-semibold uppercase leading-5">READ-ONLY MODE · CLOSED-LOOP ENGINE RUNS AUTONOMOUSLY (145ms TICK)</span>
            </div>
            <div className="mt-5 border-t border-border pt-4">
              <div className="mb-3 flex items-center justify-between gap-2">
                <p className="font-mono text-[10px] uppercase tracking-[.1em] text-muted-foreground">Autonomous observation trail</p>
                <span className="font-mono text-[9px] text-muted-foreground">#{state.tick}</span>
              </div>
              <div className="max-h-[124px] space-y-2 overflow-auto dashboard-scroll" aria-live="polite">
                {log.map((entry, i) => (
                  <div key={`${i}-${entry}`} className="flex gap-2 font-mono text-[10px] leading-relaxed">
                    <span className={i === 0 ? 'text-emerald' : 'text-muted-foreground'}>{i === 0 ? '›' : '·'}</span>
                    <span className={i === 0 ? 'text-foreground' : 'text-muted-foreground'}>{entry}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="mt-7 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 font-mono text-[9px] uppercase tracking-[.12em] text-muted-foreground">
        <span>COOLFLOW / AUTONOMOUS RESOURCE INTELLIGENCE</span>
        <span className="flex items-center gap-1.5">
          <ShieldCheck className="size-3 text-emerald" /> VIEW-ONLY OBSERVATION <span className="mx-2 text-border">/</span> 30-DAY TELEMETRY: EQCAM_Dataset_FINAL_with_RC (MARCH 2022)
        </span>
      </footer>
    </div>
  </div>;
}
export default Dashboard;
