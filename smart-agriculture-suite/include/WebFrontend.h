#pragma once

#include <Arduino.h>

namespace agri {

static const char kWebHtml[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智润智慧农业</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:16px 20px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.header h1{font-size:20px;color:#38bdf8}
.header .status{font-size:12px;padding:4px 10px;border-radius:12px}
.status.ok{background:#166534;color:#4ade80}
.status.err{background:#7f1d1d;color:#fca5a5}
.tabs{display:flex;gap:4px;padding:8px 16px;background:#1e293b;overflow-x:auto}
.tab{padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;color:#94a3b8;transition:.2s;white-space:nowrap}
.tab:hover{background:#334155}
.tab.active{background:#1e40af;color:#fff}
.content{padding:16px;max-width:1000px;margin:0 auto}
.card{background:#1e293b;border-radius:12px;padding:16px;margin-bottom:12px;border:1px solid #334155}
.card h3{font-size:14px;color:#94a3b8;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
.grid{display:grid;gap:12px}
.grid-2{grid-template-columns:1fr 1fr}
.grid-3{grid-template-columns:1fr 1fr 1fr}
@media(max-width:600px){.grid-2,.grid-3{grid-template-columns:1fr}}
.sensor-card{background:#0f172a;border-radius:8px;padding:12px;text-align:center;border:1px solid #334155}
.sensor-card .label{font-size:11px;color:#64748b;margin-bottom:4px}
.sensor-card .value{font-size:24px;font-weight:700;color:#38bdf8}
.sensor-card .unit{font-size:12px;color:#64748b}
.sensor-card.fault{border-color:#dc2626;background:#1a0505}
.sensor-card.fault .value{color:#f87171}
.sensor-card.fault .fault-tag{display:inline-block}
.fault-tag{display:none;background:#dc2626;color:#fff;font-size:10px;padding:2px 6px;border-radius:4px;margin-top:4px}
.btn{padding:8px 16px;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;transition:.2s}
.btn-primary{background:#2563eb;color:#fff}
.btn-primary:hover{background:#1d4ed8}
.btn-danger{background:#dc2626;color:#fff}
.btn-success{background:#16a34a;color:#fff}
.btn-sm{padding:6px 12px;font-size:12px}
.irr-card{padding:12px;border-radius:8px;background:#0f172a;border:1px solid #334155;margin-bottom:8px}
.irr-card .irr-label{font-size:12px;color:#64748b}
.irr-card .irr-val{font-size:16px;font-weight:600;color:#f0fdf4}
.disease-card{padding:16px;border-radius:8px;text-align:center}
.disease-healthy{background:#052e16;border:1px solid #16a34a}
.disease-warning{background:#422006;border:1px solid #f59e0b}
.disease-danger{background:#450a0a;border:1px solid #dc2626}
.stage-bar{display:flex;gap:2px;margin:12px 0}
.stage-seg{flex:1;height:8px;border-radius:4px;background:#334155}
.stage-seg.done{background:#22c55e}
.stage-seg.current{background:#3b82f6;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
select,input{background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:6px 10px;font-size:13px}
.form-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.form-row label{min-width:80px;font-size:12px;color:#94a3b8}
.toggle{position:relative;width:44px;height:24px;background:#334155;border-radius:12px;cursor:pointer;transition:.3s}
.toggle.on{background:#2563eb}
.toggle::after{content:'';position:absolute;top:2px;left:2px;width:20px;height:20px;background:#fff;border-radius:50%;transition:.3s}
.toggle.on::after{left:22px}
.hidden{display:none}
.prediction{font-size:12px;color:#94a3b8;margin-top:4px}
</style>
</head>
<body>
<div class="header">
 <h1>智润 Smart Agriculture</h1>
 <span id="connStatus" class="status ok">已连接</span>
</div>
<div class="tabs">
 <div class="tab active" data-tab="dashboard">仪表盘</div>
 <div class="tab" data-tab="irrigation">灌溉控制</div>
 <div class="tab" data-tab="world">世界模型</div>
 <div class="tab" data-tab="growth">生长追踪</div>
 <div class="tab" data-tab="config">系统配置</div>
</div>

<div class="content">
 <!-- 仪表盘 -->
 <div id="tab-dashboard">
  <div class="card">
   <h3>传感器数据</h3>
   <div class="grid grid-3" id="sensorGrid">
    <div class="sensor-card" id="sc-airTemp">
     <div class="label">空气温度</div>
     <div class="value">--</div><div class="unit">°C</div>
     <div class="fault-tag">故障</div>
    </div>
    <div class="sensor-card" id="sc-airHumi">
     <div class="label">空气湿度</div>
     <div class="value">--</div><div class="unit">%</div>
     <div class="fault-tag">故障</div>
    </div>
    <div class="sensor-card" id="sc-soilHumi">
     <div class="label">土壤湿度</div>
     <div class="value">--</div><div class="unit">%</div>
     <div class="fault-tag">故障</div>
    </div>
    <div class="sensor-card" id="sc-liquid">
     <div class="label">液位</div>
     <div class="value">--</div><div class="unit">%</div>
     <div class="fault-tag">故障</div>
    </div>
    <div class="sensor-card" id="sc-light">
     <div class="label">光照强度</div>
     <div class="value">--</div><div class="unit">lux</div>
     <div class="fault-tag">故障</div>
    </div>
    <div class="sensor-card" id="sc-actuator">
     <div class="label">执行器</div>
     <div class="value" style="font-size:16px">--</div>
     <div class="unit" id="actuatorSrc">--</div>
    </div>
   </div>
  </div>
  <div class="card">
   <h3>当前决策</h3>
   <div id="decisionInfo" style="font-size:14px;color:#94a3b8">加载中...</div>
  </div>
 </div>

 <!-- 灌溉控制 -->
 <div id="tab-irrigation" class="hidden">
  <div class="card">
   <h3>灌溉控制</h3>
   <div class="form-row">
    <label>自动模式</label>
    <div id="autoToggle" class="toggle on" onclick="toggleAuto()"></div>
   </div>
   <div id="manualPanel" class="hidden" style="margin-top:12px">
    <div class="form-row">
     <label>手动控制</label>
     <button class="btn btn-sm btn-success" onclick="manualCtrl('pump',true)">开启</button>
     <button class="btn btn-sm btn-danger" onclick="manualCtrl('pump',false)">关闭</button>
    </div>
   </div>
  </div>
  <div class="card">
   <h3>灌溉阈值配置</h3>
   <div class="form-row"><label>白天温度></label><input id="cfg-dayTemp" type="number" step="0.1" value="20"></div>
   <div class="form-row"><label>白天湿度<</label><input id="cfg-dayHumi" type="number" step="0.1" value="60"></div>
   <div class="form-row"><label>白天土壤<</label><input id="cfg-daySoil" type="number" step="0.1" value="50"></div>
   <div class="form-row"><label>夜间温度></label><input id="cfg-nightTemp" type="number" step="0.1" value="15"></div>
   <div class="form-row"><label>夜间湿度<</label><input id="cfg-nightHumi" type="number" step="0.1" value="70"></div>
   <div class="form-row"><label>夜间土壤<</label><input id="cfg-nightSoil" type="number" step="0.1" value="45"></div>
    <button class="btn btn-primary" onclick="saveIrrConfig()" style="margin-top:8px">保存配置</button>
  </div>
 </div>

 <!-- 世界模型 -->
 <div id="tab-world" class="hidden">
  <div class="card">
   <h3>病害诊断</h3>
   <div id="diseaseCard" class="disease-card disease-healthy">
    <div style="font-size:20px;font-weight:700" id="diseaseName">等待模型响应...</div>
    <div style="font-size:13px;margin-top:4px" id="diseaseConf">--</div>
    <div style="font-size:12px;color:#94a3b8;margin-top:8px" id="diseaseTreatment"></div>
   </div>
  </div>
  <div class="card">
   <h3>灌溉决策 (世界模型)</h3>
   <div class="irr-card">
    <div class="irr-label">推荐动作</div>
    <div class="irr-val" id="wmAction">--</div>
   </div>
   <div class="irr-card">
    <div class="irr-label">持续时间</div>
    <div class="irr-val" id="wmDuration">--</div>
   </div>
   <div class="irr-card">
    <div class="irr-label">置信度</div>
    <div class="irr-val" id="wmConfidence">--</div>
   </div>
   <div class="irr-card">
    <div class="irr-label">决策原因</div>
    <div class="irr-val" id="wmReason" style="font-size:13px">--</div>
   </div>
  </div>
  <div class="card">
   <h3>环境预测</h3>
   <div class="grid grid-2">
    <div class="irr-card"><div class="irr-label">预测土壤湿度</div><div class="irr-val" id="wmPredSoil">--</div></div>
    <div class="irr-card"><div class="irr-label">预测空气温度</div><div class="irr-val" id="wmPredTemp">--</div></div>
    <div class="irr-card"><div class="irr-label">预测空气湿度</div><div class="irr-val" id="wmPredHumi">--</div></div>
    <div class="irr-card"><div class="irr-label">环境风险</div><div class="irr-val" id="wmRisk">--</div></div>
   </div>
  </div>
  <div class="card">
   <h3>Atlas 服务器</h3>
   <div class="form-row"><label>地址</label><input id="atlasHost" type="text" placeholder="192.168.1.100"></div>
   <div class="form-row"><label>端口</label><input id="atlasPort" type="number" value="8080"></div>
   <button class="btn btn-primary" onclick="saveAtlas()" style="margin-top:8px">保存并连接</button>
   <div id="atlasStatus" style="font-size:12px;margin-top:8px;color:#94a3b8"></div>
  </div>
 </div>

 <!-- 生长追踪 -->
 <div id="tab-growth" class="hidden">
  <div class="card">
   <h3>作物选择</h3>
   <div class="form-row">
    <label>当前作物</label>
    <select id="cropSelect" onchange="setCrop()">
     <option value="0">番茄</option>
     <option value="1">生菜</option>
     <option value="2">辣椒</option>
     <option value="3">黄瓜</option>
     <option value="4">草莓</option>
    </select>
   </div>
   <button class="btn btn-danger btn-sm" onclick="resetGrowth()" style="margin-top:8px">重置生长数据</button>
  </div>
  <div class="card">
   <h3>生长状态</h3>
   <div class="grid grid-2">
    <div class="irr-card"><div class="irr-label">当前阶段</div><div class="irr-val" id="growStage">--</div></div>
    <div class="irr-card"><div class="irr-label">生长天数</div><div class="irr-val" id="growDay">--</div></div>
    <div class="irr-card"><div class="irr-label">累计GDD</div><div class="irr-val" id="growGDD">--</div></div>
    <div class="irr-card"><div class="irr-label">产量评分</div><div class="irr-val" id="growYield">--</div></div>
   </div>
   <div class="stage-bar" id="stageBar"></div>
   <div class="irr-card" style="margin-top:8px">
    <div class="irr-label">灌溉建议</div>
    <div class="irr-val" id="growAdvice" style="font-size:13px">--</div>
   </div>
  </div>
 </div>

 <!-- 系统配置 -->
 <div id="tab-config" class="hidden">
  <div class="card">
   <h3>WiFi 配置</h3>
   <div class="form-row"><label>SSID</label><input id="wifiSsid" type="text"></div>
   <div class="form-row"><label>密码</label><input id="wifiPass" type="password"></div>
   <button class="btn btn-primary" onclick="saveWifi()" style="margin-top:8px">保存WiFi</button>
  </div>
  <div class="card">
   <h3>系统信息</h3>
   <div id="sysInfo" style="font-size:13px;color:#94a3b8">加载中...</div>
  </div>
 </div>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
let currentTab = 'dashboard';
let refreshTimer = null;

// Tab switching
$$('.tab').forEach(t => t.addEventListener('click', () => {
 $$('.tab').forEach(x => x.classList.remove('active'));
 t.classList.add('active');
 $$('.content > div').forEach(x => x.classList.add('hidden'));
 $('#tab-' + t.dataset.tab).classList.remove('hidden');
 currentTab = t.dataset.tab;
 refresh();
}));

async function api(path, method = 'GET', body = null) {
 try {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch('/api' + path, opts);
  return await r.json();
 } catch (e) {
  console.error('API error:', e);
  return null;
 }
}

function setSensor(id, value, fault, unit) {
 const el = document.getElementById(id);
 if (!el) return;
 el.querySelector('.value').textContent = fault ? '故障' : value;
 if (fault) el.classList.add('fault');
 else el.classList.remove('fault');
}

async function refresh() {
 // Dashboard
 if (currentTab === 'dashboard' || currentTab === 'world') {
  const s = await api('/status');
  if (!s) {
   $('#connStatus').className = 'status err';
   $('#connStatus').textContent = '断开';
   return;
  }
  $('#connStatus').className = 'status ok';
  $('#connStatus').textContent = '已连接';

  const f = s.faults || {};
  setSensor('sc-airTemp', s.air_temp?.toFixed(1), f.air, '°C');
  setSensor('sc-airHumi', s.air_humi?.toFixed(1), f.air, '%');
  setSensor('sc-soilHumi', s.soil_humi?.toFixed(1), f.soil, '%');
  setSensor('sc-liquid', s.liquid_level?.toFixed(1), f.liquid, '%');
  setSensor('sc-light', s.light?.toFixed(0), f.light, 'lux');

  const act = s.actuator || {};
  $('#sc-actuator .value').textContent = act.valve_on ? '开启' : '关闭';
  $('#actuatorSrc').textContent = act.source || 'idle';

  // World model
  if (s.world_model) {
   const wm = s.world_model;
   const dc = $('#diseaseCard');
   if (wm.disease_id === 0) {
    dc.className = 'disease-card disease-healthy';
    $('#diseaseName').textContent = '健康';
   } else if (wm.disease_confidence > 0.8) {
    dc.className = 'disease-card disease-danger';
    $('#diseaseName').textContent = wm.disease_name || '未知病害';
   } else {
    dc.className = 'disease-card disease-warning';
    $('#diseaseName').textContent = wm.disease_name || '疑似病害';
   }
   $('#diseaseConf').textContent = '置信度: ' + (wm.disease_confidence * 100).toFixed(1) + '%';
   $('#diseaseTreatment').textContent = wm.treatment || '';
   $('#wmAction').textContent = ['关闭', '轻度', '中度', '重度'][wm.action] || '--';
   $('#wmDuration').textContent = wm.duration ? wm.duration + '秒' : '--';
   $('#wmConfidence').textContent = wm.action_confidence ? (wm.action_confidence * 100).toFixed(1) + '%' : '--';
   $('#wmReason').textContent = wm.reason || '--';
   $('#wmPredSoil').textContent = wm.pred_soil ? wm.pred_soil.toFixed(1) + '%' : '--';
   $('#wmPredTemp').textContent = wm.pred_temp ? wm.pred_temp.toFixed(1) + '°C' : '--';
   $('#wmPredHumi').textContent = wm.pred_humi ? wm.pred_humi.toFixed(1) + '%' : '--';
   $('#wmRisk').textContent = wm.risk !== undefined ? (wm.risk * 100).toFixed(0) + '%' : '--';
  }

  // Growth
  if (s.growth) {
   const g = s.growth;
   $('#cropSelect').value = s.crop_id || 0;
   $('#growStage').textContent = g.stage_cn || g.stage || '--';
   $('#growDay').textContent = g.day || '0';
   $('#growGDD').textContent = g.gdd ? g.gdd.toFixed(1) : '0';
   $('#growYield').textContent = g.yield_score ? g.yield_score.toFixed(0) : '--';
   $('#growAdvice').textContent = g.advice || '--';

   // Stage bar
   const stages = ['Seed','Germination','Seedling','Vegetative','Flowering','Fruiting','Maturity'];
   const ci = stages.indexOf(g.stage);
   let bar = '';
   stages.forEach((s, i) => {
    let cls = 'stage-seg';
    if (i < ci) cls += ' done';
    else if (i === ci) cls += ' current';
    bar += '<div class="' + cls + '"></div>';
   });
   $('#stageBar').innerHTML = bar;
  }

  // Actuator
  if (act) {
   $('#autoToggle').className = 'toggle' + (act.auto_mode ? ' on' : '');
   if (!act.auto_mode) $('#manualPanel').classList.remove('hidden');
   else $('#manualPanel').classList.add('hidden');
  }
 }
}

async function toggleAuto() {
 const on = !$('#autoToggle').classList.contains('on');
 await api('/irrigation/mode', 'POST', { auto: on });
 refresh();
}

async function manualCtrl(type, on) {
 await api('/irrigation/' + type, 'POST', { on });
}

async function saveIrrConfig() {
 await api('/irrigation/config', 'POST', {
  day_temp: parseFloat($('#cfg-dayTemp').value),
  day_humi: parseFloat($('#cfg-dayHumi').value),
  day_soil: parseFloat($('#cfg-daySoil').value),
  night_temp: parseFloat($('#cfg-nightTemp').value),
  night_humi: parseFloat($('#cfg-nightHumi').value),
  night_soil: parseFloat($('#cfg-nightSoil').value),
 });
}

async function saveAtlas() {
 await api('/atlas/config', 'POST', {
  host: $('#atlasHost').value,
  port: parseInt($('#atlasPort').value)
 });
 $('#atlasStatus').textContent = '已保存';
}

async function saveWifi() {
 await api('/wifi', 'POST', {
  ssid: $('#wifiSsid').value,
  pass: $('#wifiPass').value
 });
 alert('WiFi配置已保存,重启后生效');
}

async function setCrop() {
 await api('/growth/crop', 'POST', { crop: parseInt($('#cropSelect').value) });
 refresh();
}

async function resetGrowth() {
 if (confirm('确定重置生长数据?')) {
  await api('/growth/reset', 'POST');
  refresh();
 }
}

// Auto refresh
refresh();
refreshTimer = setInterval(refresh, 3000);
</script>
</body>
</html>
)rawliteral";

}  // namespace agri
