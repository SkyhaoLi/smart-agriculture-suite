#pragma once

#include <Arduino.h>
#include <pgmspace.h>

namespace agri {

const char kWebHtml[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智润 - 智慧农业套件</title>
<style>
:root{--bg:#0f1923;--card:#1a2736;--border:#2d3f52;--text:#e0e6ed;--muted:#7a8b9a;--accent:#4fc3f7;--green:#66bb6a;--orange:#ffa726;--red:#ef5350;--blue:#42a5f5}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{background:linear-gradient(135deg,#1a2736 0%,#0f1923 100%);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:18px;font-weight:600;color:var(--accent)}
.header .status{font-size:12px;color:var(--muted)}
.tabs{display:flex;background:var(--card);border-bottom:1px solid var(--border);overflow-x:auto}
.tabs button{flex:none;padding:10px 16px;border:none;background:transparent;color:var(--muted);cursor:pointer;font-size:13px;border-bottom:2px solid transparent;white-space:nowrap}
.tabs button.active{color:var(--accent);border-bottom-color:var(--accent)}
.tabs button:hover{color:var(--text)}
.page{display:none;padding:16px;max-width:800px;margin:0 auto}
.page.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:12px}
.card h3{font-size:14px;color:var(--accent);margin-bottom:10px}
.row{display:flex;gap:8px;flex-wrap:wrap}
.col{flex:1;min-width:120px}
.metric{text-align:center;padding:10px}
.metric .value{font-size:24px;font-weight:700}
.metric .label{font-size:11px;color:var(--muted);margin-top:4px}
.metric.temp .value{color:var(--orange)}
.metric.humi .value{color:var(--blue)}
.metric.soil .value{color:var(--green)}
.metric.light .value{color:#ffd54f}
.metric.liquid .value{color:var(--accent)}
.btn{padding:8px 16px;border:1px solid var(--border);border-radius:6px;background:var(--card);color:var(--text);cursor:pointer;font-size:13px;transition:all .2s}
.btn:hover{background:var(--border)}
.btn.active{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.danger{border-color:var(--red);color:var(--red)}
.btn.danger:hover{background:var(--red);color:#fff}
.btn.success{border-color:var(--green);color:var(--green)}
.btn.success:hover{background:var(--green);color:#fff}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.toggle-group{display:flex;gap:4px;align-items:center}
.toggle-group label{font-size:12px;color:var(--muted);min-width:50px}
.progress-bar{background:var(--bg);border-radius:4px;height:8px;overflow:hidden;margin-top:6px}
.progress-bar .fill{height:100%;border-radius:4px;transition:width .5s}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:500}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge.green{background:rgba(102,187,106,.2);color:var(--green)}
.badge.orange{background:rgba(255,167,38,.2);color:var(--orange)}
.badge.red{background:rgba(239,83,80,.2);color:var(--red)}
.badge.blue{background:rgba(66,165,245,.2);color:var(--blue)}
.alert-list{max-height:300px;overflow-y:auto}
.alert-item{padding:8px;border-left:3px solid var(--muted);margin-bottom:4px;font-size:12px}
.alert-item.warning{border-left-color:var(--orange)}
.alert-item.critical{border-left-color:var(--red)}
.alert-item.info{border-left-color:var(--blue)}
img.preview{max-width:100%;border-radius:8px;margin-top:8px}
.log{background:var(--bg);border-radius:4px;padding:8px;font-family:monospace;font-size:11px;max-height:200px;overflow-y:auto;white-space:pre-wrap;color:var(--green)}
</style>
</head>
<body>
<div class="header">
<h1>智润 Smart Agriculture</h1>
<div class="status" id="connStatus">Connecting...</div>
</div>
<div class="tabs">
<button class="active" onclick="showPage('dashboard')">仪表盘</button>
<button onclick="showPage('irrigation')">灌溉控制</button>
<button onclick="showPage('anomaly')">异常检测</button>
<button onclick="showPage('growth')">生长追踪</button>
<button onclick="showPage('learning')">学习模块</button>
<button onclick="showPage('fusion')">融合决策</button>
<button onclick="showPage('plant')">植物医生</button>
<button onclick="showPage('config')">系统配置</button>
</div>

<div id="page-dashboard" class="page active">
<div class="card">
<h3>传感器数据</h3>
<div class="row">
<div class="col metric temp"><div class="value" id="s-airTemp">--</div><div class="label">温度 (C)</div></div>
<div class="col metric humi"><div class="value" id="s-airHumi">--</div><div class="label">湿度 (%)</div></div>
<div class="col metric soil"><div class="value" id="s-soilHumi">--</div><div class="label">土壤 (%)</div></div>
</div>
<div class="row" style="margin-top:8px">
<div class="col metric light"><div class="value" id="s-light">--</div><div class="label">光照 (lux)</div></div>
<div class="col metric liquid"><div class="value" id="s-liquid">--</div><div class="label">液位 (%)</div></div>
</div>
</div>
<div class="card">
<h3>执行器状态</h3>
<div class="row">
<div class="col"><span class="label">阀门</span> <span id="a-valve" class="badge green">OFF</span></div>
<div class="col"><span class="label">水泵</span> <span id="a-pump" class="badge green">OFF</span></div>
<div class="col"><span class="label">模式</span> <span id="a-mode" class="badge blue">AUTO</span></div>
</div>
<div class="row" style="margin-top:6px">
<div class="col"><span class="label">来源</span> <span id="a-source">idle</span></div>
<div class="col"><span class="label">剩余</span> <span id="a-remain">0s</span></div>
</div>
</div>
<div class="card">
<h3>模块概览</h3>
<table>
<tr><th>模块</th><th>状态</th><th>关键指标</th></tr>
<tr><td>规则引擎</td><td id="m-rule">--</td><td id="m-rule-val">--</td></tr>
<tr><td>异常检测</td><td id="m-anomaly">--</td><td id="m-anomaly-val">--</td></tr>
<tr><td>生长追踪</td><td id="m-growth">--</td><td id="m-growth-val">--</td></tr>
<tr><td>Q-Learning</td><td id="m-learn">--</td><td id="m-learn-val">--</td></tr>
<tr><td>融合决策</td><td id="m-fusion">--</td><td id="m-fusion-val">--</td></tr>
<tr><td>植物医生</td><td id="m-plant">--</td><td id="m-plant-val">--</td></tr>
</table>
</div>
</div>

<div id="page-irrigation" class="page">
<div class="card">
<h3>灌溉模式</h3>
<div class="controls">
<div class="toggle-group">
<label>模式</label>
<button class="btn active" id="btn-auto" onclick="setIrrMode(true)">自动</button>
<button class="btn" id="btn-manual" onclick="setIrrMode(false)">手动</button>
</div>
</div>
</div>
<div class="card">
<h3>手动控制</h3>
<div class="controls">
<button class="btn success" onclick="setPump(true)">开启水泵</button>
<button class="btn danger" onclick="setPump(false)">关闭水泵</button>
<button class="btn success" onclick="setValve(true)">开启阀门</button>
<button class="btn danger" onclick="setValve(false)">关闭阀门</button>
</div>
</div>
<div class="card">
<h3>灌溉配置</h3>
<div class="controls">
<button class="btn" id="btn-rule" onclick="toggleRuleEngine()">规则引擎: ?</button>
</div>
<table id="irr-config-table" style="margin-top:10px">
<tr><th>参数</th><th>日间阈值</th><th>夜间阈值</th></tr>
<tr><td>空气温度</td><td id="irr-dayTemp">--</td><td id="irr-nightTemp">--</td></tr>
<tr><td>空气湿度</td><td id="irr-dayHumi">--</td><td id="irr-nightHumi">--</td></tr>
<tr><td>土壤湿度</td><td id="irr-daySoil">--</td><td id="irr-nightSoil">--</td></tr>
</table>
</div>
</div>

<div id="page-anomaly" class="page">
<div class="card">
<h3>异常检测状态</h3>
<div class="row">
<div class="col metric"><div class="value" id="an-level">normal</div><div class="label">告警级别</div></div>
<div class="col metric"><div class="value" id="an-score">--</div><div class="label">IForest分数</div></div>
</div>
<div class="row" style="margin-top:8px">
<div class="col metric"><div class="value" id="an-samples">--</div><div class="label">总采样</div></div>
<div class="col metric"><div class="value" id="an-anomalies">--</div><div class="label">异常次数</div></div>
</div>
</div>
<div class="card">
<h3>传感器详情</h3>
<table id="an-sensor-table">
<tr><th>传感器</th><th>当前值</th><th>均值</th><th>Z-Score</th><th>状态</th></tr>
</table>
</div>
<div class="card">
<h3>告警记录</h3>
<div class="alert-list" id="an-alerts"></div>
<div class="controls" style="margin-top:8px">
<button class="btn danger" onclick="clearAnomaly()">清除告警</button>
</div>
</div>
</div>

<div id="page-growth" class="page">
<div class="card">
<h3>生长状态</h3>
<div class="row">
<div class="col metric"><div class="value" id="gr-crop">--</div><div class="label">作物</div></div>
<div class="col metric"><div class="value" id="gr-stage">--</div><div class="label">生长阶段</div></div>
</div>
<div class="row" style="margin-top:8px">
<div class="col metric"><div class="value" id="gr-day">--</div><div class="label">生长天数</div></div>
<div class="col metric"><div class="value" id="gr-gdd">--</div><div class="label">累积GDD</div></div>
<div class="col metric"><div class="value" id="gr-yield">--</div><div class="label">产量评分</div></div>
</div>
<div class="progress-bar" style="margin-top:10px"><div class="fill" id="gr-progress" style="width:0%;background:var(--green)"></div></div>
</div>
<div class="card">
<h3>灌溉建议</h3>
<p id="gr-advice" style="font-size:13px;color:var(--muted)">--</p>
</div>
<div class="card">
<h3>生长预测</h3>
<div class="row">
<div class="col"><span class="label">预测花期</span> <span id="gr-flower">--</span></div>
<div class="col"><span class="label">预测成熟</span> <span id="gr-mature">--</span></div>
</div>
</div>
<div class="card">
<h3>作物选择</h3>
<div class="controls" id="gr-crops"></div>
<div class="controls" style="margin-top:8px">
<button class="btn danger" onclick="resetGrowth()">重置生长数据</button>
</div>
</div>
</div>

<div id="page-learning" class="page">
<div class="card">
<h3>Q-Learning 状态</h3>
<div class="row">
<div class="col metric"><div class="value" id="lr-episodes">--</div><div class="label">总回合</div></div>
<div class="col metric"><div class="value" id="lr-epsilon">--</div><div class="label">Epsilon</div></div>
<div class="col metric"><div class="value" id="lr-avgReward">--</div><div class="label">平均奖励</div></div>
</div>
<div class="row" style="margin-top:8px">
<div class="col metric"><div class="value" id="lr-action">--</div><div class="label">上一步动作</div></div>
<div class="col metric"><div class="value" id="lr-recommend">--</div><div class="label">推荐动作</div></div>
</div>
</div>
<div class="card">
<h3>控制</h3>
<div class="controls">
<button class="btn" id="btn-lr-auto" onclick="toggleLearningAuto()">自动控制: ?</button>
<button class="btn success" onclick="feedback(true)">正面反馈</button>
<button class="btn danger" onclick="feedback(false)">负面反馈</button>
<button class="btn danger" onclick="resetLearning()">重置Q表</button>
</div>
</div>
</div>

<div id="page-fusion" class="page">
<div class="card">
<h3>融合决策</h3>
<div class="row">
<div class="col metric"><div class="value" id="fu-decision">--</div><div class="label">决策</div></div>
<div class="col metric"><div class="value" id="fu-confidence">--</div><div class="label">置信度</div></div>
</div>
<div class="row" style="margin-top:8px">
<div class="col metric"><div class="value" id="fu-score">--</div><div class="label">最终评分</div></div>
<div class="col metric"><div class="value" id="fu-total">--</div><div class="label">总决策数</div></div>
</div>
</div>
<div class="card">
<h3>传感器通道</h3>
<table id="fu-sensor-table">
<tr><th>传感器</th><th>原始值</th><th>滤波值</th><th>权重</th><th>可靠性</th></tr>
</table>
</div>
<div class="card">
<h3>神经网络输出</h3>
<div class="row">
<div class="col metric"><div class="value" id="nn-none">--</div><div class="label">None</div></div>
<div class="col metric"><div class="value" id="nn-moderate">--</div><div class="label">Moderate</div></div>
<div class="col metric"><div class="value" id="nn-heavy">--</div><div class="label">Heavy</div></div>
</div>
</div>
<div class="controls">
<button class="btn" id="btn-fu-auto" onclick="toggleFusionAuto()">自动控制: ?</button>
</div>
</div>

<div id="page-plant" class="page">
<div class="card">
<h3>植物医生</h3>
<div class="row">
<div class="col"><span class="label">启用</span> <span id="pd-enabled" class="badge">--</span></div>
<div class="col"><span class="label">摄像头</span> <span id="pd-camera" class="badge">--</span></div>
<div class="col"><span class="label">模型</span> <span id="pd-model" class="badge">--</span></div>
</div>
</div>
<div class="card">
<h3>检测结果</h3>
<div class="row">
<div class="col metric"><div class="value" id="pd-disease">--</div><div class="label">病害</div></div>
<div class="col metric"><div class="value" id="pd-conf">--</div><div class="label">置信度</div></div>
</div>
<p id="pd-treatment" style="margin-top:8px;font-size:13px;color:var(--muted)">--</p>
</div>
<div class="card">
<h3>图像预览</h3>
<img id="pd-preview" class="preview" src="" alt="No image" style="display:none">
<div class="controls" style="margin-top:8px">
<button class="btn success" onclick="capturePlant()">拍照检测</button>
<button class="btn" onclick="detectPlant()">仅检测</button>
</div>
</div>
</div>

<div id="page-config" class="page">
<div class="card">
<h3>WiFi 配置</h3>
<div style="margin-top:8px">
<label style="font-size:12px;color:var(--muted)">SSID</label><br>
<input type="text" id="cfg-ssid" style="width:100%;padding:6px;margin:4px 0;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px">
<label style="font-size:12px;color:var(--muted)">密码</label><br>
<input type="password" id="cfg-pass" style="width:100%;padding:6px;margin:4px 0;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px">
<div class="controls" style="margin-top:8px">
<button class="btn success" onclick="saveWifi()">保存WiFi配置</button>
</div>
</div>
</div>
<div class="card">
<h3>OTA 固件更新</h3>
<div style="margin-top:8px">
<label style="font-size:12px;color:var(--muted)">固件URL</label><br>
<input type="text" id="cfg-ota-url" placeholder="http://..." style="width:100%;padding:6px;margin:4px 0;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px">
<div class="controls" style="margin-top:8px">
<button class="btn danger" onclick="startOTA()">开始OTA更新</button>
</div>
</div>
</div>
<div class="card">
<h3>模块开关</h3>
<div class="controls" style="flex-direction:column;align-items:flex-start;gap:8px">
<div class="toggle-group">
<label style="min-width:80px">规则引擎</label>
<button class="btn" id="cfg-rule" onclick="toggleModule('ruleEngineEnabled')">--</button>
</div>
<div class="toggle-group">
<label style="min-width:80px">学习自动</label>
<button class="btn" id="cfg-learn" onclick="toggleModule('learningAutoEnabled')">--</button>
</div>
<div class="toggle-group">
<label style="min-width:80px">融合自动</label>
<button class="btn" id="cfg-fusion" onclick="toggleModule('fusionAutoEnabled')">--</button>
</div>
<div class="toggle-group">
<label style="min-width:80px">植物医生</label>
<button class="btn" id="cfg-plant" onclick="toggleModule('plantDoctorEnabled')">--</button>
</div>
</div>
</div>
<div class="card">
<h3>危险操作</h3>
<div class="controls">
<button class="btn danger" onclick="factoryReset()">恢复出厂设置</button>
</div>
</div>
<div class="card">
<h3>系统信息</h3>
<table>
<tr><td>硬件档位</td><td id="cfg-hw">--</td></tr>
<tr><td>WiFi</td><td id="cfg-wifi">--</td></tr>
<tr><td>IP地址</td><td id="cfg-ip">--</td></tr>
</table>
</div>
</div>

<script>
let lastData=null;
function showPage(id){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));document.getElementById('page-'+id).classList.add('active');event.target.classList.add('active')}
async function api(path,opts){try{const r=await fetch('/api/'+path,opts||{});return await r.json()}catch(e){console.error(e);return null}}
function badge(text,cls){return `<span class="badge ${cls||'blue'}">${text}</span>`}
function f1(v,d){return typeof v==='number'?v.toFixed(d===undefined?1:d):'--'}
function updateSensors(s){if(!s)return;document.getElementById('s-airTemp').textContent=f1(s.airTemp);document.getElementById('s-airHumi').textContent=f1(s.airHumi);document.getElementById('s-soilHumi').textContent=f1(s.soilHumi);document.getElementById('s-light').textContent=Math.round(s.lightValue||0);document.getElementById('s-liquid').textContent=f1(s.liquidLevel)}
function updateActuator(a){if(!a)return;document.getElementById('a-valve').textContent=a.valveOn?'ON':'OFF';document.getElementById('a-valve').className='badge '+(a.valveOn?'green':'blue');document.getElementById('a-pump').textContent=a.pumpOn?'ON':'OFF';document.getElementById('a-pump').className='badge '+(a.pumpOn?'green':'blue');document.getElementById('a-mode').textContent=a.autoMode?'AUTO':'MANUAL';document.getElementById('a-source').textContent=a.source||'idle';document.getElementById('a-remain').textContent=(a.secondsRemaining||0)+'s'}
function updateModules(m){if(!m)return;const on=v=>v?'<span class="badge green">ON</span>':'<span class="badge red">OFF</span>';document.getElementById('m-rule').innerHTML=on(m.ruleEngineEnabled);document.getElementById('m-rule-val').textContent=m.ruleEngineEnabled?'活跃':'关闭';const al=m.anomaly||{};document.getElementById('m-anomaly').innerHTML=badge(al.alertLevelName||'normal','blue');document.getElementById('m-anomaly-val').textContent='score:'+f1(al.iforestScore);const gr=m.growth||{};document.getElementById('m-growth').innerHTML=badge(gr.stageNameCn||'--','green');document.getElementById('m-growth-val').textContent='Day:'+gr.dayOfGrowth;const lr=m.learning||{};document.getElementById('m-learn').innerHTML=lr.autoControlEnabled?badge('AUTO','green'):badge('ADVISE','orange');document.getElementById('m-learn-val').textContent='Ep:'+lr.totalEpisodes;const fu=m.fusion||{};document.getElementById('m-fusion').innerHTML=badge(fu.decisionName||'none','blue');document.getElementById('m-fusion-val').textContent='conf:'+f1(fu.confidence,2);const pd=m.plantDoctor||{};document.getElementById('m-plant').innerHTML=pd.enabled?badge('ON','green'):badge('OFF','red');document.getElementById('m-plant-val').textContent=pd.lastDiseaseName||'--'}
async function refresh(){const d=await api('status');if(!d){document.getElementById('connStatus').textContent='Disconnected';return}lastData=d;document.getElementById('connStatus').textContent=d.wifiConnected?'WiFi: '+d.ipAddress:'Offline';updateSensors(d.sensors);updateActuator(d.actuator);updateModules(d.modules)}
async function setIrrMode(auto){await api('irrigation/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({auto})});document.getElementById('btn-auto').classList.toggle('active',auto);document.getElementById('btn-manual').classList.toggle('active',!auto)}
async function setPump(on){await api('irrigation/pump',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:on})})}
async function setValve(on){await api('irrigation/valve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:on})})}
async function toggleRuleEngine(){const d=await api('status');if(!d)return;const en=!d.modules.irrigation.enabled;await api('system/modules',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ruleEngineEnabled:en})});refresh()}
async function clearAnomaly(){await api('anomaly/clear',{method:'POST'});refresh()}
async function resetGrowth(){if(confirm('确认重置生长数据?')){await api('growth/reset',{method:'POST'});refresh()}}
async function feedback(pos){await api('learning/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({positive:pos})});refresh()}
async function resetLearning(){if(confirm('确认重置Q表?')){await api('learning/reset',{method:'POST'});refresh()}}
async function toggleLearningAuto(){const d=await api('status');if(!d)return;const en=!d.modules.learning.autoControlEnabled;await api('system/modules',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({learningAutoEnabled:en})});refresh()}
async function toggleFusionAuto(){const d=await api('status');if(!d)return;const en=!d.modules.fusion.autoControlEnabled;await api('fusion/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({autoControlEnabled:en})});refresh()}
async function capturePlant(){document.getElementById('pd-preview').src='/api/plant/capture?t='+Date.now();document.getElementById('pd-preview').style.display='block';await api('plant/detect');refresh()}
async function detectPlant(){await api('plant/detect');refresh()}
async function saveWifi(){const ssid=document.getElementById('cfg-ssid').value;const pass=document.getElementById('cfg-pass').value;if(!ssid){alert('请输入SSID');return}await api('system/wifi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ssid,password:pass})});alert('WiFi配置已保存，重启后生效')}
async function startOTA(){const url=document.getElementById('cfg-ota-url').value;if(!url){alert('请输入固件URL');return}if(!confirm('确认开始OTA更新? 更新期间设备将重启')){return}await api('ota/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})}
async function toggleModule(key){const d=await api('system/modules');if(!d)return;const en=!d[key];await api('system/modules',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[key]:en})});refresh()}
async function factoryReset(){if(!confirm('确认恢复出厂设置? 所有配置将被清除')){return}await api('system/factory-reset',{method:'POST'});alert('出厂设置已恢复，设备将重启')}
async function refreshAnomaly(){const d=await api('anomaly/status');if(!d)return;document.getElementById('an-level').textContent=d.alertLevelName||'normal';document.getElementById('an-score').textContent=f1(d.iforestScore,3);document.getElementById('an-samples').textContent=d.totalSamples;document.getElementById('an-anomalies').textContent=d.totalAnomalies;let html='<tr><th>传感器</th><th>当前值</th><th>均值</th><th>Z-Score</th><th>状态</th></tr>';(d.sensors||[]).forEach(s=>{const cls=s.isAnomalous?'red':s.isStuck?'orange':'green';html+=`<tr><td>${s.label}</td><td>${f1(s.value)}</td><td>${f1(s.mean)}</td><td>${f1(s.zScore,2)}</td><td><span class="badge ${cls}">${s.isAnomalous?'异常':s.isStuck?'卡滞':'正常'}</span></td></tr>`});document.getElementById('an-sensor-table').innerHTML=html;const al=await api('anomaly/alerts');if(al){let ah='';al.forEach(a=>{const cls=a.levelName||'info';ah+=`<div class="alert-item ${cls}">${a.sensor}: ${a.message}</div>`});document.getElementById('an-alerts').innerHTML=ah||'无告警'}}
async function refreshGrowth(){const d=await api('growth/status');if(!d)return;document.getElementById('gr-crop').textContent=d.cropCn||d.crop||'--';document.getElementById('gr-stage').textContent=d.stageNameCn||'--';document.getElementById('gr-day').textContent=d.dayOfGrowth||0;document.getElementById('gr-gdd').textContent=f1(d.cumulativeGdd,0);document.getElementById('gr-yield').textContent=f1(d.yieldScore,0);document.getElementById('gr-progress').style.width=Math.min(d.progressPercent||0,100)+'%';document.getElementById('gr-advice').textContent=d.irrigationAdvice||'--';document.getElementById('gr-flower').textContent='Day '+(d.predictedFloweringDay||'--');document.getElementById('gr-mature').textContent='Day '+(d.predictedMaturityDay||'--');const pred=await api('growth/prediction');if(pred&&pred.availableCrops){let ch='';pred.availableCrops.forEach(c=>{ch+=`<button class="btn" onclick="setCrop(${c.id})">${c.nameCn||c.name}</button>`});document.getElementById('gr-crops').innerHTML=ch}}
async function setCrop(id){await api('growth/crop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cropId:id})});refreshGrowth()}
async function refreshLearning(){const d=await api('learning/status');if(!d)return;document.getElementById('lr-episodes').textContent=d.totalEpisodes;document.getElementById('lr-epsilon').textContent=f1(d.epsilon,3);document.getElementById('lr-avgReward').textContent=f1(d.averageReward,2);document.getElementById('lr-action').textContent=d.lastActionName||'--';document.getElementById('lr-recommend').textContent=d.recommendedAction||'--';document.getElementById('btn-lr-auto').textContent='自动控制: '+(d.autoControlEnabled?'开':'关');document.getElementById('btn-lr-auto').classList.toggle('active',d.autoControlEnabled)}
async function refreshFusion(){const d=await api('fusion/status');if(!d)return;document.getElementById('fu-decision').textContent=d.decisionName||'--';document.getElementById('fu-confidence').textContent=f1(d.confidence*100,0)+'%';document.getElementById('fu-score').textContent=f1(d.finalScore,1);document.getElementById('fu-total').textContent=d.totalDecisions;document.getElementById('btn-fu-auto').textContent='自动控制: '+(d.autoControlEnabled?'开':'关');document.getElementById('btn-fu-auto').classList.toggle('active',d.autoControlEnabled);if(d.nn){document.getElementById('nn-none').textContent=f1(d.nn.none*100,0)+'%';document.getElementById('nn-moderate').textContent=f1(d.nn.moderate*100,0)+'%';document.getElementById('nn-heavy').textContent=f1(d.nn.heavy*100,0)+'%'}const s=await api('fusion/sensors');if(s){let h='<tr><th>传感器</th><th>原始值</th><th>滤波值</th><th>权重</th><th>可靠性</th></tr>';s.forEach(c=>{h+=`<tr><td>${c.label}</td><td>${f1(c.raw)}</td><td>${f1(c.filtered)}</td><td>${f1(c.weight,2)}</td><td>${f1(c.reliability,2)}</td></tr>`});document.getElementById('fu-sensor-table').innerHTML=h}}
async function refreshPlant(){const d=await api('plant/status');if(!d)return;document.getElementById('pd-enabled').innerHTML=d.enabled?badge('ON','green'):badge('OFF','red');document.getElementById('pd-camera').innerHTML=d.cameraReady?badge('READY','green'):badge('NA','red');document.getElementById('pd-model').innerHTML=d.modelLoaded?badge('READY','green'):badge('NA','red');document.getElementById('pd-disease').textContent=d.lastDiseaseNameCn||d.lastDiseaseName||'--';document.getElementById('pd-conf').textContent=f1(d.lastConfidence*100,0)+'%';document.getElementById('pd-treatment').textContent=d.treatment||'--'}
async function refreshConfig(){const d=await api('status');if(!d)return;document.getElementById('cfg-hw').textContent=d.hardwareProfile||'--';document.getElementById('cfg-wifi').textContent=d.wifiConnected?'已连接':'未连接';document.getElementById('cfg-ip').textContent=d.ipAddress||'--';const m=await api('system/modules');if(m){document.getElementById('cfg-rule').textContent=m.ruleEngineEnabled?'开':'关';document.getElementById('cfg-rule').classList.toggle('active',m.ruleEngineEnabled);document.getElementById('cfg-learn').textContent=m.learningAutoEnabled?'开':'关';document.getElementById('cfg-learn').classList.toggle('active',m.learningAutoEnabled);document.getElementById('cfg-fusion').textContent=m.fusionAutoEnabled?'开':'关';document.getElementById('cfg-fusion').classList.toggle('active',m.fusionAutoEnabled);document.getElementById('cfg-plant').textContent=m.plantDoctorEnabled?'开':'关';document.getElementById('cfg-plant').classList.toggle('active',m.plantDoctorEnabled)}}
let currentTab='dashboard';
const origShowPage=showPage;
showPage=function(id){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));document.getElementById('page-'+id).classList.add('active');if(event&&event.target)event.target.classList.add('active');currentTab=id;refreshTab()};
function refreshTab(){switch(currentTab){case'anomaly':refreshAnomaly();break;case'growth':refreshGrowth();break;case'learning':refreshLearning();break;case'fusion':refreshFusion();break;case'plant':refreshPlant();break;case'config':refreshConfig();break}}
refresh();setInterval(refresh,3000);setInterval(refreshTab,5000);
</script>
</body>
</html>
)rawliteral";

inline unsigned int getWebHtmlLength() {
    return sizeof(kWebHtml) - 1;
}

}  // namespace agri
