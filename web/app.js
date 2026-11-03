'use strict';
const $ = id => document.getElementById(id);
const items = [];
let busy = false, filter = 'all', nextId = 1, connected = false;
let viewerId = null, viewerZoom = 1;
const labels = {queued:'等待检测', processing:'检测中', candidate:'候选读数', review:'需要复核', no_area:'未找到读数区', no_slots:'未找到轮位', error:'处理失败'};
const reasonLabels = {window_touches_image_edge:'读数区域接触图片边缘',too_few_digits:'数字数量不足',too_few_slots_for_decimal_format:'轮位不足以格式化小数',unusual_slot_count_review_required:'轮位数量异常，请复核'};
const phaseLabels = {classification:'电表类型分类',area_detection:'读数区域定位',digit_recognition:'液晶数字识别',slot_detection:'滚轮窗口定位',wheel_recognition:'滚轮识别与联合解码'};
function el(tag, className, text) {const n=document.createElement(tag); if(className)n.className=className;if(text!==undefined)n.textContent=text;return n;}
function notice(message) {$('notice').textContent=message;$('notice').hidden=!message;}
function needsReview(item) {return item.result && item.result.status !== 'candidate';}
function matchesFilter(item) {
  return filter==='all'||filter==='review'&&needsReview(item)||filter==='candidate'&&item.result?.status==='candidate'||filter==='error'&&item.state==='error';
}
function refresh() {
  const done=items.filter(x=>x.result || x.state==='error');
  const successful=items.filter(x=>x.result);
  $('stat-total').replaceChildren(document.createTextNode(String(items.length)),el('small','','张'));
  $('stat-done').replaceChildren(document.createTextNode(String(done.length)),el('small','','张'));
  $('stat-review').replaceChildren(document.createTextNode(String(items.filter(needsReview).length)),el('small','','张'));
  const average=successful.length?Math.round(successful.reduce((n,x)=>n+x.result.elapsed_ms,0)/successful.length):'—';
  $('stat-time').replaceChildren(document.createTextNode(String(average)),el('small','','ms'));
  $('result-count').textContent=String(items.length);
  $('empty').hidden=items.length>0;
  $('run').disabled=busy||!items.length||!connected;
  $('clear').disabled=busy||!items.length;
  $('export').disabled=!done.length;
  $('files').disabled=busy;
  const queued=items.filter(x=>!x.result).length;
  $('run').replaceChildren(document.createTextNode(busy?'正在检测…':queued?`开始检测 · ${queued} 张`:'重新检测'),el('span','','↗'));
  let visible=0;
  for(const item of items) {
    const show=matchesFilter(item);
    item.card.hidden=!show; if(show)visible++;
    item.remove.disabled=busy;
  }
  $('no-match').hidden=!items.length||visible>0;
  syncViewer();
}
function annotationLines(item) {
  const r=item.result;
  if(!r)return [labels[item.state]||item.state,`${(item.file.size/1024/1024).toFixed(2)} MiB`,...(item.error?[item.error]:[])];
  const lines=[
    `${r.meter_type==='wheel'?'机械滚轮表':'液晶电表'}  ·  ${labels[r.status]||r.status}  ·  ${Math.round(r.elapsed_ms)} ms`,
    `原始数字：${r.raw_digits??'—'}  ·  ${r.meter_type==='wheel'?r.slots.length+' 个轮位':r.digit_detections.length+' 个数字'}  ·  ${r.width} × ${r.height}`,
    r.meter_type==='wheel'?'固定一位小数 · 主读数向下取整，需人工复核':'固定两位小数 · 读数为模型候选值',
  ];
  if(r.prediction)lines.push(`四舍五入备选：${r.prediction.author_rounded_raw_digits}  ·  独立识别：${r.prediction.independent_raw_digits}`);
  if(r.status==='no_area')lines.push('未找到读数区域，请使用读数区域完整、清晰的照片');
  if(r.status==='no_slots')lines.push('已找到区域，但未检测到滚轮窗口');
  if(r.reasons.length)lines.push(r.reasons.map(x=>reasonLabels[x]||x).join('；'));
  lines.push(Object.keys(phaseLabels).filter(key=>r.timings_ms[key]!==undefined).map(key=>`${phaseLabels[key]} ${r.timings_ms[key].toFixed(1)} ms`).join(' · '));
  return lines.filter(Boolean);
}
function wrapCanvasText(ctx,text,width) {
  const lines=[];let line='';
  for(const character of String(text).replace(/\s+/g,' ')) {
    if(line&&ctx.measureText(line+character).width>width){lines.push(line);line='';}
    line+=character;
  }
  if(line)lines.push(line);
  return lines;
}
function draw(item,canvas=item.canvas,maxSide=1280) {
  const img=item.image;
  if(!img?.naturalWidth)return;
  const scale=Math.min(1,maxSide/Math.max(img.naturalWidth,img.naturalHeight));
  const photoWidth=Math.max(1,Math.round(img.naturalWidth*scale)),photoHeight=Math.max(1,Math.round(img.naturalHeight*scale));
  const width=Math.max(480,photoWidth),offsetX=(width-photoWidth)/2;
  const ctx=canvas.getContext('2d'),font=Math.max(16,width/48),padding=font*1.1,lineHeight=font*1.55;
  const fontFamily='"Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.font=`${font}px ${fontFamily}`;
  const lines=[...wrapCanvasText(ctx,item.file.name,width-padding*2),...annotationLines(item).flatMap(text=>wrapCanvasText(ctx,text,width-padding*2))];
  const panelHeight=Math.ceil(padding*2+font*2.7+lines.length*lineHeight);
  canvas.width=width;canvas.height=photoHeight+panelHeight;
  ctx.fillStyle='#142a24';ctx.fillRect(0,0,width,canvas.height);
  ctx.drawImage(img,offsetX,0,photoWidth,photoHeight);
  const r=item.result;
  const sx=r?photoWidth/r.width:1,sy=r?photoHeight/r.height:1;
  const line=Math.max(1.5,photoWidth/400),boxFont=Math.max(13,photoWidth/48);
  function box(b,color,label) {
    const [x1,y1,x2,y2]=b.map((v,i)=>v*(i%2?sy:sx)+(i%2?0:offsetX));
    ctx.strokeStyle=color;ctx.lineWidth=line;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    if(label){ctx.font=`600 ${boxFont}px ${fontFamily}`;const textW=ctx.measureText(label).width;const y=Math.max(boxFont+6,y1);ctx.fillStyle=color;ctx.fillRect(x1,y-boxFont-6,textW+10,boxFont+6);ctx.fillStyle='#fff';ctx.fillText(label,x1+5,y-4);}
  }
  if(r&&$('boxes').checked) {
    if(r.area)box(r.area.xyxy,'#168c77','读数区');
    if(r.meter_type==='digital'&&r.area_crop_xyxy)for(const d of r.digit_detections){const [a,b]=r.area_crop_xyxy;box(d.xyxy.map((v,i)=>v+(i%2?b:a)),'#d69a39',d.class_name);}
    for(const s of r.slots)box(s.xyxy_in_image,'#d69a39',String(s.index+1));
  }
  // The result band belongs to the canvas, with its own space so it cannot cover a meter window.
  let y=photoHeight+padding;
  ctx.textBaseline='top';ctx.fillStyle='#eaf5ef';ctx.font=`600 ${font*1.9}px ${fontFamily}`;
  ctx.fillText(`读数  ${r?.reading??'—'}`,padding,y,width-padding*2);y+=font*2.7;
  ctx.font=`${font}px ${fontFamily}`;
  lines.forEach((text,index)=>{ctx.fillStyle=index===0?'#f0f6f2':'#aec7ba';ctx.fillText(text,padding,y);y+=lineHeight;});
}
function makeZoomable(canvas,item) {
  canvas.classList.add('zoomable');canvas.tabIndex=0;canvas.setAttribute('role','button');
  canvas.setAttribute('aria-label',`放大查看 ${item.file.name}`);canvas.title='点击放大 · 支持左右翻页';
  canvas.addEventListener('click',()=>openViewer(item));
  canvas.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openViewer(item);}});
}
function viewerItems() {return items.filter(matchesFilter);}
function syncViewer() {
  if(!$('image-viewer').open)return;
  const gallery=viewerItems();
  if(!gallery.length){$('image-viewer').close();return;}
  let index=gallery.findIndex(item=>item.id===viewerId);
  if(index<0){viewerId=gallery[0].id;viewerZoom=1;renderViewer();return;}
  $('viewer-position').textContent=`${index+1} / ${gallery.length}`;
  $('viewer-prev').disabled=index===0;$('viewer-next').disabled=index===gallery.length-1;
}
function sizeViewer(resetScroll=false) {
  const canvas=$('viewer-canvas'),stage=$('viewer-stage');
  if(canvas.hidden)return;
  const fit=Math.min(1,Math.max(1,stage.clientWidth-112)/canvas.width,Math.max(1,stage.clientHeight-32)/canvas.height);
  canvas.style.width=`${Math.max(1,Math.round(canvas.width*fit*viewerZoom))}px`;
  canvas.style.height='auto';
  $('viewer-zoom').textContent=`${Math.round(fit*viewerZoom*100)}%`;
  $('viewer-out').disabled=viewerZoom<=0.5;$('viewer-in').disabled=viewerZoom>=8;
  if(resetScroll){stage.scrollTop=0;stage.scrollLeft=0;}
}
function renderViewer() {
  if(!$('image-viewer').open)return;
  const item=items.find(x=>x.id===viewerId);if(!item){syncViewer();return;}
  $('viewer-title').textContent=item.file.name;
  const canvas=$('viewer-canvas');canvas.hidden=!item.image?.naturalWidth;
  $('viewer-message').hidden=!canvas.hidden;
  $('viewer-message').textContent=item.previewError?'图片无法预览':'图片加载中…';
  $('viewer-in').disabled=canvas.hidden;$('viewer-out').disabled=canvas.hidden;$('viewer-fit').disabled=canvas.hidden;
  if(!canvas.hidden){draw(item,canvas,4096);sizeViewer(true);}
  else $('viewer-zoom').textContent='—';
  syncViewer();
}
function openViewer(item) {
  viewerId=item.id;viewerZoom=1;
  if(!$('image-viewer').open)$('image-viewer').showModal();
  renderViewer();
}
function stepViewer(direction) {
  const gallery=viewerItems(),index=gallery.findIndex(item=>item.id===viewerId),next=gallery[index+direction];
  if(!next)return;
  viewerId=next.id;viewerZoom=1;renderViewer();
}
function loadPreview(item,url) {
  const img=new Image();
  img.onload=()=>{item.image=img;draw(item);if(item.id===viewerId)renderViewer();};
  img.onerror=()=>{item.previewError=true;if(!item.result)item.caption.textContent='预览不可用，检测时将检查图片内容';if(item.id===viewerId)renderViewer();};
  img.src=url;
}
function createCard(item) {
  const card=el('article','card');item.card=card;
  const head=el('div','card-head'),name=el('span','filename',item.file.name),remove=el('button','icon-button','×');
  name.title=item.file.name;remove.setAttribute('aria-label',`移除 ${item.file.name}`);item.remove=remove;
  remove.onclick=()=>{if(busy)return;URL.revokeObjectURL(item.url);items.splice(items.indexOf(item),1);card.remove();refresh();};
  head.append(name,remove);
  const wrap=el('div','preview-wrap'),canvas=el('canvas');item.canvas=canvas;canvas.setAttribute('aria-label',`${item.file.name} 的图片预览和检测框`);
  makeZoomable(canvas,item);
  const cover=el('div','processing-cover');cover.append(el('span','spinner'),el('span','','正在识别'));cover.hidden=true;item.cover=cover;wrap.append(canvas,cover);
  const info=el('div','card-info'),top=el('div','card-topline'),type=el('span','type-label','尚未识别'),badge=el('span','badge','等待检测');item.type=type;item.badge=badge;top.append(type,badge);
  const row=el('div','reading-row'),reading=el('strong','reading','—'),elapsed=el('span','elapsed','');item.reading=reading;item.elapsed=elapsed;row.append(reading,elapsed);
  const raw=el('p','raw',`${(item.file.size/1024/1024).toFixed(2)} MiB`);item.raw=raw;info.append(top,row,raw);
  const bottom=el('div','card-bottom'),caption=el('span','', '检测后显示定位和识别详情'),detail=el('button','detail-button','查看详情 ↗');detail.disabled=true;detail.onclick=()=>showDetail(item);item.caption=caption;item.detail=detail;bottom.append(caption,detail);
  card.append(head,wrap,info,bottom);$('cards').append(card);loadPreview(item,item.url);
}
function updateCard(item) {
  const r=item.result,status=r?.status||item.state;
  item.badge.className=`badge ${status}`;item.badge.textContent=labels[status]||status;
  item.cover.hidden=item.state!=='processing';
  item.type.textContent=r?(r.meter_type==='wheel'?'机械滚轮表':'液晶电表'):'尚未识别';
  item.reading.textContent=r?.reading??'—';item.elapsed.textContent=r?`${Math.round(r.elapsed_ms)} ms`:'';
  item.raw.textContent=r?`原始数字  ${r.raw_digits??'—'}`:`${(item.file.size/1024/1024).toFixed(2)} MiB`;
  item.detail.disabled=!r;item.caption.title='';
  item.caption.textContent=item.error||(r?(r.meter_type==='wheel'?`${r.slots.length} 个轮位 · 请人工复核`: `${r.digit_detections.length} 个数字 · 固定两位小数`):'检测后显示定位和识别详情');
  if(r?.status==='no_area')item.caption.textContent='请使用读数区域完整、清晰的照片';
  if(r?.status==='no_slots')item.caption.textContent='已找到区域，但未检测到滚轮窗口';
  item.caption.title=item.caption.textContent;
  draw(item);
  if(item.id===viewerId)renderViewer();
}
function addFiles(files) {
  if(busy)return;
  const errors=[];
  for(const file of files) {
    if(!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)){errors.push(`${file.name}：不支持的格式`);continue;}
    if(file.size>16*1024*1024){errors.push(`${file.name}：超过 16 MiB`);continue;}
    if(items.some(x=>x.file.name===file.name&&x.file.size===file.size&&x.file.lastModified===file.lastModified))continue;
    const item={id:nextId++,file,url:URL.createObjectURL(file),state:'queued',result:null,error:null};items.push(item);createCard(item);
  }
  notice(errors.join('；'));$('files').value='';refresh();
}
async function run() {
  if(busy)return;
  const pending=items.some(x=>!x.result)?items.filter(x=>!x.result):[...items];
  if(!pending.length)return;
  busy=true;notice('');$('progress-wrap').hidden=false;$('progress').value=0;$('progress-label').textContent='正在定位与识别';
  for(const item of pending){item.result=null;item.error=null;item.state='queued';updateCard(item);}
  refresh();let completed=0;
  for(let offset=0;offset<pending.length;offset+=4) {
    const group=pending.slice(offset,offset+4),form=new FormData();
    for(const item of group){item.state='processing';updateCard(item);form.append('images',item.file,item.file.name);}
    $('progress-count').textContent=`${completed} / ${pending.length}`;
    try {
      const response=await fetch('/api/batch',{method:'POST',body:form});
      const body=await response.json();
      if(!response.ok)throw new Error(body.error?.message||`HTTP ${response.status}`);
      if(!Array.isArray(body.results)||body.results.length!==group.length)throw new Error('服务器返回了不完整的结果');
      body.results.forEach((entry,i)=>{const item=group[i];item.result=entry.result||null;item.error=entry.error||null;item.state=item.result?'done':'error';updateCard(item);});
    } catch(error) {
      for(const item of group){item.state='error';item.error=error.message;updateCard(item);}
      notice(`部分照片处理失败：${error.message}。可点击开始检测重试。`);
    }
    completed+=group.length;$('progress').value=completed/pending.length*100;$('progress-count').textContent=`${completed} / ${pending.length}`;refresh();
  }
  busy=false;$('progress-label').textContent='本轮检测完成';refresh();
}
function showDetail(item) {
  const r=item.result;if(!r)return;
  $('detail-title').textContent=item.file.name;const body=$('detail-body');body.replaceChildren();
  const canvas=el('canvas','detail-preview');draw(item,canvas);makeZoomable(canvas,item);body.append(canvas);
  const meta=el('div','detail-meta');meta.append(el('span','',r.meter_type==='wheel'?'机械滚轮表':'液晶电表'),el('b','',r.reading??'无读数'),el('span','',labels[r.status]),el('span','',`${r.width} × ${r.height}`));body.append(meta);
  const warnings=r.reasons.map(x=>reasonLabels[x]||x);
  if(r.meter_type==='wheel')warnings.unshift(`滚轮主读数采用向下取整，请人工复核。四舍五入候选：${r.prediction?.author_rounded_raw_digits??'—'}`);
  if(r.status==='no_area')warnings.push('未找到读数区域，无法继续识别。');
  if(r.status==='no_slots')warnings.push('未找到滚轮窗口，无法继续识别。');
  if(warnings.length)body.append(el('p','detail-warning',warnings.join(' ')));
  const table=el('table','timings');table.setAttribute('aria-label','各阶段推理耗时');
  for(const [phase,time] of Object.entries(r.timings_ms)){const tr=el('tr');tr.append(el('td','',phaseLabels[phase]||phase),el('td','',`${time.toFixed(1)} ms`));table.append(tr);}body.append(table);
  const details=el('details');details.append(el('summary','','完整 JSON 结果'),el('pre','',JSON.stringify(r,null,2)));body.append(details);$('detail').showModal();
}
async function health() {
  try {const response=await fetch('/api/health');if(!response.ok)throw new Error();const data=await response.json();connected=data.status==='ready';$('health').className='health ready';$('health').replaceChildren(el('i'),document.createTextNode('服务已连接'));$('version').textContent=`本地推理 · ${data.models_loaded} 个模型就绪`;}
  catch{connected=false;$('health').className='health offline';$('health').replaceChildren(el('i'),document.createTextNode('服务未连接'));}
  refresh();
}
$('files').addEventListener('change',e=>addFiles(e.target.files));
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();if(!busy)$('files').click();}});
let dragDepth=0;
for(const event of ['dragenter','dragover','dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();e.stopPropagation();});
$('dropzone').addEventListener('dragenter',()=>{dragDepth++;$('dropzone').classList.add('dragging');});
$('dropzone').addEventListener('dragleave',()=>{if(--dragDepth<=0)$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>{dragDepth=0;$('dropzone').classList.remove('dragging');addFiles(e.dataTransfer.files);});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
$('run').onclick=run;
$('clear').onclick=()=>{if(busy)return;for(const item of items)URL.revokeObjectURL(item.url);items.length=0;$('cards').replaceChildren();$('progress-wrap').hidden=true;notice('');refresh();};
$('boxes').onchange=()=>{items.forEach(x=>draw(x));renderViewer();};
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});refresh();});
$('export').onclick=()=>{
  const results=items.filter(x=>x.result||x.state==='error').map(x=>({filename:x.file.name,...(x.result?{result:x.result}:{error:x.error})}));
  const url=URL.createObjectURL(new Blob([JSON.stringify({exported_at:new Date().toISOString(),results},null,2)],{type:'application/json'}));
  const link=el('a');link.href=url;link.download=`meter-results-${Date.now()}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('close-detail').onclick=()=>$('detail').close();
$('detail').addEventListener('click',e=>{if(e.target===$('detail')){const b=e.target.getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)e.target.close();}});
$('viewer-close').onclick=()=>$('image-viewer').close();
$('viewer-prev').onclick=()=>stepViewer(-1);
$('viewer-next').onclick=()=>stepViewer(1);
$('viewer-fit').onclick=()=>{viewerZoom=1;sizeViewer(true);};
$('viewer-in').onclick=()=>{viewerZoom=Math.min(8,viewerZoom*1.5);sizeViewer();};
$('viewer-out').onclick=()=>{viewerZoom=Math.max(0.5,viewerZoom/1.5);sizeViewer();};
$('image-viewer').addEventListener('close',()=>{viewerId=null;$('viewer-canvas').width=1;$('viewer-canvas').height=1;});
document.addEventListener('keydown',event=>{
  if(!$('image-viewer').open)return;
  if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();stepViewer(event.key==='ArrowLeft'?-1:1);}
});
window.addEventListener('resize',()=>{if($('image-viewer').open)sizeViewer();});
health();setInterval(health,30000);
