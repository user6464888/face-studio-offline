import {SalonViewer} from './viewer.js';
const $=s=>document.querySelector(s);
try{
 const response=await fetch('/api/salon/state');if(!response.ok)throw Error('无法读取演示状态');
 const state=await response.json(),catalog=state.catalog,heads=state.heads;
 const viewer=new SalonViewer($('#canvas'),$('#render-status'));
 let head=(state.workflow.state==='demo'?heads.find(h=>h.id==='oval'):heads.find(h=>!h.is_standard))||heads[0],style=catalog.styles[0],display='style',sequence=0,manualFace=null;
 const shapes=catalog.face_shapes;
 $('#job-notice').textContent=state.workflow.notice+(state.workflow.quality_warnings||[]).join(' ');
 $('#head-select').innerHTML=heads.map(h=>`<option value="${h.id}">${h.name}</option>`).join('');
 $('#head-select').value=head.id;
 $('#face-select').innerHTML=Object.entries(shapes).map(([id,s])=>`<option value="${id}">${s.name}</option>`).join('');
 $('#style-select').innerHTML=catalog.styles.map(s=>`<option value="${s.id}">${String(s.number).padStart(2,'0')} ${s.name}</option>`).join('');
 function faceId(){return head.is_standard?head.face_shape:manualFace||head.analysis?.suggested_shape||'oval'}
 function recommended(s){return s.suitable_faces.includes(faceId())}
 function sync(){
  $('#style-select').value=style.id;
  $('#face-select').disabled=head.is_standard;$('#face-select').value=faceId();
  document.querySelector('[data-display="original"]').disabled=head.is_standard;
  const face=shapes[faceId()];
  const preferred=state.advice[faceId()];
  $('#details').innerHTML=`<span class="eyebrow">原 ${String(style.number).padStart(2,'0')}</span><h2>${style.name}</h2><p>${style.intro}</p><h3>${head.is_standard?'当前示意脸型':'几何初判（可对照标准头）'}：${face.name}</h3><p>${face.intro}</p><p>${face.advice}</p><div class="release-advice"><strong>建议先比较</strong>${preferred.map(s=>`<button data-style="${s.id}">${String(s.number).padStart(2,'0')} ${s.name}</button>`).join('')}</div><h3>本款适合脸型</h3><p>${style.suitable_source}</p><h3>打理建议 · ${'★'.repeat(style.difficulty)}${'☆'.repeat(5-style.difficulty)}</h3><p>${style.care}</p><p class="release-note">三维程序化近似，不代表照片精确复刻或真实理发结果。</p>`;
  $('#preview-title').textContent=head.name+' · '+(display==='style'?style.name:display==='original'?'原始采集':'光头');
  document.querySelectorAll('[data-display]').forEach(b=>b.classList.toggle('active',b.dataset.display===display));
  const sort=$('#sort').value,list=[...catalog.styles];
  list.sort((a,b)=>sort==='recommended'?Number(recommended(b))-Number(recommended(a))||a.number-b.number:sort==='easy'?a.difficulty-b.difficulty||a.number-b.number:a.number-b.number);
  $('#catalog').innerHTML=list.map(s=>`<button class="release-card ${s.id===style.id?'active':''}" data-style="${s.id}"><strong>${String(s.number).padStart(2,'0')} ${s.name}</strong><small>${s.intro}</small><small>${recommended(s)?'适合当前示意脸型 · ':''}${s.suitable_source}</small></button>`).join('');
 }
 async function redraw(){const token=++sequence;sync();document.querySelectorAll('[data-hair-layer]').forEach(i=>i.disabled=true);try{await viewer.show(head,style,display)}catch(e){if(token===sequence)$('#render-status').textContent='未载入：'+e.message}}
 viewer.onLayersChanged=(layers,visibility)=>{const names={short:'短发底层',medium:'中发过渡层',long:'长发丝层'};document.querySelectorAll('[data-hair-layer]').forEach(i=>{i.disabled=!layers.includes(i.dataset.hairLayer);i.checked=layers.includes(i.dataset.hairLayer)&&visibility[i.dataset.hairLayer]!==false});$('#layer-note').textContent=layers.length?'当前显示：'+layers.filter(id=>visibility[id]!==false).map(id=>names[id]).join('＋'):'当前显示光头底模'};
 $('#head-select').onchange=e=>{head=heads.find(h=>h.id===e.target.value);manualFace=null;if(head.is_standard&&display==='original')display='style';redraw()};
 $('#face-select').onchange=e=>{manualFace=e.target.value;sync()};
 $('#style-select').onchange=e=>{style=catalog.styles.find(s=>s.id===e.target.value);display='style';redraw()};
 $('#sort').onchange=sync;
 document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.style){style=catalog.styles.find(s=>s.id===b.dataset.style);display='style';redraw()}if(b.dataset.display){display=b.dataset.display;redraw()}});
 $('#views').onclick=e=>{const b=e.target.closest('[data-angle]');if(b){viewer.auto=false;viewer.setAngle(b.dataset.angle);$('#rotate').textContent='自动旋转：关'}};
 $('#fit').onclick=()=>{viewer.distance=.85;viewer.setAngle(25)};
 $('#rotate').onclick=()=>{viewer.auto=!viewer.auto;$('#rotate').textContent='自动旋转：'+(viewer.auto?'开':'关')};
 $('#color').onchange=e=>viewer.applyColor(e.target.value);
 $('#hair-layers').onchange=e=>{if(e.target.dataset.hairLayer)viewer.setHairLayerVisible(e.target.dataset.hairLayer,e.target.checked)};
 $('#snapshot').onclick=()=>viewer.snapshot();
 await redraw();
}catch(e){$('#job-notice').classList.add('release-error');$('#job-notice').textContent='无法启动演示：'+e.message;console.error(e)}
