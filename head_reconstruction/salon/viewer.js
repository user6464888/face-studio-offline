import * as THREE from 'three';
import {PLYLoader} from '../viewer/vendor/PLYLoader.js';
import {OrbitControls} from '../viewer/vendor/OrbitControls.js';
import {hybridMaterial} from './hybrid_material.js?v=ear-clean-1';

export class SalonViewer{
  constructor(host,status){
    this.host=host;this.status=status;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#edf2ee');
    this.camera=new THREE.PerspectiveCamera(32,1,.004,10);
    this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));host.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.root=new THREE.Group();this.scene.add(this.root);
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x85958c,1.8));const light=new THREE.DirectionalLight(0xffffff,1.8);light.position.set(-1,2,3);this.scene.add(light);
    this.loader=new PLYLoader();this.surfaceLoader=new PLYLoader();this.surfaceLoader.setCustomPropertyNameMapping({hybridParameters:['coverage','length_mm'],scalpPosition:['scalp_x','scalp_y','scalp_z'],hairFlow:['flow_x','flow_y','flow_z']});
    this.sequence=0;this.color='original';this.distance=.85;this.focus=new THREE.Vector3(0,-.04,0);this.auto=false;this.materials=[];this.textures=new Map();
    this.layerVisibility={short:true,medium:true,long:true};this.availableLayers=[];this.revision='current';
    new ResizeObserver(()=>{this.renderer.setSize(host.clientWidth,host.clientHeight,false);this.camera.aspect=host.clientWidth/host.clientHeight;this.camera.updateProjectionMatrix()}).observe(host);
    this.setAngle(25);let previous=performance.now(),sampleStart=previous,frames=0;const animate=now=>{requestAnimationFrame(animate);if(this.auto)this.root.rotation.y+=(now-previous)*.0005;previous=now;this.controls.update();this.renderer.render(this.scene,this.camera);frames++;if(now-sampleStart>1500){host.dataset.fps=(frames*1000/(now-sampleStart)).toFixed(1);frames=0;sampleStart=now}};requestAnimationFrame(animate);
  }
  setAngle(value){this.root.rotation.y=0;this.camera.up.set(0,1,0);if(value==='top'){this.camera.position.copy(this.focus).add(new THREE.Vector3(0,this.distance,.001));this.camera.up.set(0,0,-1)}else{const a=Number(value)*Math.PI/180;this.camera.position.copy(this.focus).add(new THREE.Vector3(Math.sin(a)*this.distance,0,Math.cos(a)*this.distance))}this.controls.target.copy(this.focus);this.controls.update()}
  clear(){this.root.traverse(o=>{if(o.isMesh){o.geometry.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material]))m.dispose()}});this.root.clear();this.materials=[]}
  async bareHead(head,mode,texture){
    const g=await this.loader.loadAsync((mode==='original'&&head.original_file?head.original_file:head.bald_file)+'?v='+head.fingerprint);
    if(!g.getAttribute('normal'))g.computeVertexNormals();
    if(mode==='original')return new THREE.Mesh(g,new THREE.MeshBasicMaterial({vertexColors:true}));
    if(head.is_standard)return new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.95}));
    const mask=new Uint8Array(await fetch(head.texture_faces).then(r=>r.arrayBuffer()));const indices=g.index.array,face=[],skin=[];
    for(let i=0;i<mask.length;i++)(mask[i]?skin:face).push(indices[i*3],indices[i*3+1],indices[i*3+2]);
    const original=g.getAttribute('position').count,ids=[...new Set(skin)],mapping=new Map(ids.map((id,i)=>[id,original+i]));
    for(const [name,attr]of Object.entries(g.attributes)){const size=attr.itemSize,out=new attr.array.constructor((original+ids.length)*size);out.set(attr.array);for(let i=0;i<ids.length;i++)for(let k=0;k<size;k++)out[(original+i)*size+k]=name==='color'?1:attr.array[ids[i]*size+k];g.setAttribute(name,new THREE.BufferAttribute(out,size,attr.normalized))}
    g.setIndex([...face,...skin.map(id=>mapping.get(id))]);g.clearGroups();g.addGroup(0,face.length,0);g.addGroup(face.length,skin.length,1);
    return new THREE.Mesh(g,[new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}),new THREE.MeshBasicMaterial({map:texture,vertexColors:true,side:THREE.DoubleSide})]);
  }
  async show(head,style,mode='style'){
    const token=++this.sequence;this.status.textContent='正在构建并载入 '+head.name+(mode==='style'?' · '+style.name:'')+'…';
    try{
      const textureKey=head.is_standard?'standard':head.fingerprint;
      let texture=this.textures.get(textureKey);
      if(!texture){if(head.is_standard){texture=new THREE.DataTexture(new Uint8Array([180,168,152,255]),1,1);texture.needsUpdate=true}else texture=await new THREE.TextureLoader().loadAsync(head.texture_file);this.textures.set(textureKey,texture)}
      texture.colorSpace=THREE.SRGBColorSpace;texture.wrapS=THREE.RepeatWrapping;
      const [headMesh,recipe]=await Promise.all([this.bareHead(head,mode,texture),mode==='style'?fetch('/api/salon/style?head='+head.id+'&style='+style.id+'&revision='+this.revision).then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.error);return d}):null]);
      if(token!==this.sequence){headMesh.geometry.dispose();for(const material of (Array.isArray(headMesh.material)?headMesh.material:[headMesh.material]))material.dispose();return}
      const pieces=[];const materials=[];
      if(recipe)for(const [id,spec]of Object.entries(recipe.layers)){
        const g=await (id==='long'?this.loader:this.surfaceLoader).loadAsync(spec.file+'?v='+recipe.identity);
        if(!g.getAttribute('normal'))g.computeVertexNormals();
        const material=id==='long'?new THREE.MeshStandardMaterial({vertexColors:true,roughness:recipe.roughness,side:THREE.DoubleSide}):hybridMaterial(texture,id==='medium'||id==='volume',head.centre,head.is_standard);
        const mesh=new THREE.Mesh(g,material);mesh.userData.hair=true;mesh.userData.kind=id;
        if(id==='long')g.userData.originalColors=g.getAttribute('color').array.slice();
        pieces.push(mesh);materials.push(material);
      }
      if(token!==this.sequence){headMesh.geometry.dispose();pieces.forEach(m=>{m.geometry.dispose();m.material.dispose()});return}
      this.clear();this.root.add(headMesh,...pieces);this.materials=materials;
      this.availableLayers=recipe?Object.keys(recipe.layers):[];
      for(const piece of pieces)piece.visible=this.layerVisibility[piece.userData.kind]!==false;
      const scale=.102/head.radii[1];this.root.scale.setScalar(scale);this.root.position.fromArray(head.centre).multiplyScalar(-scale);
      this.applyColor(this.color);
      const triangles=pieces.reduce((sum,m)=>sum+(m.geometry.index?m.geometry.index.count:m.geometry.attributes.position.count)/3,0);
      this.status.textContent=mode==='style'?`已载入 · ${Math.round(triangles).toLocaleString()} 个发型三角面 · 程序化三维初版`:`已载入${mode==='original'?'原始采集模型':'光头底模'}`;
      this.notifyLayers();
    }catch(e){if(token===this.sequence)this.status.textContent='未载入：'+e.message;throw e}
  }
  notifyLayers(){
    this.host.dataset.hairLayers=this.availableLayers.join(',');
    this.host.dataset.visibleHairLayers=this.availableLayers.filter(id=>this.layerVisibility[id]!==false).join(',');
    this.onLayersChanged?.(this.availableLayers,this.layerVisibility);
  }
  setHairLayerVisible(id,visible){
    if(!(id in this.layerVisibility))return;
    this.layerVisibility[id]=visible;
    this.root.traverse(o=>{if(o.userData.hair&&o.userData.kind===id)o.visible=visible});
    this.notifyLayers();
  }
  applyColor(color){
    this.color=color;const selected=new THREE.Color(color==='original'?'#252329':color);
    this.root.traverse(o=>{if(!o.userData.hair)return;const m=o.material;if(o.userData.kind==='long'){
      const attr=o.geometry.getAttribute('color'),original=o.geometry.userData.originalColors;
      if(color==='original')attr.array.set(original);else for(let i=0;i<attr.count;i++){const shade=Math.min(1.4,Math.max(.65,(original[i*3]+original[i*3+1]+original[i*3+2])/.04));attr.setXYZ(i,selected.r*shade,selected.g*shade,selected.b*shade)}attr.needsUpdate=true;
    }else m.userData.hairColor?.value.copy(selected)});
  }
  snapshot(){
    this.renderer.render(this.scene,this.camera);
    this.renderer.domElement.toBlob(blob=>{
      if(!blob){this.status.textContent='PNG 导出失败，请重试';return}
      const url=URL.createObjectURL(blob),dialog=document.createElement('dialog');
      dialog.style.cssText='max-width:90vw;max-height:95vh;border:1px solid #cad8cf;border-radius:12px;padding:16px';
      const title=document.createElement('h2');title.textContent='当前视角图片';
      const img=document.createElement('img');img.alt='当前视角 PNG';img.src=url;img.style.cssText='display:block;max-width:80vw;max-height:65vh';
      const a=document.createElement('a');a.href=url;a.download='face_studio.png';a.textContent='下载 PNG';a.style.cssText='display:inline-block;margin:12px';
      const close=document.createElement('button');close.textContent='关闭图片';close.onclick=()=>dialog.close();
      dialog.append(title,img,a,close);document.body.append(dialog);dialog.onclose=()=>{URL.revokeObjectURL(url);dialog.remove()};dialog.showModal();
      this.status.textContent='PNG 已生成，可在图片预览中下载。';
    },'image/png');
  }
}
