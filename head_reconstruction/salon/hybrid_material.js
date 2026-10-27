import * as THREE from 'three';

// A real fitted mesh supplies shape and depth. Shading represents hairs too
// short/thin to resolve as separate tubes; it is never projected onto the face.
export function hybridMaterial(texture, volume=false, centre=[0,0,0], litSkin=false){
  const material=new THREE.MeshStandardMaterial({map:texture,roughness:volume ? .82 : .95,
    side:THREE.DoubleSide,transparent:false,depthWrite:true,
    alphaTest:volume ? .18 : 0,alphaToCoverage:volume});
  material.userData.hairColor={value:new THREE.Color('#252329')};
  material.onBeforeCompile=shader=>{
    shader.uniforms.hybridHairColor=material.userData.hairColor;
    shader.uniforms.hybridVolume={value:volume?1:0};
    shader.uniforms.hybridCentre={value:new THREE.Vector3(...centre)};
    shader.uniforms.hybridLitSkin={value:litSkin?1:0};
    shader.vertexShader=`attribute vec2 hybridParameters;
attribute vec3 scalpPosition;
attribute vec3 hairFlow;
varying vec2 vHybrid;
varying vec3 vScalp;
varying vec3 vFlow;
${shader.vertexShader}`.replace('#include <begin_vertex>',`#include <begin_vertex>
vHybrid=hybridParameters;vScalp=scalpPosition;vFlow=hairFlow;`);
    shader.fragmentShader=`uniform vec3 hybridHairColor;
uniform float hybridVolume;
uniform vec3 hybridCentre;
uniform float hybridLitSkin;
varying vec2 vHybrid;
varying vec3 vScalp;
varying vec3 vFlow;
float hHash(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float hNoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
return mix(mix(mix(hHash(i),hHash(i+vec3(1,0,0)),f.x),mix(hHash(i+vec3(0,1,0)),hHash(i+vec3(1,1,0)),f.x),f.y),
mix(mix(hHash(i+vec3(0,0,1)),hHash(i+vec3(1,0,1)),f.x),mix(hHash(i+vec3(0,1,1)),hHash(i+vec3(1,1,1)),f.x),f.y),f.z);}
${shader.fragmentShader}`;
    shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>',`#include <map_fragment>
vec3 hybridSkin=diffuseColor.rgb;
vec3 hp=(vScalp-hybridCentre)*1000.0;
float fine=hNoise(hp*vec3(2.6,.9,2.6));
float broad=hNoise(hp*.42);
vec3 flow=normalize(vFlow+vec3(.0001));
vec3 crossFlow=normalize(cross(flow,normalize(vScalp-hybridCentre))+vec3(.0001));
float phase=dot(hp,crossFlow)*4.8+broad*2.0;
float filterWidth=max(fwidth(phase),.01);
float ridge=.5+.5*sin(phase)*exp(-filterWidth*filterWidth*.6);
float textureValue=mix(.76+fine*.42,.70+ridge*.36+broad*.15,hybridVolume);
float lengthBlend=smoothstep(.6,4.8,vHybrid.y);
float shortDensity=mix(.14,.995,pow(lengthBlend,.7));
float hybridCoverage=clamp(vHybrid.x,0.0,1.0);
diffuseColor.rgb=hybridHairColor*textureValue;
if(hybridVolume>.5){diffuseColor.a=hybridCoverage*.94;if(diffuseColor.a<.015)discard;}
if(hybridVolume<.5 && hybridCoverage<.001)discard;
float hybridBump=(ridge*.55+fine*.45)*mix(.000018,.000060,hybridVolume)*hybridCoverage;`);
    shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',`#include <normal_fragment_maps>
vec3 hdpx=dFdx(-vViewPosition),hdpy=dFdy(-vViewPosition);
vec3 hr1=cross(hdpy,normal),hr2=cross(normal,hdpx);
float hdet=dot(hdpx,hr1);
normal=normalize(abs(hdet)*normal-sign(hdet)*(dFdx(hybridBump)*hr1+dFdy(hybridBump)*hr2));`);
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
if(hybridVolume<.5){
  float fibreCoverage=hybridCoverage*shortDensity*(.78+.22*smoothstep(.20,.65,fine));
  vec3 skinLight=mix(hybridSkin,totalDiffuse*hybridSkin/max(diffuseColor.rgb,vec3(.0003))+totalSpecular,hybridLitSkin);
  outgoingLight=mix(skinLight,outgoingLight,fibreCoverage);
}
#include <opaque_fragment>`);
  };
  material.customProgramCacheKey=()=>`hybrid-hair-surface-v4-ear-clean-${volume}`;
  return material;
}
