function ce(e){return[e>>25&31,e>>20&31,e>>15&31,e>>10&31,e>>5&31,e&31]}const X=256,de=2,le=8,W=1,F=2,pe=4096,fe={requiredFeatures:[]},Z={workgroupSize:256,candidateCapacityPerDispatch:4096},me=de*Uint32Array.BYTES_PER_ELEMENT,ge=4294967295,j={mobile:1,integrated:2,discrete:4,unknown:1},_e=1,$=8;function Q(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function he(e){if(!Q())throw new Error("WebGPU is not available in this environment");const n=await navigator.gpu.requestAdapter({powerPreference:e?.powerPreference});if(!n)throw new Error("Failed to acquire WebGPU adapter");const r={requiredFeatures:e?.requiredFeatures??fe.requiredFeatures,requiredLimits:e?.requiredLimits,label:e?.label},[a,o]=await Promise.all([n.requestDevice(r),Pe(n)]);let u=!1;const f=a.lost.then(i=>(u=!0,console.warn("[webgpu] device lost:",i.message),i)),c=Ee(n,a),l=a.limits;return{getAdapter:()=>n,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>l,getCapabilities:()=>c,getGpuProfile:()=>o,deriveSearchJobLimits:i=>be(c.limits,o,i),isLost:()=>u,waitForLoss:()=>f,getSupportedWorkgroupSize:i=>ee(c.limits,i)}}function Ee(e,t){const n=new Set;return e.features.forEach(r=>n.add(r)),{limits:t.limits,features:n}}function be(e,t,n){const r={...Z,...n},a=De(r),o=ee(e,a.workgroupSize),u=T(e.maxComputeWorkgroupsPerDimension),f=a.maxWorkgroupsPerDispatch??u,c=Math.max(1,Math.floor(ge/Math.max(1,o))),l=v(Math.min(f,u,c),"maxWorkgroupsPerDispatch"),i=o*l,_=a.maxMessagesPerDispatch??i,m=v(Math.min(_,i),"maxMessagesPerDispatch"),g=Math.max(1,Math.floor(T(e.maxStorageBufferBindingSize)/me)),P=a.candidateCapacityPerDispatch??g,h=v(Math.min(P,g),"candidateCapacityPerDispatch"),M=Se(t,a);return{workgroupSize:o,maxWorkgroupsPerDispatch:l,maxMessagesPerDispatch:m,candidateCapacityPerDispatch:h,maxDispatchesInFlight:M}}function ee(e,t){const n=Z.workgroupSize,r=typeof t=="number"&&Number.isFinite(t)&&t>0?Math.floor(t):n,a=T(e.maxComputeWorkgroupSizeX),o=T(e.maxComputeInvocationsPerWorkgroup),u=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(r,u))}function T(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function v(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function Se(e,t){if(typeof t.maxDispatchesInFlight=="number")return v(Math.min(t.maxDispatchesInFlight,$),"maxDispatchesInFlight");const n=e.isFallbackAdapter?_e:j[e.kind]??j.unknown;return v(Math.min(n,$),"maxDispatchesInFlight")}async function Pe(e){const t=ye(),r=!!e.isFallbackAdapter,a=we();if(a){const o={description:a.renderer};return{kind:a.kind,source:"webgl",userAgent:t,adapterInfo:o,isFallbackAdapter:r}}return r?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:void 0,isFallbackAdapter:r}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:void 0,isFallbackAdapter:r}}function ye(){return typeof navigator>"u"?"":navigator.userAgent||""}const Re=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],ve=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"],Me=["intel","iris","uhd","hd graphics","radeon graphics","apple"];function z(e,t){return t.some(n=>e.includes(n))}function xe(e){if(!e)return;const t=e.toLowerCase();if(z(t,Re))return"mobile";if(z(t,ve))return"discrete";if(z(t,Me))return"integrated"}function we(){const e=Ae();if(!e)return;const t=xe(e);if(t)return{kind:t,renderer:e}}function Ae(){const e=Ce();if(e)try{const t=Te(e);if(!t)return;const n=t.getExtension("WEBGL_debug_renderer_info");if(!n)return;const r=t.getParameter(n.UNMASKED_RENDERER_WEBGL),a=t.getExtension("WEBGL_lose_context");return a&&a.loseContext(),typeof r=="string"?r:void 0}catch(t){console.warn("[webgpu] webgl renderer detection failed:",t);return}}function Ce(){if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);if(typeof document<"u"&&typeof document.createElement=="function"){const e=document.createElement("canvas");return e.width=1,e.height=1,e}}function Te(e){const t=e,n=t.getContext;if(typeof n!="function")return null;const r=a=>n.call(t,a)??null;return r("webgl2")??r("webgl")}function De(e,t){return e}var Ue=`// MT Seed 32bit全探索用 WGSL シェーダー
//
// MT19937のSeed空間を全探索し、指定されたIVコードリストにマッチする
// MT Seedを検出する。各スレッドが1つのMT Seedを担当。

// === 定数 ===
const N: u32 = 624u;
const M: u32 = 397u;
const MATRIX_A: u32 = 0x9908B0DFu;
const UPPER_MASK: u32 = 0x80000000u;
const LOWER_MASK: u32 = 0x7FFFFFFFu;

// ワークグループサイズ（プレースホルダー、TypeScript側で置換）
const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_PLACEHOLDERu;

// === バインディング ===

// 検索パラメータ
struct SearchParams {
    start_seed: u32,      // 検索開始Seed
    end_seed: u32,        // 検索終了Seed（inclusive）
    advances: u32,        // MT消費数
    target_count: u32,    // 検索対象IVコード数
    max_results: u32,     // 最大結果数
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

// マッチ結果レコード
struct MatchRecord {
    seed: u32,
    iv_code: u32,
}

// 結果出力バッファ
struct ResultBuffer {
    match_count: atomic<u32>,
    records: array<MatchRecord>,
}

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<storage, read> target_codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: ResultBuffer;

// === MT19937 実装 ===
// privateメモリにstate配列を保持（約2.5KB）

var<private> mt_state: array<u32, 624>;
var<private> mt_index: u32;

// MT19937初期化
fn mt_init(seed: u32) {
    mt_state[0] = seed;
    for (var i = 1u; i < N; i++) {
        let prev = mt_state[i - 1u];
        mt_state[i] = 1812433253u * (prev ^ (prev >> 30u)) + i;
    }
    mt_index = N;
}

// MT19937 twist操作
fn mt_twist() {
    for (var i = 0u; i < N; i++) {
        let next_idx = (i + 1u) % N;
        let m_idx = (i + M) % N;
        
        let x = (mt_state[i] & UPPER_MASK) | (mt_state[next_idx] & LOWER_MASK);
        var x_a = x >> 1u;
        if ((x & 1u) != 0u) {
            x_a ^= MATRIX_A;
        }
        mt_state[i] = mt_state[m_idx] ^ x_a;
    }
    mt_index = 0u;
}

// MT19937 次の乱数を取得
fn mt_next() -> u32 {
    if (mt_index >= N) {
        mt_twist();
    }
    
    var y = mt_state[mt_index];
    mt_index++;
    
    // Tempering
    y ^= y >> 11u;
    y ^= (y << 7u) & 0x9D2C5680u;
    y ^= (y << 15u) & 0xEFC60000u;
    y ^= y >> 18u;
    
    return y;
}

// === IVコードエンコード ===
// 6ステータスのIV（各5bit）を30bitの整数にパック
// 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]

fn encode_iv_code(hp: u32, atk: u32, def: u32, spa: u32, spd: u32, spe: u32) -> u32 {
    return (hp << 25u) | (atk << 20u) | (def << 15u) | (spa << 10u) | (spd << 5u) | spe;
}

// === 線形探索 ===
// IVコード数は最大1024件のため、GPUの並列性を活かせば線形探索で十分高速
// 二分探索は分岐コストが高く、却って遅くなる可能性がある

fn linear_search(code: u32) -> bool {
    for (var i = 0u; i < params.target_count; i++) {
        if (target_codes[i] == code) {
            return true;
        }
    }
    return false;
}

// === メインエントリポイント ===

@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 担当するMT Seed を計算
    let seed = params.start_seed + global_id.x;
    
    // 範囲外チェック（オーバーフロー含む）
    if (seed < params.start_seed || seed > params.end_seed) {
        return;
    }
    
    // MT19937初期化
    mt_init(seed);
    
    // MT消費（advances回）
    for (var i = 0u; i < params.advances; i++) {
        _ = mt_next();
    }
    
    // IV取得（6回の乱数取得、上位5bit）
    let hp  = mt_next() >> 27u;
    let atk = mt_next() >> 27u;
    let def = mt_next() >> 27u;
    let spa = mt_next() >> 27u;
    let spd = mt_next() >> 27u;
    let spe = mt_next() >> 27u;
    
    // IVコードにエンコード
    let code = encode_iv_code(hp, atk, def, spa, spd, spe);
    
    // 線形探索でマッチング判定
    if (linear_search(code)) {
        // アトミックにカウンタをインクリメントして結果を格納
        let idx = atomicAdd(&results.match_count, 1u);
        
        // バッファオーバーフロー防止
        if (idx < params.max_results) {
            results.records[idx].seed = seed;
            results.records[idx].iv_code = code;
        }
    }
}
`;function C(e){return Math.ceil(e/X)*X}function Le(e){const t=e?.workgroupSize??256,n=pe;let r=null,a=null,o=null,u=null,f=null,c=null,l=null,i=null,_=0,m=t;const g=()=>Q(),P=async()=>{if(!g())throw new Error("WebGPU is not available in this environment");r=await he({powerPreference:"high-performance",label:"mt-seed-search-device"}),a=r.deriveSearchJobLimits({workgroupSize:t,candidateCapacityPerDispatch:n}),m=a.workgroupSize;const d=r.getDevice(),p=Ue.replace(/WORKGROUP_SIZE_PLACEHOLDERu/g,`${m}u`).replace(/WORKGROUP_SIZE_PLACEHOLDER/g,`${m}`),E=d.createShaderModule({label:"mt-seed-search-shader",code:p});u=d.createBindGroupLayout({label:"mt-seed-search-bind-group-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});const b=d.createPipelineLayout({label:"mt-seed-search-pipeline-layout",bindGroupLayouts:[u]});o=d.createComputePipeline({label:"mt-seed-search-pipeline",layout:b,compute:{module:E,entryPoint:"main"}});const D=C(le*Uint32Array.BYTES_PER_ELEMENT);f=d.createBuffer({label:"mt-seed-search-params",size:D,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const w=W+n*F,R=C(w*Uint32Array.BYTES_PER_ELEMENT);l=d.createBuffer({label:"mt-seed-search-results",size:R,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),i=d.createBuffer({label:"mt-seed-search-readback",size:R,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})},h=d=>{if(!r)throw new Error("Engine not initialized");const p=r.getDevice(),E=d.length;if(c&&_>=E)return;c?.destroy();const b=C(E*Uint32Array.BYTES_PER_ELEMENT);c=p.createBuffer({label:"mt-seed-search-target-codes",size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),_=E};return{initialize:P,executeJob:async d=>{if(!r||!o||!u||!f||!l||!i||!a)throw new Error("Engine not initialized");const p=r.getDevice(),{searchRange:E,ivCodes:b,mtAdvances:D}=d,{start:w,end:R}=E,K=R-w+1,ne=Math.ceil(K/m);h(b);const U=new Uint32Array(b);p.queue.writeBuffer(c,0,U.buffer,U.byteOffset,U.byteLength);const L=new Uint32Array([w,R,D,b.length,n,0,0,0]);p.queue.writeBuffer(f,0,L.buffer,L.byteOffset,L.byteLength);const k=new Uint32Array([0]);p.queue.writeBuffer(l,0,k.buffer,k.byteOffset,k.byteLength);const re=p.createBindGroup({label:`mt-seed-search-bind-group-${d.jobId}`,layout:u,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:l}}]}),G=p.createCommandEncoder({label:`mt-seed-search-encoder-${d.jobId}`}),A=G.beginComputePass({label:`mt-seed-search-pass-${d.jobId}`});A.setPipeline(o),A.setBindGroup(0,re),A.dispatchWorkgroups(ne),A.end();const se=W+n*F,O=C(se*Uint32Array.BYTES_PER_ELEMENT);G.copyBufferToBuffer(l,0,i,0,O);const ae=G.finish();p.queue.submit([ae]),await i.mapAsync(GPUMapMode.READ,0,O);const ie=i.getMappedRange(0,O),I=new Uint32Array(ie.slice(0));i.unmap();const Y=Math.min(I[0]??0,n),H=[];for(let B=0;B<Y;B++){const J=W+B*F,oe=I[J],V=I[J+1],ue=ce(V);H.push({mtSeed:oe,ivCode:V,ivSet:ue})}return{matches:H,matchCount:Y,processedCount:K}},dispose:()=>{f?.destroy(),c?.destroy(),l?.destroy(),i?.destroy(),f=null,c=null,l=null,i=null,_=0,r=null,o=null,u=null,a=null},isAvailable:g,getWorkgroupSize:()=>m,getJobLimits:()=>a}}const s={job:null,engine:null,running:!1,stopRequested:!1,isPaused:!1,pauseResolve:null},N=self,S=e=>N.postMessage(e);S({type:"READY",version:"1"});N.onmessage=e=>{const t=e.data;(async()=>{try{switch(t.type){case"START":await Ie(t.job);break;case"PAUSE":ke();break;case"RESUME":Ge();break;case"STOP":s.stopRequested=!0,s.isPaused&&s.pauseResolve&&(s.pauseResolve(),s.pauseResolve=null,s.isPaused=!1);break;default:break}}catch(n){const r=n instanceof Error?n.message:String(n);S({type:"ERROR",message:r,category:"RUNTIME"})}})()};function ke(){!s.running||s.isPaused||(s.isPaused=!0)}function Ge(){!s.running||!s.isPaused||(s.isPaused=!1,s.pauseResolve&&(s.pauseResolve(),s.pauseResolve=null))}async function Oe(){await new Promise(e=>setTimeout(e,0)),s.isPaused&&await new Promise(e=>{s.pauseResolve=e})}async function Ie(e){if(s.running)return;s.job=e,s.stopRequested=!1,s.running=!0;const t=performance.now();try{if(!s.engine){if(s.engine=Le(),!s.engine.isAvailable())throw new Error("WebGPU is not available");await s.engine.initialize()}const n=await We(e,t),r={reason:s.stopRequested?"stopped":"finished",totalProcessed:n.processedCount,totalMatches:n.matchesFound,elapsedMs:performance.now()-t};S({type:"COMPLETE",payload:r})}catch(n){const r=n instanceof Error?n.message:String(n),a=r.includes("WebGPU")?"GPU_INIT":"RUNTIME";S({type:"ERROR",message:r,category:a})}finally{te()}}const Be=500;async function We(e,t){const n=s.engine;if(!n)throw new Error("GPU engine not initialized");const{searchRange:r,ivCodes:a,mtAdvances:o,jobId:u}=e,{start:f,end:c}=r,l=n.getJobLimits(),i=l?l.maxMessagesPerDispatch:16777216,_=c-f+1;let m=0,g=0,P=t,h=f;for(;h<=c&&(await Oe(),!s.stopRequested);){const x=Math.min(h+i-1,c),q={searchRange:{start:h,end:x},ivCodes:a,mtAdvances:o,jobId:u},y=await n.executeJob(q);if(y.matches.length>0){const p={jobId:u,matches:y.matches};S({type:"RESULTS",payload:p}),g+=y.matches.length}m+=y.processedCount;const d=performance.now();if(d-P>=Be){const p={jobId:u,processedCount:m,totalCount:_,elapsedMs:d-t,matchesFound:g};S({type:"PROGRESS",payload:p}),P=d}if(x>=c)break;h=x+1}const M={jobId:u,processedCount:m,totalCount:_,elapsedMs:performance.now()-t,matchesFound:g};return S({type:"PROGRESS",payload:M}),{processedCount:m,matchesFound:g}}function te(){s.running=!1,s.job=null,s.stopRequested=!1,s.isPaused=!1,s.pauseResolve=null}N.onclose=()=>{s.engine?.dispose(),s.engine=null,te()};
