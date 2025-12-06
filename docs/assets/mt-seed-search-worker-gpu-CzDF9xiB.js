function ce(e){return[e>>25&31,e>>20&31,e>>15&31,e>>10&31,e>>5&31,e&31]}const X=256,de=2,le=8,W=1,F=2,pe=4096,fe={requiredFeatures:[]},Z={workgroupSize:256,candidateCapacityPerDispatch:4096},me=de*Uint32Array.BYTES_PER_ELEMENT,ge=4294967295,j={mobile:1,integrated:2,discrete:4,unknown:1},_e=1,$=8;function Q(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function he(e){if(!Q())throw new Error("WebGPU is not available in this environment");const r=await navigator.gpu.requestAdapter({powerPreference:e?.powerPreference});if(!r)throw new Error("Failed to acquire WebGPU adapter");const n={requiredFeatures:e?.requiredFeatures??fe.requiredFeatures,requiredLimits:e?.requiredLimits,label:e?.label},[a,o]=await Promise.all([r.requestDevice(n),Pe(r)]);let u=!1;const f=a.lost.then(i=>(u=!0,console.warn("[webgpu] device lost:",i.message),i)),c=Ee(r,a),l=a.limits;return{getAdapter:()=>r,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>l,getCapabilities:()=>c,getGpuProfile:()=>o,deriveSearchJobLimits:i=>be(c.limits,o,i),isLost:()=>u,waitForLoss:()=>f,getSupportedWorkgroupSize:i=>ee(c.limits,i)}}function Ee(e,t){const r=new Set;return e.features.forEach(n=>r.add(n)),{limits:t.limits,features:r}}function be(e,t,r){const n={...Z,...r},a=De(n),o=ee(e,a.workgroupSize),u=T(e.maxComputeWorkgroupsPerDimension),f=a.maxWorkgroupsPerDispatch??u,c=Math.max(1,Math.floor(ge/Math.max(1,o))),l=v(Math.min(f,u,c),"maxWorkgroupsPerDispatch"),i=o*l,_=a.maxMessagesPerDispatch??i,m=v(Math.min(_,i),"maxMessagesPerDispatch"),g=Math.max(1,Math.floor(T(e.maxStorageBufferBindingSize)/me)),P=a.candidateCapacityPerDispatch??g,h=v(Math.min(P,g),"candidateCapacityPerDispatch"),M=Se(t,a);return{workgroupSize:o,maxWorkgroupsPerDispatch:l,maxMessagesPerDispatch:m,candidateCapacityPerDispatch:h,maxDispatchesInFlight:M}}function ee(e,t){const r=Z.workgroupSize,n=typeof t=="number"&&Number.isFinite(t)&&t>0?Math.floor(t):r,a=T(e.maxComputeWorkgroupSizeX),o=T(e.maxComputeInvocationsPerWorkgroup),u=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(n,u))}function T(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function v(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function Se(e,t){if(typeof t.maxDispatchesInFlight=="number")return v(Math.min(t.maxDispatchesInFlight,$),"maxDispatchesInFlight");const r=e.isFallbackAdapter?_e:j[e.kind]??j.unknown;return v(Math.min(r,$),"maxDispatchesInFlight")}async function Pe(e){const t=ye(),n=!!e.isFallbackAdapter,a=we();if(a){const o={description:a.renderer};return{kind:a.kind,source:"webgl",userAgent:t,adapterInfo:o,isFallbackAdapter:n}}return n?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}}function ye(){return typeof navigator>"u"?"":navigator.userAgent||""}const Re=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],ve=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"],Me=["intel","iris","uhd","hd graphics","radeon graphics","apple"];function z(e,t){return t.some(r=>e.includes(r))}function xe(e){if(!e)return;const t=e.toLowerCase();if(z(t,Re))return"mobile";if(z(t,ve))return"discrete";if(z(t,Me))return"integrated"}function we(){const e=Ae();if(!e)return;const t=xe(e);if(t)return{kind:t,renderer:e}}function Ae(){const e=Ce();if(e)try{const t=Te(e);if(!t)return;const r=t.getExtension("WEBGL_debug_renderer_info");if(!r)return;const n=t.getParameter(r.UNMASKED_RENDERER_WEBGL),a=t.getExtension("WEBGL_lose_context");return a&&a.loseContext(),typeof n=="string"?n:void 0}catch(t){console.warn("[webgpu] webgl renderer detection failed:",t);return}}function Ce(){if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);if(typeof document<"u"&&typeof document.createElement=="function"){const e=document.createElement("canvas");return e.width=1,e.height=1,e}}function Te(e){const t=e,r=t.getContext;if(typeof r!="function")return null;const n=a=>r.call(t,a)??null;return n("webgl2")??n("webgl")}function De(e,t){return e}var Ue=`// MT Seed 32bit全探索用 WGSL シェーダー\r
//\r
// MT19937のSeed空間を全探索し、指定されたIVコードリストにマッチする\r
// MT Seedを検出する。各スレッドが1つのMT Seedを担当。\r
\r
// === 定数 ===\r
const N: u32 = 624u;\r
const M: u32 = 397u;\r
const MATRIX_A: u32 = 0x9908B0DFu;\r
const UPPER_MASK: u32 = 0x80000000u;\r
const LOWER_MASK: u32 = 0x7FFFFFFFu;\r
\r
// ワークグループサイズ（プレースホルダー、TypeScript側で置換）\r
const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
\r
// === バインディング ===\r
\r
// 検索パラメータ\r
struct SearchParams {\r
    start_seed: u32,      // 検索開始Seed\r
    end_seed: u32,        // 検索終了Seed（inclusive）\r
    advances: u32,        // MT消費数\r
    target_count: u32,    // 検索対象IVコード数\r
    max_results: u32,     // 最大結果数\r
    reserved0: u32,\r
    reserved1: u32,\r
    reserved2: u32,\r
}\r
\r
// マッチ結果レコード\r
struct MatchRecord {\r
    seed: u32,\r
    iv_code: u32,\r
}\r
\r
// 結果出力バッファ\r
struct ResultBuffer {\r
    match_count: atomic<u32>,\r
    records: array<MatchRecord>,\r
}\r
\r
@group(0) @binding(0) var<uniform> params: SearchParams;\r
@group(0) @binding(1) var<storage, read> target_codes: array<u32>;\r
@group(0) @binding(2) var<storage, read_write> results: ResultBuffer;\r
\r
// === MT19937 実装 ===\r
// privateメモリにstate配列を保持（約2.5KB）\r
\r
var<private> mt_state: array<u32, 624>;\r
var<private> mt_index: u32;\r
\r
// MT19937初期化\r
fn mt_init(seed: u32) {\r
    mt_state[0] = seed;\r
    for (var i = 1u; i < N; i++) {\r
        let prev = mt_state[i - 1u];\r
        mt_state[i] = 1812433253u * (prev ^ (prev >> 30u)) + i;\r
    }\r
    mt_index = N;\r
}\r
\r
// MT19937 twist操作\r
fn mt_twist() {\r
    for (var i = 0u; i < N; i++) {\r
        let next_idx = (i + 1u) % N;\r
        let m_idx = (i + M) % N;\r
        \r
        let x = (mt_state[i] & UPPER_MASK) | (mt_state[next_idx] & LOWER_MASK);\r
        var x_a = x >> 1u;\r
        if ((x & 1u) != 0u) {\r
            x_a ^= MATRIX_A;\r
        }\r
        mt_state[i] = mt_state[m_idx] ^ x_a;\r
    }\r
    mt_index = 0u;\r
}\r
\r
// MT19937 次の乱数を取得\r
fn mt_next() -> u32 {\r
    if (mt_index >= N) {\r
        mt_twist();\r
    }\r
    \r
    var y = mt_state[mt_index];\r
    mt_index++;\r
    \r
    // Tempering\r
    y ^= y >> 11u;\r
    y ^= (y << 7u) & 0x9D2C5680u;\r
    y ^= (y << 15u) & 0xEFC60000u;\r
    y ^= y >> 18u;\r
    \r
    return y;\r
}\r
\r
// === IVコードエンコード ===\r
// 6ステータスのIV（各5bit）を30bitの整数にパック\r
// 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]\r
\r
fn encode_iv_code(hp: u32, atk: u32, def: u32, spa: u32, spd: u32, spe: u32) -> u32 {\r
    return (hp << 25u) | (atk << 20u) | (def << 15u) | (spa << 10u) | (spd << 5u) | spe;\r
}\r
\r
// === 線形探索 ===\r
// IVコード数は最大1024件のため、GPUの並列性を活かせば線形探索で十分高速\r
// 二分探索は分岐コストが高く、却って遅くなる可能性がある\r
\r
fn linear_search(code: u32) -> bool {\r
    for (var i = 0u; i < params.target_count; i++) {\r
        if (target_codes[i] == code) {\r
            return true;\r
        }\r
    }\r
    return false;\r
}\r
\r
// === メインエントリポイント ===\r
\r
@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)\r
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\r
    // 担当するMT Seed を計算\r
    let seed = params.start_seed + global_id.x;\r
    \r
    // 範囲外チェック（オーバーフロー含む）\r
    if (seed < params.start_seed || seed > params.end_seed) {\r
        return;\r
    }\r
    \r
    // MT19937初期化\r
    mt_init(seed);\r
    \r
    // MT消費（advances回）\r
    for (var i = 0u; i < params.advances; i++) {\r
        _ = mt_next();\r
    }\r
    \r
    // IV取得（6回の乱数取得、上位5bit）\r
    let hp  = mt_next() >> 27u;\r
    let atk = mt_next() >> 27u;\r
    let def = mt_next() >> 27u;\r
    let spa = mt_next() >> 27u;\r
    let spd = mt_next() >> 27u;\r
    let spe = mt_next() >> 27u;\r
    \r
    // IVコードにエンコード\r
    let code = encode_iv_code(hp, atk, def, spa, spd, spe);\r
    \r
    // 線形探索でマッチング判定\r
    if (linear_search(code)) {\r
        // アトミックにカウンタをインクリメントして結果を格納\r
        let idx = atomicAdd(&results.match_count, 1u);\r
        \r
        // バッファオーバーフロー防止\r
        if (idx < params.max_results) {\r
            results.records[idx].seed = seed;\r
            results.records[idx].iv_code = code;\r
        }\r
    }\r
}\r
`;function C(e){return Math.ceil(e/X)*X}function Le(e){const t=e?.workgroupSize??256,r=pe;let n=null,a=null,o=null,u=null,f=null,c=null,l=null,i=null,_=0,m=t;const g=()=>Q(),P=async()=>{if(!g())throw new Error("WebGPU is not available in this environment");n=await he({powerPreference:"high-performance",label:"mt-seed-search-device"}),a=n.deriveSearchJobLimits({workgroupSize:t,candidateCapacityPerDispatch:r}),m=a.workgroupSize;const d=n.getDevice(),p=Ue.replace(/WORKGROUP_SIZE_PLACEHOLDERu/g,`${m}u`).replace(/WORKGROUP_SIZE_PLACEHOLDER/g,`${m}`),E=d.createShaderModule({label:"mt-seed-search-shader",code:p});u=d.createBindGroupLayout({label:"mt-seed-search-bind-group-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});const b=d.createPipelineLayout({label:"mt-seed-search-pipeline-layout",bindGroupLayouts:[u]});o=d.createComputePipeline({label:"mt-seed-search-pipeline",layout:b,compute:{module:E,entryPoint:"main"}});const D=C(le*Uint32Array.BYTES_PER_ELEMENT);f=d.createBuffer({label:"mt-seed-search-params",size:D,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const w=W+r*F,R=C(w*Uint32Array.BYTES_PER_ELEMENT);l=d.createBuffer({label:"mt-seed-search-results",size:R,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),i=d.createBuffer({label:"mt-seed-search-readback",size:R,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})},h=d=>{if(!n)throw new Error("Engine not initialized");const p=n.getDevice(),E=d.length;if(c&&_>=E)return;c?.destroy();const b=C(E*Uint32Array.BYTES_PER_ELEMENT);c=p.createBuffer({label:"mt-seed-search-target-codes",size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),_=E};return{initialize:P,executeJob:async d=>{if(!n||!o||!u||!f||!l||!i||!a)throw new Error("Engine not initialized");const p=n.getDevice(),{searchRange:E,ivCodes:b,mtAdvances:D}=d,{start:w,end:R}=E,K=R-w+1,re=Math.ceil(K/m);h(b);const U=new Uint32Array(b);p.queue.writeBuffer(c,0,U.buffer,U.byteOffset,U.byteLength);const L=new Uint32Array([w,R,D,b.length,r,0,0,0]);p.queue.writeBuffer(f,0,L.buffer,L.byteOffset,L.byteLength);const k=new Uint32Array([0]);p.queue.writeBuffer(l,0,k.buffer,k.byteOffset,k.byteLength);const ne=p.createBindGroup({label:`mt-seed-search-bind-group-${d.jobId}`,layout:u,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:l}}]}),G=p.createCommandEncoder({label:`mt-seed-search-encoder-${d.jobId}`}),A=G.beginComputePass({label:`mt-seed-search-pass-${d.jobId}`});A.setPipeline(o),A.setBindGroup(0,ne),A.dispatchWorkgroups(re),A.end();const se=W+r*F,O=C(se*Uint32Array.BYTES_PER_ELEMENT);G.copyBufferToBuffer(l,0,i,0,O);const ae=G.finish();p.queue.submit([ae]),await i.mapAsync(GPUMapMode.READ,0,O);const ie=i.getMappedRange(0,O),I=new Uint32Array(ie.slice(0));i.unmap();const Y=Math.min(I[0]??0,r),H=[];for(let B=0;B<Y;B++){const J=W+B*F,oe=I[J],V=I[J+1],ue=ce(V);H.push({mtSeed:oe,ivCode:V,ivSet:ue})}return{matches:H,matchCount:Y,processedCount:K}},dispose:()=>{f?.destroy(),c?.destroy(),l?.destroy(),i?.destroy(),f=null,c=null,l=null,i=null,_=0,n=null,o=null,u=null,a=null},isAvailable:g,getWorkgroupSize:()=>m,getJobLimits:()=>a}}const s={job:null,engine:null,running:!1,stopRequested:!1,isPaused:!1,pauseResolve:null},N=self,S=e=>N.postMessage(e);S({type:"READY",version:"1"});N.onmessage=e=>{const t=e.data;(async()=>{try{switch(t.type){case"START":await Ie(t.job);break;case"PAUSE":ke();break;case"RESUME":Ge();break;case"STOP":s.stopRequested=!0,s.isPaused&&s.pauseResolve&&(s.pauseResolve(),s.pauseResolve=null,s.isPaused=!1);break;default:break}}catch(r){const n=r instanceof Error?r.message:String(r);S({type:"ERROR",message:n,category:"RUNTIME"})}})()};function ke(){!s.running||s.isPaused||(s.isPaused=!0)}function Ge(){!s.running||!s.isPaused||(s.isPaused=!1,s.pauseResolve&&(s.pauseResolve(),s.pauseResolve=null))}async function Oe(){await new Promise(e=>setTimeout(e,0)),s.isPaused&&await new Promise(e=>{s.pauseResolve=e})}async function Ie(e){if(s.running)return;s.job=e,s.stopRequested=!1,s.running=!0;const t=performance.now();try{if(!s.engine){if(s.engine=Le(),!s.engine.isAvailable())throw new Error("WebGPU is not available");await s.engine.initialize()}const r=await We(e,t),n={reason:s.stopRequested?"stopped":"finished",totalProcessed:r.processedCount,totalMatches:r.matchesFound,elapsedMs:performance.now()-t};S({type:"COMPLETE",payload:n})}catch(r){const n=r instanceof Error?r.message:String(r),a=n.includes("WebGPU")?"GPU_INIT":"RUNTIME";S({type:"ERROR",message:n,category:a})}finally{te()}}const Be=500;async function We(e,t){const r=s.engine;if(!r)throw new Error("GPU engine not initialized");const{searchRange:n,ivCodes:a,mtAdvances:o,jobId:u}=e,{start:f,end:c}=n,l=r.getJobLimits(),i=l?l.maxMessagesPerDispatch:16777216,_=c-f+1;let m=0,g=0,P=t,h=f;for(;h<=c&&(await Oe(),!s.stopRequested);){const x=Math.min(h+i-1,c),q={searchRange:{start:h,end:x},ivCodes:a,mtAdvances:o,jobId:u},y=await r.executeJob(q);if(y.matches.length>0){const p={jobId:u,matches:y.matches};S({type:"RESULTS",payload:p}),g+=y.matches.length}m+=y.processedCount;const d=performance.now();if(d-P>=Be){const p={jobId:u,processedCount:m,totalCount:_,elapsedMs:d-t,matchesFound:g};S({type:"PROGRESS",payload:p}),P=d}if(x>=c)break;h=x+1}const M={jobId:u,processedCount:m,totalCount:_,elapsedMs:performance.now()-t,matchesFound:g};return S({type:"PROGRESS",payload:M}),{processedCount:m,matchesFound:g}}function te(){s.running=!1,s.job=null,s.stopRequested=!1,s.isPaused=!1,s.pauseResolve=null}N.onclose=()=>{s.engine?.dispose(),s.engine=null,te()};
