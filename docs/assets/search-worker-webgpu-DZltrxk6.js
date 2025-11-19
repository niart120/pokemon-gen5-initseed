const Oe={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},Ue=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],ee=Ue.reduce((t,[n,r])=>(t[n]=r,t),{}),Ke=Ue.length,qe=(1<<Ke)-1,ze=12287,je=[1<<ee["[↑]"]|1<<ee["[↓]"],1<<ee["[←]"]|1<<ee["[→]"],1<<ee.Select|1<<ee.Start|1<<ee.L|1<<ee.R];function Le(t,n){return Number.isFinite(t)?t&qe:0}function Ze(t){const n=Le(t);return ze^n}function Je(t){const n=Le(t);for(const r of je)if((n&r)===r)return!0;return!1}function Xe(t){return Ze(t)}const me=1e3,be=60,ke=60,Fe=24,Ie=be*ke,Qe=Ie*Fe,Ge=Qe*me;function et(t){const n=t.timeRange;if(!n)throw new Error("timeRange is required for WebGPU search");const r=we("hour",n.hour,0,Fe-1),s=we("minute",n.minute,0,ke-1),o=we("second",n.second,0,be-1),e=new Date(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,0,0,0),l=new Date(t.dateRange.endYear,t.dateRange.endMonth-1,t.dateRange.endDay,0,0,0),u=e.getTime(),i=l.getTime();if(u>i)throw new Error("開始日が終了日より後に設定されています");const c=Math.floor((i-u)/Ge)+1;if(c<=0)throw new Error("探索日数が検出できませんでした");const f=r.count*s.count*o.count;if(f<=0)throw new Error("時刻レンジの組み合わせ数が0です");const _=new Date(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,r.start,s.start,o.start,0);return{plan:{dayCount:c,combosPerDay:f,hourRangeStart:r.start,hourRangeCount:r.count,minuteRangeStart:s.start,minuteRangeCount:s.count,secondRangeStart:o.start,secondRangeCount:o.count,startDayTimestampMs:u},firstCombinationDate:_}}function xe(t,n){const r=Math.max(t.minuteRangeCount,1),s=Math.max(t.secondRangeCount,1),o=Math.max(t.combosPerDay,1),e=Math.max(0,Math.trunc(n)),l=Math.floor(e/o),u=e-l*o,i=r*s,c=Math.floor(u/i),f=u-c*i,_=Math.floor(f/s),w=f-_*s,y=t.hourRangeStart+c,C=t.minuteRangeStart+_,P=t.secondRangeStart+w,O=t.startDayTimestampMs+l*Ge+y*Ie*me+C*be*me+P*me;return new Date(O)}function we(t,n,r,s){if(!n)throw new Error(`${t} range is required for WebGPU search`);const o=Math.trunc(n.start),e=Math.trunc(n.end);if(Number.isNaN(o)||Number.isNaN(e))throw new Error(`${t} range must be numeric`);if(o<r||e>s)throw new Error(`${t} range must be within ${r} to ${s}`);if(o>e)throw new Error(`${t} range start must be <= end`);return{start:o,end:e,count:e-o+1}}const tt=Date.UTC(2e3,0,1,0,0,0),nt=100663296,rt={DS:8,DS_LITE:6,"3DS":9};function at(t){const n=[];for(let e=0;e<12;e++)(t&1<<e)!==0&&n.push(e);const r=n.length,s=1<<r,o=[];for(let e=0;e<s;e++){let l=0;for(let i=0;i<r;i++)(e&1<<i)!==0&&(l|=1<<n[i]);if(Je(l))continue;const u=l^ze;o.push(u)}return o}function ot(t){const{plan:n,firstCombinationDate:r}=et(t),s=r,o=st(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond);if(s.getTime()>o.getTime())throw new Error("開始日時が終了日時より後ろに設定されています");const e=it(t),l=ct(t,e),u=ut(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,n.hourRangeStart,n.minuteRangeStart,n.secondRangeStart),i=Math.floor((u-tt)/1e3);if(i<0)throw new Error("2000年より前の日時は指定できません");const c=n.dayCount*n.combosPerDay;if(c<=0)throw new Error("探索秒数が0秒以下です");const f=s.getFullYear(),_=_t(s),w=pt(s),y=s.getDay(),C=rt[t.hardware],{macLower:P,data7Swapped:x}=dt(t.macAddress,C),O=at(t.keyInput),S=gt(e.nazo),U=[];let ne=0;for(const N of O){const q=Me(N>>>0);for(let j=0;j<l.length;j+=1){const z=l[j],H=z.timer0Max-z.timer0Min+1,a=c*H,p={startSecondsSince2000:i>>>0,rangeSeconds:c>>>0,timer0Min:z.timer0Min>>>0,timer0Max:z.timer0Max>>>0,timer0Count:H>>>0,vcountMin:z.vcount>>>0,vcountMax:z.vcount>>>0,vcountCount:1,totalMessages:a>>>0,hardwareType:mt(t.hardware),macLower:P>>>0,data7Swapped:x>>>0,keyInputSwapped:q>>>0,nazoSwapped:S,startYear:f>>>0,startDayOfYear:_>>>0,startSecondOfDay:w>>>0,startDayOfWeek:y>>>0,dayCount:n.dayCount>>>0,hourRangeStart:n.hourRangeStart>>>0,hourRangeCount:n.hourRangeCount>>>0,minuteRangeStart:n.minuteRangeStart>>>0,minuteRangeCount:n.minuteRangeCount>>>0,secondRangeStart:n.secondRangeStart>>>0,secondRangeCount:n.secondRangeCount>>>0};U.push({index:j,baseOffset:ne,timer0Min:z.timer0Min,timer0Max:z.timer0Max,timer0Count:H,vcount:z.vcount,rangeSeconds:c,totalMessages:a,keyCode:N,config:p}),ne+=a}}const X=U.reduce((N,q)=>N+q.totalMessages,0);return{conditions:t,startDate:s,startTimestampMs:s.getTime(),rangeSeconds:c,totalMessages:X,segments:U,timePlan:n}}function st(t,n,r,s,o,e){return new Date(t,n-1,r,s,o,e)}function ut(t,n,r,s,o,e){return Date.UTC(t,n,r,s,o,e,0)}function it(t){const n=Oe[t.romVersion];if(!n)throw new Error(`ROMバージョン ${t.romVersion} は未対応です`);const r=n[t.romRegion];if(!r)throw new Error(`ROMリージョン ${t.romRegion} は未対応です`);return{nazo:[...r.nazo],vcountTimerRanges:r.vcountTimerRanges.map(s=>[...s])}}function ct(t,n){const r=[],s=t.timer0VCountConfig.timer0Range.min,o=t.timer0VCountConfig.timer0Range.max;let e=null;for(let l=s;l<=o;l+=1){const u=lt(n,l);e&&e.vcount===u&&l===e.timer0Max+1?e.timer0Max=l:(e&&r.push(e),e={timer0Min:l,timer0Max:l,vcount:u})}return e&&r.push(e),r}function lt(t,n){for(const[r,s,o]of t.vcountTimerRanges)if(n>=s&&n<=o)return r;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}function dt(t,n){const r=ft(t),s=(r[4]&255)<<8|r[5]&255,e=((r[0]&255|(r[1]&255)<<8|(r[2]&255)<<16|(r[3]&255)<<24)^nt^n)>>>0;return{macLower:s,data7Swapped:Me(e)}}function ft(t){const n=new Array(6).fill(0);for(let r=0;r<6;r+=1){const s=t[r]??0;n[r]=(Number(s)&255)>>>0}return n}function mt(t){switch(t){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function Me(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}function gt(t){const n=new Uint32Array(t.length);for(let r=0;r<t.length;r+=1)n[r]=Me(t[r]>>>0);return n}function _t(t){const n=new Date(t.getFullYear(),0,1),r=t.getTime()-n.getTime();return Math.floor(r/(1440*60*1e3))+1}function pt(t){return t.getHours()*3600+t.getMinutes()*60+t.getSeconds()}class De{calculateHash(n){if(n.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const r=1732584193,s=4023233417,o=2562383102,e=271733878,l=3285377520,u=new Array(80);for(let S=0;S<16;S++)u[S]=n[S];for(let S=16;S<80;S++)u[S]=this.leftRotate(u[S-3]^u[S-8]^u[S-14]^u[S-16],1);let i=r,c=s,f=o,_=e,w=l;for(let S=0;S<80;S++){let U;S<20?U=this.leftRotate(i,5)+(c&f|~c&_)+w+u[S]+1518500249&4294967295:S<40?U=this.leftRotate(i,5)+(c^f^_)+w+u[S]+1859775393&4294967295:S<60?U=this.leftRotate(i,5)+(c&f|c&_|f&_)+w+u[S]+2400959708&4294967295:U=this.leftRotate(i,5)+(c^f^_)+w+u[S]+3395469782&4294967295,w=_,_=f,f=this.leftRotate(c,30),c=i,i=U}const y=this.add32(r,i),C=this.add32(s,c),P=this.add32(o,f),x=this.add32(e,_),O=this.add32(l,w);return{h0:y,h1:C,h2:P,h3:x,h4:O}}leftRotate(n,r){return(n<<r|n>>>32-r)>>>0}add32(n,r){return(n+r&4294967295)>>>0}static hashToHex(n,r,s,o,e){return n.toString(16).padStart(8,"0")+r.toString(16).padStart(8,"0")+s.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")+e.toString(16).padStart(8,"0")}}let te=null,ue=null;async function ht(){return te||ue||(ue=(async()=>{try{const t=await import("./wasm_pkg-DRWLiY4b.js");let n;if(typeof process<"u"&&!!process.versions?.node){const s=await import("./__vite-browser-external-9wXp6ZBx.js"),e=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");n={module_or_path:s.readFileSync(e)}}else n={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-D27IxIOn.wasm",import.meta.url)};return await t.default(n),te={IntegratedSeedSearcher:t.IntegratedSeedSearcher,BWGenerationConfig:t.BWGenerationConfig,PokemonGenerator:t.PokemonGenerator,SeedEnumerator:t.SeedEnumerator,EncounterType:t.EncounterType,GameVersion:t.GameVersion,GameMode:t.GameMode,calculate_game_offset:t.calculate_game_offset,sha1_hash_batch:t.sha1_hash_batch},te}catch(t){throw console.error("Failed to load WebAssembly module:",t),te=null,ue=null,t}})(),ue)}function yt(){if(!te)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return te}function Be(){return te!==null}const St={DS:8,DS_LITE:6,"3DS":9};class wt{sha1;useWasm=!1;constructor(){this.sha1=new De}async initializeWasm(){try{return await ht(),this.useWasm=!0,!0}catch(n){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",n),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&Be()}getWasmModule(){return yt()}setUseWasm(n){if(n&&!Be()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=n}getROMParameters(n,r){const s=Oe[n];if(!s)return console.error(`ROM version not found: ${n}`),null;const o=s[r];return o?{nazo:[...o.nazo],vcountTimerRanges:o.vcountTimerRanges.map(e=>[...e])}:(console.error(`ROM region not found: ${r} for version ${n}`),null)}toLittleEndian32(n){return((n&255)<<24|(n>>>8&255)<<16|(n>>>16&255)<<8|n>>>24&255)>>>0}toLittleEndian16(n){return(n&255)<<8|n>>>8&255}getDayOfWeek(n,r,s){return new Date(n,r-1,s).getDay()}generateMessage(n,r,s,o,e){const l=this.getROMParameters(n.romVersion,n.romRegion);if(!l)throw new Error(`No parameters found for ${n.romVersion} ${n.romRegion}`);const u=new Array(16).fill(0);for(let m=0;m<5;m++)u[m]=this.toLittleEndian32(l.nazo[m]);u[5]=this.toLittleEndian32(s<<16|r);const i=n.macAddress[4]<<8|n.macAddress[5];u[6]=i;const c=n.macAddress[0]<<0|n.macAddress[1]<<8|n.macAddress[2]<<16|n.macAddress[3]<<24,f=100663296,_=St[n.hardware],w=c^f^_;u[7]=this.toLittleEndian32(w);const y=o.getFullYear()%100,C=o.getMonth()+1,P=o.getDate(),x=this.getDayOfWeek(o.getFullYear(),C,P),O=Math.floor(y/10)*16+y%10,S=Math.floor(C/10)*16+C%10,U=Math.floor(P/10)*16+P%10,ne=Math.floor(x/10)*16+x%10;u[8]=O<<24|S<<16|U<<8|ne;const X=o.getHours(),N=o.getMinutes(),q=o.getSeconds(),j=(n.hardware==="DS"||n.hardware==="DS_LITE")&&X>=12?1:0,z=Math.floor(X/10)*16+X%10,H=Math.floor(N/10)*16+N%10,a=Math.floor(q/10)*16+q%10;u[9]=j<<30|z<<24|H<<16|a<<8|0,u[10]=0,u[11]=0;const p=e??Xe(n.keyInput);return u[12]=this.toLittleEndian32(p),u[13]=2147483648,u[14]=0,u[15]=416,u}calculateSeed(n){const r=this.sha1.calculateHash(n),s=BigInt(this.toLittleEndian32(r.h0)),e=BigInt(this.toLittleEndian32(r.h1))<<32n|s,i=e*0x5D588B656C078965n+0x269EC3n;return{seed:Number(i>>32n&0xFFFFFFFFn),hash:De.hashToHex(r.h0,r.h1,r.h2,r.h3,r.h4),lcgSeed:e}}parseTargetSeeds(n){const r=n.split(`
`).map(l=>l.trim()).filter(l=>l.length>0),s=[],o=[],e=new Set;return r.forEach((l,u)=>{try{let i=l.toLowerCase();if(i.startsWith("0x")&&(i=i.substring(2)),!/^[0-9a-f]{1,8}$/.test(i)){o.push({line:u+1,value:l,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const c=parseInt(i,16);if(e.has(c))return;e.add(c),s.push(c)}catch(i){const c=i instanceof Error?i.message:String(i);o.push({line:u+1,value:l,error:c||"Failed to parse as hexadecimal number."})}}),{validSeeds:s,errors:o}}getVCountForTimer0(n,r){for(const[s,o,e]of n.vcountTimerRanges)if(r>=o&&r<=e)return s;return n.vcountTimerRanges.length>0?n.vcountTimerRanges[0][0]:96}}const We=Uint32Array.BYTES_PER_ELEMENT,ce=2,W=ce*We,se=1,Ae=se*We,Ee=32,Rt={requiredFeatures:[]};function Ne(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function bt(t){if(!Ne())throw new Error("WebGPU is not available in this environment");const r=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!r)throw new Error("Failed to acquire WebGPU adapter");const s={requiredFeatures:Rt.requiredFeatures,requiredLimits:t?.requiredLimits,label:"seed-search-device"},o=await r.requestDevice(s);let e=!1;const l=o.lost.then(u=>(e=!0,console.warn("[webgpu] device lost:",u.message),u));return{getAdapter:()=>r,getDevice:()=>o,getQueue:()=>o.queue,getLimits:()=>o.limits,isLost:()=>e,waitForLoss:()=>l,getSupportedWorkgroupSize:u=>{const i=o.limits,c=x=>typeof x!="number"||!Number.isFinite(x)||x<=0?Number.POSITIVE_INFINITY:Math.floor(x),f=c(i.maxComputeInvocationsPerWorkgroup),_=c(i.maxComputeWorkgroupSizeX),w=Math.min(f,_),y=Number.isFinite(w)&&w>0?w:1,C=typeof u=="number"&&Number.isFinite(u)&&u>0?Math.floor(u):y,P=Math.min(C,y);if(P<=0)throw new Error("WebGPU workgroup size limits are invalid");return P}}}const Mt=1024*1024*1024,Et=1024*1024,vt=.25,Ct=.5,Pt=64,ve=t=>typeof t=="number"&&Number.isFinite(t)&&t>0,Tt=()=>{if(typeof navigator>"u")return null;const t=navigator;return ve(t.deviceMemory)?Math.floor(t.deviceMemory*Mt):null},xt=()=>{if(typeof performance>"u")return null;const t=performance;return ve(t.memory?.jsHeapSizeLimit)?Math.floor(t.memory.jsHeapSizeLimit):null},Dt=()=>typeof navigator>"u"||!ve(navigator.hardwareConcurrency)?4:Math.max(1,Math.floor(navigator.hardwareConcurrency)),He=()=>{const t=Tt();if(t!==null){const s=Math.floor(t*vt);return Math.max(W,s)}const n=xt();if(n!==null){const s=Math.floor(n*Ct);return Math.max(W,s)}const r=Dt()*Pt*Et;return Math.max(W,r)};var Bt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
\r
struct GeneratedConfig {\r
  message_count : u32,\r
  base_timer0_index : u32,\r
  base_vcount_index : u32,\r
  base_second_offset : u32,\r
  range_seconds : u32,\r
  timer0_min : u32,\r
  timer0_count : u32,\r
  vcount_min : u32,\r
  vcount_count : u32,\r
  start_second_of_day : u32,\r
  start_day_of_week : u32,\r
  mac_lower : u32,\r
  data7_swapped : u32,\r
  key_input_swapped : u32,\r
  hardware_type : u32,\r
  nazo0 : u32,\r
  nazo1 : u32,\r
  nazo2 : u32,\r
  nazo3 : u32,\r
  nazo4 : u32,\r
  start_year : u32,\r
  start_day_of_year : u32,\r
  groups_per_dispatch : u32,\r
  configured_workgroup_size : u32,\r
  candidate_capacity : u32,\r
  day_count : u32,\r
  hour_range_start : u32,\r
  hour_range_count : u32,\r
  minute_range_start : u32,\r
  minute_range_count : u32,\r
  second_range_start : u32,\r
  second_range_count : u32,\r
};\r
\r
struct TargetSeedBuffer {\r
  count : u32,\r
  values : array<u32>,\r
};\r
\r
struct MatchRecord {\r
  message_index : u32,\r
  seed : u32,\r
};\r
\r
struct MatchOutputBuffer {\r
  match_count : atomic<u32>,\r
  records : array<MatchRecord>,\r
};\r
\r
struct WideProduct {\r
  lo : u32,\r
  hi : u32,\r
};\r
\r
struct CarryResult {\r
  sum : u32,\r
  carry : u32,\r
};\r
\r
const MONTH_LENGTHS_COMMON : array<u32, 12> = array<u32, 12>(\r
  31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u\r
);\r
const MONTH_LENGTHS_LEAP : array<u32, 12> = array<u32, 12>(\r
  31u, 29u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u\r
);\r
\r
@group(0) @binding(0) var<storage, read> config : GeneratedConfig;\r
@group(0) @binding(1) var<storage, read> target_seeds : TargetSeedBuffer;\r
@group(0) @binding(2) var<storage, read_write> output_buffer : MatchOutputBuffer;\r
\r
fn left_rotate(value : u32, amount : u32) -> u32 {\r
  return (value << amount) | (value >> (32u - amount));\r
}\r
\r
fn swap32(value : u32) -> u32 {\r
  return ((value & 0x000000FFu) << 24u) |\r
    ((value & 0x0000FF00u) << 8u) |\r
    ((value & 0x00FF0000u) >> 8u) |\r
    ((value & 0xFF000000u) >> 24u);\r
}\r
\r
fn to_bcd(value : u32) -> u32 {\r
  let tens = value / 10u;\r
  let ones = value - tens * 10u;\r
  return (tens << 4u) | ones;\r
}\r
\r
fn is_leap_year(year : u32) -> bool {\r
  return (year % 4u == 0u && year % 100u != 0u) || (year % 400u == 0u);\r
}\r
\r
fn month_day_from_day_of_year(day_of_year : u32, leap : bool) -> vec2<u32> {\r
  var remaining = day_of_year;\r
  var month = 1u;\r
  for (var i = 0u; i < 12u; i = i + 1u) {\r
    let length = select(MONTH_LENGTHS_COMMON[i], MONTH_LENGTHS_LEAP[i], leap);\r
    if (remaining <= length) {\r
      return vec2<u32>(month, remaining);\r
    }\r
    remaining = remaining - length;\r
    month = month + 1u;\r
  }\r
  return vec2<u32>(12u, 31u);\r
}\r
\r
fn mulExtended(a : u32, b : u32) -> WideProduct {\r
  let a_lo = a & 0xFFFFu;\r
  let a_hi = a >> 16u;\r
  let b_lo = b & 0xFFFFu;\r
  let b_hi = b >> 16u;\r
\r
  let low = a_lo * b_lo;\r
  let mid1 = a_lo * b_hi;\r
  let mid2 = a_hi * b_lo;\r
  let high = a_hi * b_hi;\r
\r
  let carry_mid = (low >> 16u) + (mid1 & 0xFFFFu) + (mid2 & 0xFFFFu);\r
  let lo = (low & 0xFFFFu) | ((carry_mid & 0xFFFFu) << 16u);\r
  let hi = high + (mid1 >> 16u) + (mid2 >> 16u) + (carry_mid >> 16u);\r
\r
  return WideProduct(lo, hi);\r
}\r
\r
fn addCarry(a : u32, b : u32) -> CarryResult {\r
  let sum = a + b;\r
  let carry = select(0u, 1u, sum < a);\r
  return CarryResult(sum, carry);\r
}\r
\r
fn compute_seed_from_hash(h0 : u32, h1 : u32) -> u32 {\r
  let le0 = swap32(h0);\r
  let le1 = swap32(h1);\r
\r
  let mul_lo : u32 = 0x6C078965u;\r
  let mul_hi : u32 = 0x5D588B65u;\r
  let increment : u32 = 0x00269EC3u;\r
\r
  let prod0 = mulExtended(le0, mul_lo);\r
  let prod1 = mulExtended(le0, mul_hi);\r
  let prod2 = mulExtended(le1, mul_lo);\r
  let inc = addCarry(prod0.lo, increment);\r
\r
  // Upper 32-bit word of ((le1<<32 | le0) * multiplier + increment)\r
  var upper_word = prod0.hi;\r
  upper_word = upper_word + prod1.lo;\r
  upper_word = upper_word + prod2.lo;\r
  upper_word = upper_word + inc.carry;\r
\r
  return upper_word;\r
}\r
\r
@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)\r
fn sha1_generate(\r
  @builtin(global_invocation_id) global_id : vec3<u32>\r
) {\r
\r
  let global_linear_index = global_id.x;\r
  let is_active = global_linear_index < config.message_count;\r
  var seed : u32 = 0u;\r
  var matched = false;\r
\r
  if (is_active) {\r
    let safe_range_seconds = max(config.range_seconds, 1u);\r
    let safe_vcount_count = max(config.vcount_count, 1u);\r
    let messages_per_vcount = safe_range_seconds;\r
    let messages_per_timer0 = messages_per_vcount * safe_vcount_count;\r
\r
    let local_timer0_index = global_linear_index / messages_per_timer0;\r
    let local_remainder_after_timer0 = global_linear_index - local_timer0_index * messages_per_timer0;\r
    let local_vcount_index = local_remainder_after_timer0 / messages_per_vcount;\r
    let local_second_offset = local_remainder_after_timer0 - local_vcount_index * messages_per_vcount;\r
\r
    let combined_second_offset = config.base_second_offset + local_second_offset;\r
    let carry_to_vcount = combined_second_offset / messages_per_vcount;\r
    let second_offset = combined_second_offset - carry_to_vcount * messages_per_vcount;\r
\r
    let combined_vcount_index = config.base_vcount_index + local_vcount_index + carry_to_vcount;\r
    let carry_to_timer0 = combined_vcount_index / safe_vcount_count;\r
    let vcount_index = combined_vcount_index - carry_to_timer0 * safe_vcount_count;\r
\r
    let timer0_index = config.base_timer0_index + local_timer0_index + carry_to_timer0;\r
\r
    let timer0 = config.timer0_min + timer0_index;\r
    let vcount = config.vcount_min + vcount_index;\r
    let safe_hour_count = max(config.hour_range_count, 1u);\r
    let safe_minute_count = max(config.minute_range_count, 1u);\r
    let safe_second_count = max(config.second_range_count, 1u);\r
    let combos_per_day = safe_hour_count * safe_minute_count * safe_second_count;\r
\r
    let day_offset = second_offset / combos_per_day;\r
    let remainder_after_day = second_offset - day_offset * combos_per_day;\r
\r
    let entries_per_hour = safe_minute_count * safe_second_count;\r
    let hour_index = remainder_after_day / entries_per_hour;\r
    let remainder_after_hour = remainder_after_day - hour_index * entries_per_hour;\r
    let minute_index = remainder_after_hour / safe_second_count;\r
    let second_index = remainder_after_hour - minute_index * safe_second_count;\r
\r
    let hour = config.hour_range_start + hour_index;\r
    let minute = config.minute_range_start + minute_index;\r
    let second = config.second_range_start + second_index;\r
    let seconds_of_day = hour * 3600u + minute * 60u + second;\r
\r
    var year = config.start_year;\r
    var day_of_year = config.start_day_of_year + day_offset;\r
    loop {\r
      let year_length = select(365u, 366u, is_leap_year(year));\r
      if (day_of_year <= year_length) {\r
        break;\r
      }\r
      day_of_year = day_of_year - year_length;\r
      year = year + 1u;\r
    }\r
\r
    let leap = is_leap_year(year);\r
    let month_day = month_day_from_day_of_year(day_of_year, leap);\r
    let month = month_day.x;\r
    let day = month_day.y;\r
\r
    let day_of_week = (config.start_day_of_week + day_offset) % 7u;\r
    let year_mod = year % 100u;\r
    let date_word = (to_bcd(year_mod) << 24u) | (to_bcd(month) << 16u) | (to_bcd(day) << 8u) | to_bcd(day_of_week);\r
    let is_pm = (config.hardware_type <= 1u) && (hour >= 12u);\r
    let pm_flag = select(0u, 1u, is_pm);\r
    let time_word = (pm_flag << 30u) | (to_bcd(hour) << 24u) | (to_bcd(minute) << 16u) | (to_bcd(second) << 8u);\r
\r
    var w : array<u32, 16>;\r
    w[0] = config.nazo0;\r
    w[1] = config.nazo1;\r
    w[2] = config.nazo2;\r
    w[3] = config.nazo3;\r
    w[4] = config.nazo4;\r
    w[5] = swap32((vcount << 16u) | timer0);\r
    w[6] = config.mac_lower;\r
    w[7] = config.data7_swapped;\r
    w[8] = date_word;\r
    w[9] = time_word;\r
    w[10] = 0u;\r
    w[11] = 0u;\r
    w[12] = config.key_input_swapped;\r
    w[13] = 0x80000000u;\r
    w[14] = 0u;\r
    w[15] = 0x000001A0u;\r
\r
    var a : u32 = 0x67452301u;\r
    var b : u32 = 0xEFCDAB89u;\r
    var c : u32 = 0x98BADCFEu;\r
    var d : u32 = 0x10325476u;\r
    var e : u32 = 0xC3D2E1F0u;\r
\r
    var i : u32 = 0u;\r
    for (; i < 20u; i = i + 1u) {\r
      let w_index = i & 15u;\r
      var w_value : u32;\r
      if (i < 16u) {\r
        w_value = w[w_index];\r
      } else {\r
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];\r
        let rotated = left_rotate(expanded, 1u);\r
        w[w_index] = rotated;\r
        w_value = rotated;\r
      }\r
\r
      let temp = left_rotate(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5A827999u + w_value;\r
      e = d;\r
      d = c;\r
      c = left_rotate(b, 30u);\r
      b = a;\r
      a = temp;\r
    }\r
\r
    for (; i < 40u; i = i + 1u) {\r
      let w_index = i & 15u;\r
      var w_value : u32;\r
      if (i < 16u) {\r
        w_value = w[w_index];\r
      } else {\r
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];\r
        let rotated = left_rotate(expanded, 1u);\r
        w[w_index] = rotated;\r
        w_value = rotated;\r
      }\r
\r
      let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0x6ED9EBA1u + w_value;\r
      e = d;\r
      d = c;\r
      c = left_rotate(b, 30u);\r
      b = a;\r
      a = temp;\r
    }\r
\r
    for (; i < 60u; i = i + 1u) {\r
      let w_index = i & 15u;\r
      var w_value : u32;\r
      if (i < 16u) {\r
        w_value = w[w_index];\r
      } else {\r
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];\r
        let rotated = left_rotate(expanded, 1u);\r
        w[w_index] = rotated;\r
        w_value = rotated;\r
      }\r
\r
      let temp = left_rotate(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + w_value;\r
      e = d;\r
      d = c;\r
      c = left_rotate(b, 30u);\r
      b = a;\r
      a = temp;\r
    }\r
\r
    for (; i < 80u; i = i + 1u) {\r
      let w_index = i & 15u;\r
      var w_value : u32;\r
      if (i < 16u) {\r
        w_value = w[w_index];\r
      } else {\r
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];\r
        let rotated = left_rotate(expanded, 1u);\r
        w[w_index] = rotated;\r
        w_value = rotated;\r
      }\r
\r
      let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0xCA62C1D6u + w_value;\r
      e = d;\r
      d = c;\r
      c = left_rotate(b, 30u);\r
      b = a;\r
      a = temp;\r
    }\r
\r
    let h0 = 0x67452301u + a;\r
    let h1 = 0xEFCDAB89u + b;\r
    let h2 = 0x98BADCFEu + c;\r
    let h3 = 0x10325476u + d;\r
    let h4 = 0xC3D2E1F0u + e;\r
\r
    seed = compute_seed_from_hash(h0, h1);\r
\r
    let target_count = target_seeds.count;\r
    matched = target_count == 0u;\r
    for (var j = 0u; j < target_count; j = j + 1u) {\r
      if (target_seeds.values[j] == seed) {\r
        matched = true;\r
        break;\r
      }\r
    }\r
  }\r
\r
  if (!matched) {\r
    return;\r
  }\r
\r
  let record_index = atomicAdd(&output_buffer.match_count, 1u);\r
  if (record_index >= config.candidate_capacity) {\r
    atomicSub(&output_buffer.match_count, 1u);\r
    return;\r
  }\r
\r
  output_buffer.records[record_index].message_index = global_linear_index;\r
  output_buffer.records[record_index].seed = seed;\r
}\r
`;function At(t){return Bt.replace(/WORKGROUP_SIZE_PLACEHOLDER/g,String(t))}function Ot(t,n){const r=t.createShaderModule({label:"gpu-seed-sha1-generated-module",code:At(n)});r.getCompilationInfo?.().then(e=>{e.messages.length>0&&console.warn("[pipeline-factory] compilation diagnostics",e.messages.map(l=>({message:l.message,line:l.lineNum,column:l.linePos,type:l.type})))}).catch(e=>{console.warn("[pipeline-factory] compilation info failed",e)});const s=t.createBindGroupLayout({label:"gpu-seed-generate-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});return{pipeline:t.createComputePipeline({label:"gpu-seed-generate-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-generate-pipeline-layout",bindGroupLayouts:[s]}),compute:{module:r,entryPoint:"sha1_generate"}}),layout:s}}function Ut(t,n){const r=n?.slots??Ee;if(r<=0)throw new Error("buffer pool must have at least one slot");const s=Array.from({length:r},()=>({output:null,readback:null,matchCount:null,outputSize:0,readbackSize:0,matchCountSize:0})),o=u=>Math.ceil(u/256)*256;return{get slotCount(){return s.length},acquire:(u,i)=>{if(u<0||u>=s.length)throw new Error(`buffer slot ${u} is out of range`);if(!Number.isFinite(i)||i<=0)throw new Error("messageCount must be a positive integer");const c=s[u],f=i,_=o(Ae+f*W),w=f,y=o(Ae);return(!c.output||_>c.outputSize)&&(c.output?.destroy(),c.output=t.createBuffer({label:`gpu-seed-output-${u}`,size:_,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),c.outputSize=_),(!c.readback||_>c.readbackSize)&&(c.readback?.destroy(),c.readback=t.createBuffer({label:`gpu-seed-readback-${u}`,size:_,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),c.readbackSize=_),(!c.matchCount||y>c.matchCountSize)&&(c.matchCount?.destroy(),c.matchCount=t.createBuffer({label:`gpu-seed-match-header-${u}`,size:y,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),c.matchCountSize=y),{output:c.output,readback:c.readback,matchCount:c.matchCount,outputSize:c.outputSize,candidateCapacity:w,maxRecords:f}},dispose:()=>{for(const u of s)u.output?.destroy(),u.readback?.destroy(),u.matchCount?.destroy(),u.output=null,u.readback=null,u.matchCount=null,u.outputSize=0,u.readbackSize=0,u.matchCountSize=0}}}function zt(t,n){const r=typeof n?.hostMemoryLimitBytes=="number"&&Number.isFinite(n.hostMemoryLimitBytes)&&n.hostMemoryLimitBytes>0?Math.max(W,Math.floor(n.hostMemoryLimitBytes)):He(),s=n?.bufferSetCount??Ee,o=(()=>{const i=n?.hostMemoryLimitPerSlot;if(typeof i=="number"&&Number.isFinite(i)&&i>0)return Math.max(W,Math.floor(i));const c=Math.floor(r/s);return Math.max(W,c)})(),e=typeof n?.workgroupSize=="number"&&Number.isFinite(n.workgroupSize)&&n.workgroupSize>0?Math.floor(n.workgroupSize):t.getSupportedWorkgroupSize();if(r<=0)throw new Error("host memory limit must be positive");if(s<=0)throw new Error("buffer set count must be positive");const l=i=>{const f=t.getDevice().limits,_=Math.max(1,f.maxStorageBufferBindingSize??W),w=Math.max(1,Math.floor(_/W)),y=Math.max(1,Math.floor(o/W)),C=t.getSupportedWorkgroupSize(e),P=f.maxComputeWorkgroupsPerDimension??65535,x=Math.max(1,C*P),O=Math.min(w,y,x);return i<=O?i<=1?1:Math.max(1,Math.min(O,Math.ceil(i/2))):O};return{computePlan:i=>{if(!Number.isFinite(i)||i<0)throw new Error("totalMessages must be a non-negative finite value");if(i===0)return{maxMessagesPerDispatch:0,dispatches:[]};const c=l(i),f=[];let _=i,w=0;for(;_>0;){const y=Math.min(c,_);f.push({baseOffset:w,messageCount:y}),w+=y,_-=y}if(f.length===1&&i>1){const y=f[0],C=Math.ceil(y.messageCount/2),P=y.messageCount-C;P>0&&(f[0]={baseOffset:y.baseOffset,messageCount:C},f.push({baseOffset:y.baseOffset+C,messageCount:P}))}return{maxMessagesPerDispatch:c,dispatches:f}}}}const Lt=32,kt=500,Ft=1024,Re=new Uint32Array([0]),It=Ne;function Gt(t){const n=a=>Ee,r=He(),s=n(),o=(()=>{const a=Math.floor(r/s);return Math.max(W,a)})(),e={workgroupSize:0,bufferSlotCount:s,hostMemoryLimitBytes:r,hostMemoryLimitPerSlotBytes:o,deviceContext:null,pipeline:null,bindGroupLayout:null,configBuffer:null,configData:null,bufferPool:null,planner:null,targetBuffer:null,targetBufferCapacity:0,seedCalculator:new wt,isRunning:!1,isPaused:!1,shouldStop:!1,lastProgressUpdateMs:0,timerState:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1}},l=async(a,p,m)=>(async()=>await Promise.resolve(m()))(),u=async()=>{if(e.pipeline&&e.bufferPool&&e.planner&&e.deviceContext)return;const a=await bt(),p=a.getDevice(),m=a.getSupportedWorkgroupSize(e.workgroupSize),{pipeline:R,layout:b}=Ot(p,m),M=new Uint32Array(Lt),T=H(M.byteLength),h=p.createBuffer({label:"gpu-seed-config-buffer",size:T,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),v=Ut(p,{slots:e.bufferSlotCount}),E=zt(a,{workgroupSize:m,bufferSetCount:e.bufferSlotCount,hostMemoryLimitBytes:e.hostMemoryLimitBytes,hostMemoryLimitPerSlot:e.hostMemoryLimitPerSlotBytes});e.deviceContext=a,e.pipeline=R,e.bindGroupLayout=b,e.configBuffer=h,e.configData=M,e.bufferPool=v,e.planner=E,e.workgroupSize=m},i=a=>{if(!e.deviceContext)throw new Error("WebGPU device is not initialized");const p=e.deviceContext.getDevice(),m=a.length,R=1+m,b=H(R*Uint32Array.BYTES_PER_ELEMENT),M=e.targetBufferCapacity;if(!e.targetBuffer||M<m){e.targetBuffer?.destroy(),e.targetBuffer=p.createBuffer({label:"gpu-seed-target-buffer",size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const E=Math.floor(b/Uint32Array.BYTES_PER_ELEMENT)-1;e.targetBufferCapacity=Math.max(E,m)}const h=new Uint32Array(R);h[0]=m>>>0;for(let E=0;E<m;E+=1)h[1+E]=a[E]>>>0;const v=h.byteLength;p.queue.writeBuffer(e.targetBuffer,0,h.buffer,h.byteOffset,v)},c=async a=>{if(e.isRunning)throw new Error("WebGPU search is already running");if((!e.pipeline||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)&&await u(),!e.pipeline||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)throw new Error("WebGPU runner failed to initialize");const{context:p,targetSeeds:m,callbacks:R,signal:b}=a;if(p.totalMessages===0){R.onComplete("探索対象の組み合わせが存在しません");return}if(!e.bindGroupLayout)throw new Error("WebGPU runner missing bind group layout");i(m),e.isRunning=!0,e.isPaused=!1,e.shouldStop=!1,e.lastProgressUpdateMs=Date.now();const M={currentStep:0,totalSteps:p.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0};let T;if(b)if(b.aborted)e.shouldStop=!0;else{const h=()=>{e.shouldStop=!0};b.addEventListener("abort",h),T=()=>b.removeEventListener("abort",h)}ne();try{await C(p,M,R);const h=q(),v={...M,elapsedTime:h,estimatedTimeRemaining:0};e.shouldStop?R.onStopped("検索を停止しました",v):(R.onProgress(v),R.onComplete(`検索が完了しました。${M.matchesFound}件ヒットしました。`))}catch(h){const v=h instanceof Error?h.message:"WebGPU検索中に不明なエラーが発生しました",E=h instanceof GPUValidationError?"WEBGPU_VALIDATION_ERROR":void 0;throw R.onError(v,E),h}finally{e.isRunning=!1,e.isPaused=!1,e.shouldStop=!1,X(),T&&T()}},f=()=>{!e.isRunning||e.isPaused||(e.isPaused=!0,X())},_=()=>{!e.isRunning||!e.isPaused||(e.isPaused=!1,N())},w=()=>{e.isRunning&&(e.shouldStop=!0,e.isPaused=!1,N())},y=()=>{e.bufferPool?.dispose(),e.configBuffer?.destroy(),e.configBuffer=null,e.configData=null,e.pipeline=null,e.bindGroupLayout=null,e.bufferPool=null,e.planner=null,e.deviceContext=null,e.targetBuffer?.destroy(),e.targetBuffer=null,e.targetBufferCapacity=0},C=async(a,p,m)=>{if(!e.deviceContext||!e.pipeline||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.targetBuffer||!e.bindGroupLayout)throw new Error("WebGPU runner is not ready");const b=e.deviceContext.getDevice().queue,M=e.bufferPool.slotCount,T=Array.from({length:M},(d,A)=>M-1-A),h=[],v=[],E=[],Y=d=>{E.push(d)},F=()=>new Promise(d=>{if(T.length>0){const A=T.pop();d(A);return}h.push(d)}),G=d=>{const A=h.shift();if(A){A(d);return}T.push(d)};let g=0;for(const d of a.segments){if(e.shouldStop)break;const A=await l("planner.computePlan",{segmentIndex:d.index,totalMessages:d.totalMessages},()=>Promise.resolve(e.planner.computePlan(d.totalMessages)));for(const I of A.dispatches){if(e.shouldStop||(await j(),e.shouldStop))break;const L=await F();if(e.shouldStop){G(L);break}const Z={segment:d,dispatchIndex:g,messageCount:I.messageCount,slotIndex:L},V=P(Z,I.baseOffset,a,p,m,b,G,Y);v.push(V),g+=1}}v.length>0&&await Promise.all(v),E.length>0&&await Promise.all(E)},P=async(a,p,m,R,b,M,T,h)=>{if(!e.deviceContext||!e.pipeline||!e.bufferPool||!e.configBuffer||!e.configData||!e.targetBuffer||!e.bindGroupLayout)throw new Error("WebGPU runner is not ready");const v=e.deviceContext.getDevice(),E=e.configBuffer,Y=e.configData,F=e.bindGroupLayout,G=e.pipeline,g=e.targetBuffer,d=e.bufferPool.acquire(a.slotIndex,a.messageCount);let A=!1,I=!1;const L=()=>{A||(A=!0,T(a.slotIndex))},Z=Math.ceil(a.messageCount/e.workgroupSize),V=Math.max(1,Z),$=H(se*Uint32Array.BYTES_PER_ELEMENT),k={dispatchIndex:a.dispatchIndex,messageCount:a.messageCount,slotIndex:a.slotIndex,workgroupCount:Z,candidateCapacity:d.candidateCapacity,segmentIndex:a.segment.index,segmentBaseOffset:p};try{await l("dispatch",k,async()=>{M.writeBuffer(d.output,0,Re.buffer,Re.byteOffset,Re.byteLength),U(a.segment,p,a.messageCount,V,d.candidateCapacity),M.writeBuffer(E,0,Y.buffer,Y.byteOffset,Y.byteLength);const re=v.createBindGroup({label:`gpu-seed-generate-group-${a.dispatchIndex}`,layout:F,entries:[{binding:0,resource:{buffer:E}},{binding:1,resource:{buffer:g}},{binding:2,resource:{buffer:d.output}}]}),ae=v.createCommandEncoder({label:`gpu-seed-compute-${a.dispatchIndex}`}),oe=ae.beginComputePass({label:`gpu-seed-generate-pass-${a.dispatchIndex}`});oe.setPipeline(G),oe.setBindGroup(0,re),oe.dispatchWorkgroups(Z),oe.end(),ae.copyBufferToBuffer(d.output,0,d.matchCount,0,$);const le=ae.finish();await l("dispatch.submit",{...k},async()=>{await l("dispatch.submit.encode",{...k},async()=>{M.submit([le])})});const de=await l("dispatch.mapMatchCount",{...k,headerCopyBytes:$},async()=>{await d.matchCount.mapAsync(GPUMapMode.READ,0,$);const Q=new Uint32Array(d.matchCount.getMappedRange(0,$))[0]??0;return d.matchCount.unmap(),Q}),fe=Math.min(de,d.maxRecords)*ce*Uint32Array.BYTES_PER_ELEMENT,J=H(se*Uint32Array.BYTES_PER_ELEMENT+fe);await l("dispatch.copyResults",{...k,totalCopyBytes:J},async()=>{const K=v.createCommandEncoder({label:`gpu-seed-copy-${a.dispatchIndex}`});K.copyBufferToBuffer(d.output,0,d.readback,0,J);const Q=K.finish();await l("dispatch.copyResults.encode",{...k,totalCopyBytes:J},async()=>{M.submit([Q])})});const he=(async()=>{try{const{results:K,clampedMatchCount:Q}=await l("dispatch.mapResults",{...k,totalCopyBytes:J},async()=>{await d.readback.mapAsync(GPUMapMode.READ,0,J);const ye=d.readback.getMappedRange(0,J),Se=new Uint32Array(ye),Ve=Se[0]??0,$e=Math.max(0,Math.floor((Se.length-se)/ce)),Ce=Math.min(Ve,d.maxRecords,$e),Pe=se+Ce*ce,Te=new Uint32Array(Pe);return Te.set(Se.subarray(0,Pe)),d.readback.unmap(),{results:Te,clampedMatchCount:Ce}});try{L(),await l("dispatch.processMatches",{...k,matchCount:Q},()=>x(K,Q,a,p,m,R,b))}finally{L()}}catch(K){throw L(),K}})();I=!0,h(he)})}finally{I||L()}},x=async(a,p,m,R,b,M,T)=>{const h=m.segment,v=h.rangeSeconds,E=Math.max(v,1),Y=Math.max(h.config.vcountCount,1),F=E,G=F*Y,g=h.config.timer0Min,d=h.config.vcountMin,A=R;for(let I=0;I<p&&!(e.shouldStop||I%Ft===0&&(await j(),e.shouldStop));I+=1){const L=se+I*ce,Z=a[L],V=A+Z,$=a[L+1]>>>0,k=Math.floor(V/G),re=V-k*G,ae=Math.floor(re/F),oe=re-ae*F,le=g+k,de=d+ae,pe=xe(b.timePlan,oe),fe=m.segment.keyCode,J=e.seedCalculator.generateMessage(b.conditions,le,de,pe,fe),{hash:he,seed:K,lcgSeed:Q}=e.seedCalculator.calculateSeed(J);K!==$&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:$,cpuSeed:K,messageIndex:V});const ye={seed:$,datetime:pe,timer0:le,vcount:de,keyCode:fe,conditions:b.conditions,message:J,sha1Hash:he,lcgSeed:Q,isMatch:!0};T.onResult(ye),M.matchesFound+=1}if(m.messageCount>0){const I=m.messageCount-1,L=A+I,Z=Math.floor(L/G),V=L-Z*G,$=Math.floor(V/F),k=V-$*F,re=xe(b.timePlan,k).toISOString();M.currentDateTime=re}M.currentStep+=m.messageCount,O(M,T)},O=(a,p)=>{const m=Date.now();if(m-e.lastProgressUpdateMs<kt&&a.currentStep<a.totalSteps)return;const R=q(),b=S(a.currentStep,a.totalSteps,R);p.onProgress({currentStep:a.currentStep,totalSteps:a.totalSteps,elapsedTime:R,estimatedTimeRemaining:b,matchesFound:a.matchesFound,currentDateTime:a.currentDateTime}),e.lastProgressUpdateMs=m},S=(a,p,m)=>{if(a===0||a>=p)return 0;const R=m/a,b=p-a;return Math.round(R*b)},U=(a,p,m,R,b)=>{if(!e.configData)throw new Error("config buffer not prepared");const M=Math.max(a.config.rangeSeconds,1),T=Math.max(a.config.vcountCount,1),h=M,v=h*T,E=Math.floor(p/v),Y=p-E*v,F=Math.floor(Y/h),G=Y-F*h,g=e.configData;g[0]=m>>>0,g[1]=E>>>0,g[2]=F>>>0,g[3]=G>>>0,g[4]=a.config.rangeSeconds>>>0,g[5]=a.config.timer0Min>>>0,g[6]=a.config.timer0Count>>>0,g[7]=a.config.vcountMin>>>0,g[8]=a.config.vcountCount>>>0,g[9]=a.config.startSecondOfDay>>>0,g[10]=a.config.startDayOfWeek>>>0,g[11]=a.config.macLower>>>0,g[12]=a.config.data7Swapped>>>0,g[13]=a.config.keyInputSwapped>>>0,g[14]=a.config.hardwareType>>>0;for(let d=0;d<a.config.nazoSwapped.length;d+=1)g[15+d]=a.config.nazoSwapped[d]>>>0;g[20]=a.config.startYear>>>0,g[21]=a.config.startDayOfYear>>>0,g[22]=R>>>0,g[23]=e.workgroupSize>>>0,g[24]=b>>>0,g[25]=a.config.dayCount>>>0,g[26]=a.config.hourRangeStart>>>0,g[27]=a.config.hourRangeCount>>>0,g[28]=a.config.minuteRangeStart>>>0,g[29]=a.config.minuteRangeCount>>>0,g[30]=a.config.secondRangeStart>>>0,g[31]=a.config.secondRangeCount>>>0},ne=()=>{e.timerState.cumulativeRunTime=0,e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1},X=()=>{e.timerState.isPaused||(e.timerState.cumulativeRunTime+=Date.now()-e.timerState.segmentStartTime,e.timerState.isPaused=!0)},N=()=>{e.timerState.isPaused&&(e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1)},q=()=>e.timerState.isPaused?e.timerState.cumulativeRunTime:e.timerState.cumulativeRunTime+(Date.now()-e.timerState.segmentStartTime),j=async()=>{for(;e.isPaused&&!e.shouldStop;)await z(25)},z=a=>new Promise(p=>setTimeout(p,a)),H=a=>Math.ceil(a/256)*256;return{init:u,run:c,pause:f,resume:_,stop:w,dispose:y}}const Ye=self,B={isRunning:!1,isPaused:!1},_e=Gt();let ge=null;function D(t){Ye.postMessage(t)}function Wt(){D({type:"READY",message:"WebGPU worker initialized"})}function ie(){B.isRunning=!1,B.isPaused=!1,ge=null}function Nt(){return It()?!0:(D({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Ht(t){if(B.isRunning){D({type:"ERROR",error:"Search is already running"});return}if(!t.conditions||!t.targetSeeds){D({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!Nt())return;B.isRunning=!0,B.isPaused=!1;let n;try{n=ot(t.conditions)}catch(o){ie();const e=o instanceof Error?o.message:"検索条件の解析中にエラーが発生しました";D({type:"ERROR",error:e,errorCode:"WEBGPU_CONTEXT_ERROR"});return}ge=new AbortController;const r={onProgress:o=>{D({type:"PROGRESS",progress:o})},onResult:o=>{D({type:"RESULT",result:o})},onComplete:o=>{ie(),D({type:"COMPLETE",message:o})},onError:(o,e)=>{ie(),D({type:"ERROR",error:o,errorCode:e})},onPaused:()=>{B.isPaused=!0,D({type:"PAUSED"})},onResumed:()=>{B.isPaused=!1,D({type:"RESUMED"})},onStopped:(o,e)=>{ie(),D({type:"STOPPED",message:o,progress:e})}},s={context:n,targetSeeds:t.targetSeeds,callbacks:r,signal:ge.signal};try{await _e.run(s)}catch(o){if(!B.isRunning)return;ie();const e=o instanceof Error?o.message:"WebGPU search failed with unknown error";D({type:"ERROR",error:e,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function Yt(){!B.isRunning||B.isPaused||(_e.pause(),B.isPaused=!0,D({type:"PAUSED"}))}function Vt(){!B.isRunning||!B.isPaused||(_e.resume(),B.isPaused=!1,D({type:"RESUMED"}))}function $t(){B.isRunning&&(_e.stop(),ge?.abort())}Wt();Ye.onmessage=t=>{const n=t.data;switch(n.type){case"START_SEARCH":Ht(n);break;case"PAUSE_SEARCH":Yt();break;case"RESUME_SEARCH":Vt();break;case"STOP_SEARCH":$t();break;default:D({type:"ERROR",error:`Unknown request type: ${n.type}`})}};
