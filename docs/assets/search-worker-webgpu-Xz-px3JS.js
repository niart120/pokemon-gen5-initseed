const ze={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},ke=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],J=ke.reduce((t,[n,r])=>(t[n]=r,t),{}),Xe=ke.length,Je=(1<<Xe)-1,Ge=12287,Qe=[1<<J["[↑]"]|1<<J["[↓]"],1<<J["[←]"]|1<<J["[→]"],1<<J.Select|1<<J.Start|1<<J.L|1<<J.R];function Le(t,n){return Number.isFinite(t)?t&Je:0}function et(t){const n=Le(t);return Ge^n}function tt(t){const n=Le(t);for(const r of Qe)if((n&r)===r)return!0;return!1}function nt(t){return et(t)}const me=1e3,Ce=60,We=60,Ie=24,Fe=Ce*We,rt=Fe*Ie,Ne=rt*me;function at(t){const n=t.timeRange;if(!n)throw new Error("timeRange is required for WebGPU search");const r=ve("hour",n.hour,0,Ie-1),i=ve("minute",n.minute,0,We-1),s=ve("second",n.second,0,Ce-1),e=new Date(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,0,0,0),l=new Date(t.dateRange.endYear,t.dateRange.endMonth-1,t.dateRange.endDay,0,0,0),c=e.getTime(),a=l.getTime();if(c>a)throw new Error("開始日が終了日より後に設定されています");const d=Math.floor((a-c)/Ne)+1;if(d<=0)throw new Error("探索日数が検出できませんでした");const u=r.count*i.count*s.count;if(u<=0)throw new Error("時刻レンジの組み合わせ数が0です");const p=new Date(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,r.start,i.start,s.start,0);return{plan:{dayCount:d,combosPerDay:u,hourRangeStart:r.start,hourRangeCount:r.count,minuteRangeStart:i.start,minuteRangeCount:i.count,secondRangeStart:s.start,secondRangeCount:s.count,startDayTimestampMs:c},firstCombinationDate:p}}function Oe(t,n){const r=Math.max(t.minuteRangeCount,1),i=Math.max(t.secondRangeCount,1),s=Math.max(t.combosPerDay,1),e=Math.max(0,Math.trunc(n)),l=Math.floor(e/s),c=e-l*s,a=r*i,d=Math.floor(c/a),u=c-d*a,p=Math.floor(u/i),b=u-p*i,S=t.hourRangeStart+d,P=t.minuteRangeStart+p,M=t.secondRangeStart+b,T=t.startDayTimestampMs+l*Ne+S*Fe*me+P*Ce*me+M*me;return new Date(T)}function ve(t,n,r,i){if(!n)throw new Error(`${t} range is required for WebGPU search`);const s=Math.trunc(n.start),e=Math.trunc(n.end);if(Number.isNaN(s)||Number.isNaN(e))throw new Error(`${t} range must be numeric`);if(s<r||e>i)throw new Error(`${t} range must be within ${r} to ${i}`);if(s>e)throw new Error(`${t} range start must be <= end`);return{start:s,end:e,count:e-s+1}}const ot=Date.UTC(2e3,0,1,0,0,0),st=100663296,ut={DS:8,DS_LITE:6,"3DS":9};function it(t){const n=[];for(let e=0;e<12;e++)(t&1<<e)!==0&&n.push(e);const r=n.length,i=1<<r,s=[];for(let e=0;e<i;e++){let l=0;for(let a=0;a<r;a++)(e&1<<a)!==0&&(l|=1<<n[a]);if(tt(l))continue;const c=l^Ge;s.push(c)}return s}function ct(t){const{plan:n,firstCombinationDate:r}=at(t),i=r,s=lt(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond);if(i.getTime()>s.getTime())throw new Error("開始日時が終了日時より後ろに設定されています");const e=ft(t),l=gt(t,e),c=dt(t.dateRange.startYear,t.dateRange.startMonth-1,t.dateRange.startDay,n.hourRangeStart,n.minuteRangeStart,n.secondRangeStart),a=Math.floor((c-ot)/1e3);if(a<0)throw new Error("2000年より前の日時は指定できません");const d=n.dayCount*n.combosPerDay;if(d<=0)throw new Error("探索秒数が0秒以下です");const u=i.getFullYear(),p=bt(i),b=St(i),S=i.getDay(),P=ut[t.hardware],{macLower:M,data7Swapped:x}=_t(t.macAddress,P),T=it(t.keyInput),y=yt(e.nazo),z=[];let ae=0;for(const N of T){const V=Pe(N>>>0);for(let K=0;K<l.length;K+=1){const k=l[K],H=k.timer0Max-k.timer0Min+1,o=d*H,_={startSecondsSince2000:a>>>0,rangeSeconds:d>>>0,timer0Min:k.timer0Min>>>0,timer0Max:k.timer0Max>>>0,timer0Count:H>>>0,vcountMin:k.vcount>>>0,vcountMax:k.vcount>>>0,vcountCount:1,totalMessages:o>>>0,hardwareType:ht(t.hardware),macLower:M>>>0,data7Swapped:x>>>0,keyInputSwapped:V>>>0,nazoSwapped:y,startYear:u>>>0,startDayOfYear:p>>>0,startSecondOfDay:b>>>0,startDayOfWeek:S>>>0,dayCount:n.dayCount>>>0,hourRangeStart:n.hourRangeStart>>>0,hourRangeCount:n.hourRangeCount>>>0,minuteRangeStart:n.minuteRangeStart>>>0,minuteRangeCount:n.minuteRangeCount>>>0,secondRangeStart:n.secondRangeStart>>>0,secondRangeCount:n.secondRangeCount>>>0};z.push({index:K,baseOffset:ae,timer0Min:k.timer0Min,timer0Max:k.timer0Max,timer0Count:H,vcount:k.vcount,rangeSeconds:d,totalMessages:o,keyCode:N,config:_}),ae+=o}}const q=z.reduce((N,V)=>N+V.totalMessages,0);return{conditions:t,startDate:i,startTimestampMs:i.getTime(),rangeSeconds:d,totalMessages:q,segments:z,timePlan:n}}function lt(t,n,r,i,s,e){return new Date(t,n-1,r,i,s,e)}function dt(t,n,r,i,s,e){return Date.UTC(t,n,r,i,s,e,0)}function ft(t){const n=ze[t.romVersion];if(!n)throw new Error(`ROMバージョン ${t.romVersion} は未対応です`);const r=n[t.romRegion];if(!r)throw new Error(`ROMリージョン ${t.romRegion} は未対応です`);return{nazo:[...r.nazo],vcountTimerRanges:r.vcountTimerRanges.map(i=>[...i])}}function gt(t,n){const r=[],i=t.timer0VCountConfig.timer0Range.min,s=t.timer0VCountConfig.timer0Range.max;let e=null;for(let l=i;l<=s;l+=1){const c=mt(n,l);e&&e.vcount===c&&l===e.timer0Max+1?e.timer0Max=l:(e&&r.push(e),e={timer0Min:l,timer0Max:l,vcount:c})}return e&&r.push(e),r}function mt(t,n){for(const[r,i,s]of t.vcountTimerRanges)if(n>=i&&n<=s)return r;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}function _t(t,n){const r=pt(t),i=(r[4]&255)<<8|r[5]&255,e=((r[0]&255|(r[1]&255)<<8|(r[2]&255)<<16|(r[3]&255)<<24)^st^n)>>>0;return{macLower:i,data7Swapped:Pe(e)}}function pt(t){const n=new Array(6).fill(0);for(let r=0;r<6;r+=1){const i=t[r]??0;n[r]=(Number(i)&255)>>>0}return n}function ht(t){switch(t){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function Pe(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}function yt(t){const n=new Uint32Array(t.length);for(let r=0;r<t.length;r+=1)n[r]=Pe(t[r]>>>0);return n}function bt(t){const n=new Date(t.getFullYear(),0,1),r=t.getTime()-n.getTime();return Math.floor(r/(1440*60*1e3))+1}function St(t){return t.getHours()*3600+t.getMinutes()*60+t.getSeconds()}class Be{calculateHash(n){if(n.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const r=1732584193,i=4023233417,s=2562383102,e=271733878,l=3285377520,c=new Array(80);for(let y=0;y<16;y++)c[y]=n[y];for(let y=16;y<80;y++)c[y]=this.leftRotate(c[y-3]^c[y-8]^c[y-14]^c[y-16],1);let a=r,d=i,u=s,p=e,b=l;for(let y=0;y<80;y++){let z;y<20?z=this.leftRotate(a,5)+(d&u|~d&p)+b+c[y]+1518500249&4294967295:y<40?z=this.leftRotate(a,5)+(d^u^p)+b+c[y]+1859775393&4294967295:y<60?z=this.leftRotate(a,5)+(d&u|d&p|u&p)+b+c[y]+2400959708&4294967295:z=this.leftRotate(a,5)+(d^u^p)+b+c[y]+3395469782&4294967295,b=p,p=u,u=this.leftRotate(d,30),d=a,a=z}const S=this.add32(r,a),P=this.add32(i,d),M=this.add32(s,u),x=this.add32(e,p),T=this.add32(l,b);return{h0:S,h1:P,h2:M,h3:x,h4:T}}leftRotate(n,r){return(n<<r|n>>>32-r)>>>0}add32(n,r){return(n+r&4294967295)>>>0}static hashToHex(n,r,i,s,e){return n.toString(16).padStart(8,"0")+r.toString(16).padStart(8,"0")+i.toString(16).padStart(8,"0")+s.toString(16).padStart(8,"0")+e.toString(16).padStart(8,"0")}}let Q=null,ie=null;async function wt(){return Q||ie||(ie=(async()=>{try{const t=await import("./wasm_pkg-DRWLiY4b.js");let n;if(typeof process<"u"&&!!process.versions?.node){const i=await import("./__vite-browser-external-9wXp6ZBx.js"),e=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");n={module_or_path:i.readFileSync(e)}}else n={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-D27IxIOn.wasm",import.meta.url)};return await t.default(n),Q={IntegratedSeedSearcher:t.IntegratedSeedSearcher,BWGenerationConfig:t.BWGenerationConfig,PokemonGenerator:t.PokemonGenerator,SeedEnumerator:t.SeedEnumerator,EncounterType:t.EncounterType,GameVersion:t.GameVersion,GameMode:t.GameMode,calculate_game_offset:t.calculate_game_offset,sha1_hash_batch:t.sha1_hash_batch},Q}catch(t){throw console.error("Failed to load WebAssembly module:",t),Q=null,ie=null,t}})(),ie)}function Rt(){if(!Q)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return Q}function Ue(){return Q!==null}const vt={DS:8,DS_LITE:6,"3DS":9};class Et{sha1;useWasm=!1;constructor(){this.sha1=new Be}async initializeWasm(){try{return await wt(),this.useWasm=!0,!0}catch(n){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",n),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&Ue()}getWasmModule(){return Rt()}setUseWasm(n){if(n&&!Ue()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=n}getROMParameters(n,r){const i=ze[n];if(!i)return console.error(`ROM version not found: ${n}`),null;const s=i[r];return s?{nazo:[...s.nazo],vcountTimerRanges:s.vcountTimerRanges.map(e=>[...e])}:(console.error(`ROM region not found: ${r} for version ${n}`),null)}toLittleEndian32(n){return((n&255)<<24|(n>>>8&255)<<16|(n>>>16&255)<<8|n>>>24&255)>>>0}toLittleEndian16(n){return(n&255)<<8|n>>>8&255}getDayOfWeek(n,r,i){return new Date(n,r-1,i).getDay()}generateMessage(n,r,i,s,e){const l=this.getROMParameters(n.romVersion,n.romRegion);if(!l)throw new Error(`No parameters found for ${n.romVersion} ${n.romRegion}`);const c=new Array(16).fill(0);for(let g=0;g<5;g++)c[g]=this.toLittleEndian32(l.nazo[g]);c[5]=this.toLittleEndian32(i<<16|r);const a=n.macAddress[4]<<8|n.macAddress[5];c[6]=a;const d=n.macAddress[0]<<0|n.macAddress[1]<<8|n.macAddress[2]<<16|n.macAddress[3]<<24,u=100663296,p=vt[n.hardware],b=d^u^p;c[7]=this.toLittleEndian32(b);const S=s.getFullYear()%100,P=s.getMonth()+1,M=s.getDate(),x=this.getDayOfWeek(s.getFullYear(),P,M),T=Math.floor(S/10)*16+S%10,y=Math.floor(P/10)*16+P%10,z=Math.floor(M/10)*16+M%10,ae=Math.floor(x/10)*16+x%10;c[8]=T<<24|y<<16|z<<8|ae;const q=s.getHours(),N=s.getMinutes(),V=s.getSeconds(),K=(n.hardware==="DS"||n.hardware==="DS_LITE")&&q>=12?1:0,k=Math.floor(q/10)*16+q%10,H=Math.floor(N/10)*16+N%10,o=Math.floor(V/10)*16+V%10;c[9]=K<<30|k<<24|H<<16|o<<8|0,c[10]=0,c[11]=0;const _=e??nt(n.keyInput);return c[12]=this.toLittleEndian32(_),c[13]=2147483648,c[14]=0,c[15]=416,c}calculateSeed(n){const r=this.sha1.calculateHash(n),i=BigInt(this.toLittleEndian32(r.h0)),e=BigInt(this.toLittleEndian32(r.h1))<<32n|i,a=e*0x5D588B656C078965n+0x269EC3n;return{seed:Number(a>>32n&0xFFFFFFFFn),hash:Be.hashToHex(r.h0,r.h1,r.h2,r.h3,r.h4),lcgSeed:e}}parseTargetSeeds(n){const r=n.split(`
`).map(l=>l.trim()).filter(l=>l.length>0),i=[],s=[],e=new Set;return r.forEach((l,c)=>{try{let a=l.toLowerCase();if(a.startsWith("0x")&&(a=a.substring(2)),!/^[0-9a-f]{1,8}$/.test(a)){s.push({line:c+1,value:l,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const d=parseInt(a,16);if(e.has(d))return;e.add(d),i.push(d)}catch(a){const d=a instanceof Error?a.message:String(a);s.push({line:c+1,value:l,error:d||"Failed to parse as hexadecimal number."})}}),{validSeeds:i,errors:s}}getVCountForTimer0(n,r){for(const[i,s,e]of n.vcountTimerRanges)if(r>=s&&r<=e)return i;return n.vcountTimerRanges.length>0?n.vcountTimerRanges[0][0]:96}}const He=Uint32Array.BYTES_PER_ELEMENT,le=2,de=le*He,ue=1,Ae=ue*He,pe=256,Ye=256*1024*1024,Me=2,Ct={requiredFeatures:[]};function $e(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function Pt(t){if(!$e())throw new Error("WebGPU is not available in this environment");const r=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!r)throw new Error("Failed to acquire WebGPU adapter");const i={requiredFeatures:Ct.requiredFeatures,requiredLimits:t?.requiredLimits,label:"seed-search-device"},s=await r.requestDevice(i);let e=!1;const l=s.lost.then(c=>(e=!0,console.warn("[webgpu] device lost:",c.message),c));return{getAdapter:()=>r,getDevice:()=>s,getQueue:()=>s.queue,getLimits:()=>s.limits,isLost:()=>e,waitForLoss:()=>l,getSupportedWorkgroupSize:(c=pe)=>{const a=s.limits,d=a.maxComputeInvocationsPerWorkgroup??c,u=a.maxComputeWorkgroupSizeX??c,p=Math.min(c,d,u);if(p<=0)throw new Error("WebGPU workgroup size limits are invalid");return p}}}var Mt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
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
struct CandidateRecord {\r
  message_index : u32,\r
  seed : u32,\r
};\r
\r
struct CandidateBuffer {\r
  records : array<CandidateRecord>,\r
};\r
\r
struct GroupCountBuffer {\r
  values : array<u32>,\r
};\r
\r
struct GroupOffsetBuffer {\r
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
@group(0) @binding(2) var<storage, read_write> candidate_buffer : CandidateBuffer;\r
@group(0) @binding(3) var<storage, read_write> group_counts_buffer : GroupCountBuffer;\r
@group(0) @binding(4) var<storage, read_write> group_offsets_buffer : GroupOffsetBuffer;\r
@group(0) @binding(5) var<storage, read_write> output_buffer : MatchOutputBuffer;\r
\r
var<workgroup> scan_values : array<u32, WORKGROUP_SIZE>;\r
var<workgroup> group_total_matches : u32;\r
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
  @builtin(global_invocation_id) global_id : vec3<u32>,\r
  @builtin(local_invocation_id) local_id : vec3<u32>,\r
  @builtin(local_invocation_index) local_linear_index : u32,\r
  @builtin(workgroup_id) workgroup_id : vec3<u32>\r
) {\r
\r
  let global_linear_index = global_id.x;\r
  let is_active = global_linear_index < config.message_count;\r
  let group_index = workgroup_id.x;\r
  let configured_workgroup_size = config.configured_workgroup_size;\r
\r
  var local_message_index : u32 = 0u;\r
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
  let match_flag = select(0u, 1u, matched);\r
  scan_values[local_linear_index] = match_flag;\r
  workgroupBarrier();\r
\r
  var offset = 1u;\r
  while (offset < WORKGROUP_SIZE) {\r
    workgroupBarrier();\r
    let current_value = scan_values[local_linear_index];\r
    var addend = 0u;\r
    if (local_linear_index >= offset) {\r
      addend = scan_values[local_linear_index - offset];\r
    }\r
    workgroupBarrier();\r
    scan_values[local_linear_index] = current_value + addend;\r
    offset = offset << 1u;\r
  }\r
\r
  workgroupBarrier();\r
  let inclusive_sum = scan_values[local_linear_index];\r
  if (local_linear_index == (WORKGROUP_SIZE - 1u)) {\r
    group_total_matches = inclusive_sum;\r
  }\r
  workgroupBarrier();\r
\r
  if (local_linear_index == 0u) {\r
    group_counts_buffer.values[group_index] = group_total_matches;\r
  }\r
\r
  if (match_flag == 0u) {\r
    return;\r
  }\r
\r
  let record_rank = inclusive_sum - 1u;\r
  let candidate_index = group_index * configured_workgroup_size + record_rank;\r
  if (candidate_index >= config.candidate_capacity) {\r
    return;\r
  }\r
\r
  local_message_index = global_linear_index;\r
  candidate_buffer.records[candidate_index].message_index = local_message_index;\r
  candidate_buffer.records[candidate_index].seed = seed;\r
}\r
\r
@compute @workgroup_size(1)\r
fn exclusive_scan_groups(@builtin(global_invocation_id) global_id : vec3<u32>) {\r
  if (global_id.x != 0u) {\r
    return;\r
  }\r
\r
  let group_count = config.groups_per_dispatch;\r
  var running_total = 0u;\r
  for (var i = 0u; i < group_count; i = i + 1u) {\r
    let count = group_counts_buffer.values[i];\r
    group_offsets_buffer.values[i] = running_total;\r
    running_total = running_total + count;\r
  }\r
\r
  atomicStore(&output_buffer.match_count, running_total);\r
}\r
\r
@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)\r
fn scatter_matches(@builtin(global_invocation_id) global_id : vec3<u32>) {\r
  let candidate_index = global_id.x;\r
  if (candidate_index >= config.candidate_capacity) {\r
    return;\r
  }\r
\r
  let configured_workgroup_size = config.configured_workgroup_size;\r
  let group_index = candidate_index / configured_workgroup_size;\r
  if (group_index >= config.groups_per_dispatch) {\r
    return;\r
  }\r
\r
  let local_rank = candidate_index - group_index * configured_workgroup_size;\r
  let group_match_count = group_counts_buffer.values[group_index];\r
  if (local_rank >= group_match_count) {\r
    return;\r
  }\r
\r
  let base_offset = group_offsets_buffer.values[group_index];\r
  let final_index = base_offset + local_rank;\r
  if (final_index >= config.message_count) {\r
    return;\r
  }\r
\r
  let record = candidate_buffer.records[candidate_index];\r
  output_buffer.records[final_index].message_index = record.message_index;\r
  output_buffer.records[final_index].seed = record.seed;\r
}\r
`;function xt(t){return Mt.replace(/WORKGROUP_SIZE_PLACEHOLDER/g,String(t))}function Tt(t,n){const r=t.createShaderModule({label:"gpu-seed-sha1-generated-module",code:xt(n)});r.getCompilationInfo?.().then(d=>{d.messages.length>0&&console.warn("[pipeline-factory] compilation diagnostics",d.messages.map(u=>({message:u.message,line:u.lineNum,column:u.linePos,type:u.type})))}).catch(d=>{console.warn("[pipeline-factory] compilation info failed",d)});const i=t.createBindGroupLayout({label:"gpu-seed-generate-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),s=t.createBindGroupLayout({label:"gpu-seed-scan-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),e=t.createBindGroupLayout({label:"gpu-seed-scatter-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),l=t.createComputePipeline({label:"gpu-seed-generate-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-generate-pipeline-layout",bindGroupLayouts:[i]}),compute:{module:r,entryPoint:"sha1_generate"}}),c=t.createComputePipeline({label:"gpu-seed-scan-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scan-pipeline-layout",bindGroupLayouts:[s]}),compute:{module:r,entryPoint:"exclusive_scan_groups"}}),a=t.createComputePipeline({label:"gpu-seed-scatter-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scatter-pipeline-layout",bindGroupLayouts:[e]}),compute:{module:r,entryPoint:"scatter_matches"}});return{pipelines:{generate:l,scan:c,scatter:a},layouts:{generate:i,scan:s,scatter:e}}}function Dt(t,n){const r=n?.slots??Me,i=n?.workgroupSize??pe;if(r<=0)throw new Error("buffer pool must have at least one slot");const s=Array.from({length:r},()=>({output:null,readback:null,candidate:null,groupCounts:null,groupOffsets:null,matchCount:null,outputSize:0,readbackSize:0,candidateSize:0,groupCountSize:0,groupOffsetSize:0,matchCountSize:0})),e=a=>Math.ceil(a/256)*256;return{get slotCount(){return s.length},acquire:(a,d)=>{if(a<0||a>=s.length)throw new Error(`buffer slot ${a} is out of range`);if(!Number.isFinite(d)||d<=0)throw new Error("messageCount must be a positive integer");const u=s[a],p=d,b=e(Ae+p*de),S=Math.max(1,Math.ceil(d/i)),P=S*i,M=e(P*de),x=e(S*Uint32Array.BYTES_PER_ELEMENT),T=e(Ae);return(!u.output||b>u.outputSize)&&(u.output?.destroy(),u.output=t.createBuffer({label:`gpu-seed-output-${a}`,size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),u.outputSize=b),(!u.readback||b>u.readbackSize)&&(u.readback?.destroy(),u.readback=t.createBuffer({label:`gpu-seed-readback-${a}`,size:b,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.readbackSize=b),(!u.candidate||M>u.candidateSize)&&(u.candidate?.destroy(),u.candidate=t.createBuffer({label:`gpu-seed-candidate-${a}`,size:M,usage:GPUBufferUsage.STORAGE}),u.candidateSize=M),(!u.groupCounts||x>u.groupCountSize)&&(u.groupCounts?.destroy(),u.groupCounts=t.createBuffer({label:`gpu-seed-group-counts-${a}`,size:x,usage:GPUBufferUsage.STORAGE}),u.groupCountSize=x),(!u.groupOffsets||x>u.groupOffsetSize)&&(u.groupOffsets?.destroy(),u.groupOffsets=t.createBuffer({label:`gpu-seed-group-offsets-${a}`,size:x,usage:GPUBufferUsage.STORAGE}),u.groupOffsetSize=x),(!u.matchCount||T>u.matchCountSize)&&(u.matchCount?.destroy(),u.matchCount=t.createBuffer({label:`gpu-seed-match-header-${a}`,size:T,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.matchCountSize=T),{output:u.output,readback:u.readback,candidate:u.candidate,groupCounts:u.groupCounts,groupOffsets:u.groupOffsets,matchCount:u.matchCount,outputSize:u.outputSize,candidateCapacity:P,groupCount:S,maxRecords:p}},dispose:()=>{for(const a of s)a.output?.destroy(),a.readback?.destroy(),a.candidate?.destroy(),a.groupCounts?.destroy(),a.groupOffsets?.destroy(),a.matchCount?.destroy(),a.output=null,a.readback=null,a.candidate=null,a.groupCounts=null,a.groupOffsets=null,a.matchCount=null,a.outputSize=0,a.readbackSize=0,a.candidateSize=0,a.groupCountSize=0,a.groupOffsetSize=0,a.matchCountSize=0}}}function Ot(t,n){const r=n?.hostMemoryLimitBytes??Ye,i=n?.bufferSetCount??Me,s=(()=>{const a=n?.hostMemoryLimitPerSlot;if(typeof a=="number"&&Number.isFinite(a)&&a>0)return a;const d=Math.floor(r/i);return Math.max(1,d)})(),e=n?.workgroupSize??pe;if(r<=0)throw new Error("host memory limit must be positive");if(i<=0)throw new Error("buffer set count must be positive");const l=a=>{const u=t.getDevice().limits,p=Math.max(1,u.maxStorageBufferBindingSize??de),b=Math.max(1,Math.floor(p/de)),S=Math.max(1,Math.floor(s/de)),P=t.getSupportedWorkgroupSize(e),M=u.maxComputeWorkgroupsPerDimension??65535,x=Math.max(1,P*M),T=Math.min(b,S,x);return a<=T?a<=1?1:Math.max(1,Math.min(T,Math.ceil(a/2))):T};return{computePlan:a=>{if(!Number.isFinite(a)||a<0)throw new Error("totalMessages must be a non-negative finite value");if(a===0)return{maxMessagesPerDispatch:0,dispatches:[]};const d=l(a),u=[];let p=a,b=0;for(;p>0;){const S=Math.min(d,p);u.push({baseOffset:b,messageCount:S}),b+=S,p-=S}if(u.length===1&&a>1){const S=u[0],P=Math.ceil(S.messageCount/2),M=S.messageCount-P;M>0&&(u[0]={baseOffset:S.baseOffset,messageCount:P},u.push({baseOffset:S.baseOffset+P,messageCount:M}))}return{maxMessagesPerDispatch:d,dispatches:u}}}}const Bt=32,Ut=500,At=1024,Ee=new Uint32Array([0]),zt=$e;function kt(t){const n=o=>Me,r=Ye,i=n(),s=(()=>{const o=Math.floor(r/i);return Math.max(1,o)})(),e={workgroupSize:pe,bufferSlotCount:i,hostMemoryLimitBytes:r,hostMemoryLimitPerSlotBytes:s,deviceContext:null,pipelines:null,bindGroupLayouts:null,configBuffer:null,configData:null,bufferPool:null,planner:null,targetBuffer:null,targetBufferCapacity:0,seedCalculator:new Et,isRunning:!1,isPaused:!1,shouldStop:!1,lastProgressUpdateMs:0,timerState:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1}},l=async(o,_,g)=>(async()=>await Promise.resolve(g()))(),c=async()=>{if(e.pipelines&&e.bufferPool&&e.planner&&e.deviceContext)return;const o=await Pt(),_=o.getDevice(),g=o.getSupportedWorkgroupSize(e.workgroupSize),{pipelines:w,layouts:R}=Tt(_,g),E=new Uint32Array(Bt),D=H(E.byteLength),h=_.createBuffer({label:"gpu-seed-config-buffer",size:D,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),C=Dt(_,{slots:e.bufferSlotCount,workgroupSize:g}),v=Ot(o,{workgroupSize:g,bufferSetCount:e.bufferSlotCount,hostMemoryLimitBytes:e.hostMemoryLimitBytes,hostMemoryLimitPerSlot:e.hostMemoryLimitPerSlotBytes});e.deviceContext=o,e.pipelines=w,e.bindGroupLayouts=R,e.configBuffer=h,e.configData=E,e.bufferPool=C,e.planner=v,e.workgroupSize=g},a=o=>{if(!e.deviceContext)throw new Error("WebGPU device is not initialized");const _=e.deviceContext.getDevice(),g=o.length,w=1+g,R=H(w*Uint32Array.BYTES_PER_ELEMENT),E=e.targetBufferCapacity;if(!e.targetBuffer||E<g){e.targetBuffer?.destroy(),e.targetBuffer=_.createBuffer({label:"gpu-seed-target-buffer",size:R,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const v=Math.floor(R/Uint32Array.BYTES_PER_ELEMENT)-1;e.targetBufferCapacity=Math.max(v,g)}const h=new Uint32Array(w);h[0]=g>>>0;for(let v=0;v<g;v+=1)h[1+v]=o[v]>>>0;const C=h.byteLength;_.queue.writeBuffer(e.targetBuffer,0,h.buffer,h.byteOffset,C)},d=async o=>{if(e.isRunning)throw new Error("WebGPU search is already running");if((!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)&&await c(),!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)throw new Error("WebGPU runner failed to initialize");const{context:_,targetSeeds:g,callbacks:w,signal:R}=o;if(_.totalMessages===0){w.onComplete("探索対象の組み合わせが存在しません");return}if(!e.bindGroupLayouts)throw new Error("WebGPU runner missing bind group layout");a(g),e.isRunning=!0,e.isPaused=!1,e.shouldStop=!1,e.lastProgressUpdateMs=Date.now();const E={currentStep:0,totalSteps:_.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0};let D;if(R)if(R.aborted)e.shouldStop=!0;else{const h=()=>{e.shouldStop=!0};R.addEventListener("abort",h),D=()=>R.removeEventListener("abort",h)}ae();try{await P(_,E,w);const h=V(),C={...E,elapsedTime:h,estimatedTimeRemaining:0};e.shouldStop?w.onStopped("検索を停止しました",C):(w.onProgress(C),w.onComplete(`検索が完了しました。${E.matchesFound}件ヒットしました。`))}catch(h){const C=h instanceof Error?h.message:"WebGPU検索中に不明なエラーが発生しました",v=h instanceof GPUValidationError?"WEBGPU_VALIDATION_ERROR":void 0;throw w.onError(C,v),h}finally{e.isRunning=!1,e.isPaused=!1,e.shouldStop=!1,q(),D&&D()}},u=()=>{!e.isRunning||e.isPaused||(e.isPaused=!0,q())},p=()=>{!e.isRunning||!e.isPaused||(e.isPaused=!1,N())},b=()=>{e.isRunning&&(e.shouldStop=!0,e.isPaused=!1,N())},S=()=>{e.bufferPool?.dispose(),e.configBuffer?.destroy(),e.configBuffer=null,e.configData=null,e.pipelines=null,e.bindGroupLayouts=null,e.bufferPool=null,e.planner=null,e.deviceContext=null,e.targetBuffer?.destroy(),e.targetBuffer=null,e.targetBufferCapacity=0},P=async(o,_,g)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const R=e.deviceContext.getDevice().queue,E=e.bufferPool.slotCount,D=Array.from({length:E},(f,A)=>E-1-A),h=[],C=[],v=[],Y=f=>{v.push(f)},U=()=>new Promise(f=>{if(D.length>0){const A=D.pop();f(A);return}h.push(f)}),G=f=>{const A=h.shift();if(A){A(f);return}D.push(f)};let m=0;for(const f of o.segments){if(e.shouldStop)break;const A=await l("planner.computePlan",{segmentIndex:f.index,totalMessages:f.totalMessages},()=>Promise.resolve(e.planner.computePlan(f.totalMessages)));for(const I of A.dispatches){if(e.shouldStop||(await K(),e.shouldStop))break;const L=await U();if(e.shouldStop){G(L);break}const Z={segment:f,dispatchIndex:m,messageCount:I.messageCount,slotIndex:L},F=M(Z,I.baseOffset,o,_,g,R,G,Y);C.push(F),m+=1}}C.length>0&&await Promise.all(C),v.length>0&&await Promise.all(v)},M=async(o,_,g,w,R,E,D,h)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const C=e.deviceContext.getDevice(),v=e.configBuffer,Y=e.configData,U=e.bindGroupLayouts,G=e.pipelines,m=e.targetBuffer,f=e.bufferPool.acquire(o.slotIndex,o.messageCount);let A=!1,I=!1;const L=()=>{A||(A=!0,D(o.slotIndex))},Z=Math.ceil(o.messageCount/e.workgroupSize),F=Math.max(1,Math.ceil(f.candidateCapacity/e.workgroupSize)),$=H(ue*Uint32Array.BYTES_PER_ELEMENT),W={dispatchIndex:o.dispatchIndex,messageCount:o.messageCount,slotIndex:o.slotIndex,workgroupCount:Z,scatterWorkgroupCount:F,candidateCapacity:f.candidateCapacity,groupCount:f.groupCount,segmentIndex:o.segment.index,segmentBaseOffset:_};try{await l("dispatch",W,async()=>{E.writeBuffer(f.output,0,Ee.buffer,Ee.byteOffset,Ee.byteLength),z(o.segment,_,o.messageCount,f.groupCount,f.candidateCapacity),E.writeBuffer(v,0,Y.buffer,Y.byteOffset,Y.byteLength);const oe=C.createBindGroup({label:`gpu-seed-generate-group-${o.dispatchIndex}`,layout:U.generate,entries:[{binding:0,resource:{buffer:v}},{binding:1,resource:{buffer:m}},{binding:2,resource:{buffer:f.candidate}},{binding:3,resource:{buffer:f.groupCounts}}]}),fe=C.createBindGroup({label:`gpu-seed-scan-group-${o.dispatchIndex}`,layout:U.scan,entries:[{binding:0,resource:{buffer:v}},{binding:3,resource:{buffer:f.groupCounts}},{binding:4,resource:{buffer:f.groupOffsets}},{binding:5,resource:{buffer:f.output}}]}),ye=C.createBindGroup({label:`gpu-seed-scatter-group-${o.dispatchIndex}`,layout:U.scatter,entries:[{binding:0,resource:{buffer:v}},{binding:2,resource:{buffer:f.candidate}},{binding:3,resource:{buffer:f.groupCounts}},{binding:4,resource:{buffer:f.groupOffsets}},{binding:5,resource:{buffer:f.output}}]}),j=C.createCommandEncoder({label:`gpu-seed-compute-${o.dispatchIndex}`}),ee=j.beginComputePass({label:`gpu-seed-generate-pass-${o.dispatchIndex}`});ee.setPipeline(G.generate),ee.setBindGroup(0,oe),ee.dispatchWorkgroups(Z),ee.end();const te=j.beginComputePass({label:`gpu-seed-scan-pass-${o.dispatchIndex}`});te.setPipeline(G.scan),te.setBindGroup(0,fe),te.dispatchWorkgroups(1),te.end();const ne=j.beginComputePass({label:`gpu-seed-scatter-pass-${o.dispatchIndex}`});ne.setPipeline(G.scatter),ne.setBindGroup(0,ye),ne.dispatchWorkgroups(F),ne.end(),j.copyBufferToBuffer(f.output,0,f.matchCount,0,$);const ge=j.finish();await l("dispatch.submit",{...W},async()=>{await l("dispatch.submit.encode",{...W},async()=>{E.submit([ge])})});const be=await l("dispatch.mapMatchCount",{...W,headerCopyBytes:$},async()=>{await f.matchCount.mapAsync(GPUMapMode.READ,0,$);const se=new Uint32Array(f.matchCount.getMappedRange(0,$))[0]??0;return f.matchCount.unmap(),se}),we=Math.min(be,f.maxRecords)*le*Uint32Array.BYTES_PER_ELEMENT,X=H(ue*Uint32Array.BYTES_PER_ELEMENT+we);await l("dispatch.copyResults",{...W,totalCopyBytes:X},async()=>{const re=C.createCommandEncoder({label:`gpu-seed-copy-${o.dispatchIndex}`});re.copyBufferToBuffer(f.output,0,f.readback,0,X);const se=re.finish();await l("dispatch.copyResults.encode",{...W,totalCopyBytes:X},async()=>{E.submit([se])})});const Ke=(async()=>{try{const{results:re,clampedMatchCount:se}=await l("dispatch.mapResults",{...W,totalCopyBytes:X},async()=>{await f.readback.mapAsync(GPUMapMode.READ,0,X);const qe=f.readback.getMappedRange(0,X),Re=new Uint32Array(qe),Ze=Re[0]??0,je=Math.max(0,Math.floor((Re.length-ue)/le)),xe=Math.min(Ze,f.maxRecords,je),Te=ue+xe*le,De=new Uint32Array(Te);return De.set(Re.subarray(0,Te)),f.readback.unmap(),{results:De,clampedMatchCount:xe}});try{L(),await l("dispatch.processMatches",{...W,matchCount:se},()=>x(re,se,o,_,g,w,R))}finally{L()}}catch(re){throw L(),re}})();I=!0,h(Ke)})}finally{I||L()}},x=async(o,_,g,w,R,E,D)=>{const h=g.segment,C=h.rangeSeconds,v=Math.max(C,1),Y=Math.max(h.config.vcountCount,1),U=v,G=U*Y,m=h.config.timer0Min,f=h.config.vcountMin,A=w;for(let I=0;I<_&&!(e.shouldStop||I%At===0&&(await K(),e.shouldStop));I+=1){const L=ue+I*le,Z=o[L],F=A+Z,$=o[L+1]>>>0,W=Math.floor(F/G),oe=F-W*G,fe=Math.floor(oe/U),ye=oe-fe*U,j=m+W,ee=f+fe,te=Oe(R.timePlan,ye),ne=g.segment.keyCode,ge=e.seedCalculator.generateMessage(R.conditions,j,ee,te,ne),{hash:be,seed:Se,lcgSeed:we}=e.seedCalculator.calculateSeed(ge);Se!==$&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:$,cpuSeed:Se,messageIndex:F});const X={seed:$,datetime:te,timer0:j,vcount:ee,keyCode:ne,conditions:R.conditions,message:ge,sha1Hash:be,lcgSeed:we,isMatch:!0};D.onResult(X),E.matchesFound+=1}if(g.messageCount>0){const I=g.messageCount-1,L=A+I,Z=Math.floor(L/G),F=L-Z*G,$=Math.floor(F/U),W=F-$*U,oe=Oe(R.timePlan,W).toISOString();E.currentDateTime=oe}E.currentStep+=g.messageCount,T(E,D)},T=(o,_)=>{const g=Date.now();if(g-e.lastProgressUpdateMs<Ut&&o.currentStep<o.totalSteps)return;const w=V(),R=y(o.currentStep,o.totalSteps,w);_.onProgress({currentStep:o.currentStep,totalSteps:o.totalSteps,elapsedTime:w,estimatedTimeRemaining:R,matchesFound:o.matchesFound,currentDateTime:o.currentDateTime}),e.lastProgressUpdateMs=g},y=(o,_,g)=>{if(o===0||o>=_)return 0;const w=g/o,R=_-o;return Math.round(w*R)},z=(o,_,g,w,R)=>{if(!e.configData)throw new Error("config buffer not prepared");const E=Math.max(o.config.rangeSeconds,1),D=Math.max(o.config.vcountCount,1),h=E,C=h*D,v=Math.floor(_/C),Y=_-v*C,U=Math.floor(Y/h),G=Y-U*h,m=e.configData;m[0]=g>>>0,m[1]=v>>>0,m[2]=U>>>0,m[3]=G>>>0,m[4]=o.config.rangeSeconds>>>0,m[5]=o.config.timer0Min>>>0,m[6]=o.config.timer0Count>>>0,m[7]=o.config.vcountMin>>>0,m[8]=o.config.vcountCount>>>0,m[9]=o.config.startSecondOfDay>>>0,m[10]=o.config.startDayOfWeek>>>0,m[11]=o.config.macLower>>>0,m[12]=o.config.data7Swapped>>>0,m[13]=o.config.keyInputSwapped>>>0,m[14]=o.config.hardwareType>>>0;for(let f=0;f<o.config.nazoSwapped.length;f+=1)m[15+f]=o.config.nazoSwapped[f]>>>0;m[20]=o.config.startYear>>>0,m[21]=o.config.startDayOfYear>>>0,m[22]=w>>>0,m[23]=e.workgroupSize>>>0,m[24]=R>>>0,m[25]=o.config.dayCount>>>0,m[26]=o.config.hourRangeStart>>>0,m[27]=o.config.hourRangeCount>>>0,m[28]=o.config.minuteRangeStart>>>0,m[29]=o.config.minuteRangeCount>>>0,m[30]=o.config.secondRangeStart>>>0,m[31]=o.config.secondRangeCount>>>0},ae=()=>{e.timerState.cumulativeRunTime=0,e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1},q=()=>{e.timerState.isPaused||(e.timerState.cumulativeRunTime+=Date.now()-e.timerState.segmentStartTime,e.timerState.isPaused=!0)},N=()=>{e.timerState.isPaused&&(e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1)},V=()=>e.timerState.isPaused?e.timerState.cumulativeRunTime:e.timerState.cumulativeRunTime+(Date.now()-e.timerState.segmentStartTime),K=async()=>{for(;e.isPaused&&!e.shouldStop;)await k(25)},k=o=>new Promise(_=>setTimeout(_,o)),H=o=>Math.ceil(o/256)*256;return{init:c,run:d,pause:u,resume:p,stop:b,dispose:S}}const Ve=self,B={isRunning:!1,isPaused:!1},he=kt();let _e=null;function O(t){Ve.postMessage(t)}function Gt(){O({type:"READY",message:"WebGPU worker initialized"})}function ce(){B.isRunning=!1,B.isPaused=!1,_e=null}function Lt(){return zt()?!0:(O({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Wt(t){if(B.isRunning){O({type:"ERROR",error:"Search is already running"});return}if(!t.conditions||!t.targetSeeds){O({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!Lt())return;B.isRunning=!0,B.isPaused=!1;let n;try{n=ct(t.conditions)}catch(s){ce();const e=s instanceof Error?s.message:"検索条件の解析中にエラーが発生しました";O({type:"ERROR",error:e,errorCode:"WEBGPU_CONTEXT_ERROR"});return}_e=new AbortController;const r={onProgress:s=>{O({type:"PROGRESS",progress:s})},onResult:s=>{O({type:"RESULT",result:s})},onComplete:s=>{ce(),O({type:"COMPLETE",message:s})},onError:(s,e)=>{ce(),O({type:"ERROR",error:s,errorCode:e})},onPaused:()=>{B.isPaused=!0,O({type:"PAUSED"})},onResumed:()=>{B.isPaused=!1,O({type:"RESUMED"})},onStopped:(s,e)=>{ce(),O({type:"STOPPED",message:s,progress:e})}},i={context:n,targetSeeds:t.targetSeeds,callbacks:r,signal:_e.signal};try{await he.run(i)}catch(s){if(!B.isRunning)return;ce();const e=s instanceof Error?s.message:"WebGPU search failed with unknown error";O({type:"ERROR",error:e,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function It(){!B.isRunning||B.isPaused||(he.pause(),B.isPaused=!0,O({type:"PAUSED"}))}function Ft(){!B.isRunning||!B.isPaused||(he.resume(),B.isPaused=!1,O({type:"RESUMED"}))}function Nt(){B.isRunning&&(he.stop(),_e?.abort())}Gt();Ve.onmessage=t=>{const n=t.data;switch(n.type){case"START_SEARCH":Wt(n);break;case"PAUSE_SEARCH":It();break;case"RESUME_SEARCH":Ft();break;case"STOP_SEARCH":Nt();break;default:O({type:"ERROR",error:`Unknown request type: ${n.type}`})}};
