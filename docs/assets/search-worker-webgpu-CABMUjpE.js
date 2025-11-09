const Be={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},Ie=Date.UTC(2e3,0,1,0,0,0),He=100663296,Ne={DS:8,DS_LITE:6,"3DS":9};function Ye(t){const r=Ce(t.dateRange.startYear,t.dateRange.startMonth,t.dateRange.startDay,t.dateRange.startHour,t.dateRange.startMinute,t.dateRange.startSecond),n=Ce(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond);if(r.getTime()>n.getTime())throw new Error("開始日時が終了日時より後ろに設定されています");const i=Ve(t),s=$e(t,i),e=Me(t.dateRange.startYear,t.dateRange.startMonth,t.dateRange.startDay,t.dateRange.startHour,t.dateRange.startMinute,t.dateRange.startSecond),c=Me(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond),f=Math.floor((e-Ie)/1e3);if(f<0)throw new Error("2000年より前の日時は指定できません");const a=Math.floor((c-e)/1e3)+1;if(a<=0)throw new Error("探索秒数が0秒以下です");const d=r.getFullYear(),u=Xe(r),y=Qe(r),b=r.getDay(),w=Ne[t.hardware],{macLower:C,data7Swapped:M}=Ke(t.macAddress,w),O=we(t.keyInput>>>0),U=Je(i.nazo),p=[];let W=0;for(let F=0;F<s.length;F+=1){const x=s[F],V=x.timer0Max-x.timer0Min+1,$=a*V,se={startSecondsSince2000:f>>>0,rangeSeconds:a>>>0,timer0Min:x.timer0Min>>>0,timer0Max:x.timer0Max>>>0,timer0Count:V>>>0,vcountMin:x.vcount>>>0,vcountMax:x.vcount>>>0,vcountCount:1,totalMessages:$>>>0,hardwareType:je(t.hardware),macLower:C>>>0,data7Swapped:M>>>0,keyInputSwapped:O>>>0,nazoSwapped:U,startYear:d>>>0,startDayOfYear:u>>>0,startSecondOfDay:y>>>0,startDayOfWeek:b>>>0};p.push({index:F,baseOffset:W,timer0Min:x.timer0Min,timer0Max:x.timer0Max,timer0Count:V,vcount:x.vcount,rangeSeconds:a,totalMessages:$,config:se}),W+=$}const j=p.reduce((F,x)=>F+x.totalMessages,0);return{conditions:t,startDate:r,startTimestampMs:r.getTime(),rangeSeconds:a,totalMessages:j,segments:p}}function Ce(t,r,n,i,s,e){return new Date(t,r-1,n,i,s,e)}function Me(t,r,n,i,s,e){return Date.UTC(t,r-1,n,i,s,e,0)}function Ve(t){const r=Be[t.romVersion];if(!r)throw new Error(`ROMバージョン ${t.romVersion} は未対応です`);const n=r[t.romRegion];if(!n)throw new Error(`ROMリージョン ${t.romRegion} は未対応です`);return{nazo:[...n.nazo],vcountTimerRanges:n.vcountTimerRanges.map(i=>[...i])}}function $e(t,r){const n=[],i=t.timer0VCountConfig.timer0Range.min,s=t.timer0VCountConfig.timer0Range.max;let e=null;for(let c=i;c<=s;c+=1){const f=qe(r,c);e&&e.vcount===f&&c===e.timer0Max+1?e.timer0Max=c:(e&&n.push(e),e={timer0Min:c,timer0Max:c,vcount:f})}return e&&n.push(e),n}function qe(t,r){for(const[n,i,s]of t.vcountTimerRanges)if(r>=i&&r<=s)return n;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}function Ke(t,r){const n=Ze(t),i=(n[4]&255)<<8|n[5]&255,e=((n[0]&255|(n[1]&255)<<8|(n[2]&255)<<16|(n[3]&255)<<24)^He^r)>>>0;return{macLower:i,data7Swapped:we(e)}}function Ze(t){const r=new Array(6).fill(0);for(let n=0;n<6;n+=1){const i=t[n]??0;r[n]=(Number(i)&255)>>>0}return r}function je(t){switch(t){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function we(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}function Je(t){const r=new Uint32Array(t.length);for(let n=0;n<t.length;n+=1)r[n]=we(t[n]>>>0);return r}function Xe(t){const r=new Date(t.getFullYear(),0,1),n=t.getTime()-r.getTime();return Math.floor(n/(1440*60*1e3))+1}function Qe(t){return t.getHours()*3600+t.getMinutes()*60+t.getSeconds()}class xe{calculateHash(r){if(r.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const n=1732584193,i=4023233417,s=2562383102,e=271733878,c=3285377520,f=new Array(80);for(let p=0;p<16;p++)f[p]=r[p];for(let p=16;p<80;p++)f[p]=this.leftRotate(f[p-3]^f[p-8]^f[p-14]^f[p-16],1);let a=n,d=i,u=s,y=e,b=c;for(let p=0;p<80;p++){let W;p<20?W=this.leftRotate(a,5)+(d&u|~d&y)+b+f[p]+1518500249&4294967295:p<40?W=this.leftRotate(a,5)+(d^u^y)+b+f[p]+1859775393&4294967295:p<60?W=this.leftRotate(a,5)+(d&u|d&y|u&y)+b+f[p]+2400959708&4294967295:W=this.leftRotate(a,5)+(d^u^y)+b+f[p]+3395469782&4294967295,b=y,y=u,u=this.leftRotate(d,30),d=a,a=W}const w=this.add32(n,a),C=this.add32(i,d),M=this.add32(s,u),O=this.add32(e,y),U=this.add32(c,b);return{h0:w,h1:C,h2:M,h3:O,h4:U}}leftRotate(r,n){return(r<<n|r>>>32-n)>>>0}add32(r,n){return(r+n&4294967295)>>>0}static hashToHex(r,n,i,s,e){return r.toString(16).padStart(8,"0")+n.toString(16).padStart(8,"0")+i.toString(16).padStart(8,"0")+s.toString(16).padStart(8,"0")+e.toString(16).padStart(8,"0")}}let Z=null,ue=null;async function et(){return Z||ue||(ue=(async()=>{try{const t=await import("./wasm_pkg-HLQ4E-jy.js");let r;if(typeof process<"u"&&!!process.versions?.node){const i=await import("./__vite-browser-external-9wXp6ZBx.js"),e=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");r={module_or_path:i.readFileSync(e)}}else r={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-CDMG1ChF.wasm",import.meta.url)};return await t.default(r),Z={IntegratedSeedSearcher:t.IntegratedSeedSearcher,BWGenerationConfig:t.BWGenerationConfig,PokemonGenerator:t.PokemonGenerator,SeedEnumerator:t.SeedEnumerator,EncounterType:t.EncounterType,GameVersion:t.GameVersion,GameMode:t.GameMode,calculate_game_offset:t.calculate_game_offset,sha1_hash_batch:t.sha1_hash_batch},Z}catch(t){throw console.error("Failed to load WebAssembly module:",t),Z=null,ue=null,t}})(),ue)}function tt(){if(!Z)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return Z}function Te(){return Z!==null}const rt={DS:8,DS_LITE:6,"3DS":9};class nt{sha1;useWasm=!1;constructor(){this.sha1=new xe}async initializeWasm(){try{return await et(),this.useWasm=!0,!0}catch(r){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",r),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&Te()}getWasmModule(){return tt()}setUseWasm(r){if(r&&!Te()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=r}getROMParameters(r,n){const i=Be[r];if(!i)return console.error(`ROM version not found: ${r}`),null;const s=i[n];return s?{nazo:[...s.nazo],vcountTimerRanges:s.vcountTimerRanges.map(e=>[...e])}:(console.error(`ROM region not found: ${n} for version ${r}`),null)}toLittleEndian32(r){return((r&255)<<24|(r>>>8&255)<<16|(r>>>16&255)<<8|r>>>24&255)>>>0}toLittleEndian16(r){return(r&255)<<8|r>>>8&255}getDayOfWeek(r,n,i){return new Date(r,n-1,i).getDay()}generateMessage(r,n,i,s){const e=this.getROMParameters(r.romVersion,r.romRegion);if(!e)throw new Error(`No parameters found for ${r.romVersion} ${r.romRegion}`);const c=new Array(16).fill(0);for(let o=0;o<5;o++)c[o]=this.toLittleEndian32(e.nazo[o]);c[5]=this.toLittleEndian32(i<<16|n);const f=r.macAddress[4]<<8|r.macAddress[5];c[6]=f;const a=r.macAddress[0]<<0|r.macAddress[1]<<8|r.macAddress[2]<<16|r.macAddress[3]<<24,d=100663296,u=rt[r.hardware],y=a^d^u;c[7]=this.toLittleEndian32(y);const b=s.getFullYear()%100,w=s.getMonth()+1,C=s.getDate(),M=this.getDayOfWeek(s.getFullYear(),w,C),O=Math.floor(b/10)*16+b%10,U=Math.floor(w/10)*16+w%10,p=Math.floor(C/10)*16+C%10,W=Math.floor(M/10)*16+M%10;c[8]=O<<24|U<<16|p<<8|W;const j=s.getHours(),F=s.getMinutes(),x=s.getSeconds(),V=(r.hardware==="DS"||r.hardware==="DS_LITE")&&j>=12?1:0,$=Math.floor(j/10)*16+j%10,se=Math.floor(F/10)*16+F%10,te=Math.floor(x/10)*16+x%10;return c[9]=V<<30|$<<24|se<<16|te<<8|0,c[10]=0,c[11]=0,c[12]=this.toLittleEndian32(r.keyInput),c[13]=2147483648,c[14]=0,c[15]=416,c}calculateSeed(r){const n=this.sha1.calculateHash(r),i=BigInt(this.toLittleEndian32(n.h0)),a=(BigInt(this.toLittleEndian32(n.h1))<<32n|i)*0x5D588B656C078965n+0x269EC3n;return{seed:Number(a>>32n&0xFFFFFFFFn),hash:xe.hashToHex(n.h0,n.h1,n.h2,n.h3,n.h4)}}parseTargetSeeds(r){const n=r.split(`
`).map(c=>c.trim()).filter(c=>c.length>0),i=[],s=[],e=new Set;return n.forEach((c,f)=>{try{let a=c.toLowerCase();if(a.startsWith("0x")&&(a=a.substring(2)),!/^[0-9a-f]{1,8}$/.test(a)){s.push({line:f+1,value:c,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const d=parseInt(a,16);if(e.has(d))return;e.add(d),i.push(d)}catch(a){const d=a instanceof Error?a.message:String(a);s.push({line:f+1,value:c,error:d||"Failed to parse as hexadecimal number."})}}),{validSeeds:i,errors:s}}getVCountForTimer0(r,n){for(const[i,s,e]of r.vcountTimerRanges)if(n>=s&&n<=e)return i;return r.vcountTimerRanges.length>0?r.vcountTimerRanges[0][0]:96}}const Ue=Uint32Array.BYTES_PER_ELEMENT,ce=2,le=ce*Ue,oe=1,Oe=oe*Ue,pe=256,De=256*1024*1024,Se=2,at={requiredFeatures:[]};function ze(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function ot(t){if(!ze())throw new Error("WebGPU is not available in this environment");const n=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!n)throw new Error("Failed to acquire WebGPU adapter");const i={requiredFeatures:at.requiredFeatures,requiredLimits:t?.requiredLimits,label:"seed-search-device"},s=await n.requestDevice(i);let e=!1;const c=s.lost.then(f=>(e=!0,console.warn("[webgpu] device lost:",f.message),f));return{getAdapter:()=>n,getDevice:()=>s,getQueue:()=>s.queue,getLimits:()=>s.limits,isLost:()=>e,waitForLoss:()=>c,getSupportedWorkgroupSize:(f=pe)=>{const a=s.limits,d=a.maxComputeInvocationsPerWorkgroup??f,u=a.maxComputeWorkgroupSizeX??f,y=Math.min(f,d,u);if(y<=0)throw new Error("WebGPU workgroup size limits are invalid");return y}}}var st=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
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
\r
    let total_seconds = config.start_second_of_day + second_offset;\r
    let day_offset = total_seconds / 86400u;\r
    let seconds_of_day = total_seconds - day_offset * 86400u;\r
\r
    let hour = seconds_of_day / 3600u;\r
    let minute = (seconds_of_day % 3600u) / 60u;\r
    let second = seconds_of_day % 60u;\r
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
`;function ut(t){return st.replace(/WORKGROUP_SIZE_PLACEHOLDER/g,String(t))}function it(t,r){const n=t.createShaderModule({label:"gpu-seed-sha1-generated-module",code:ut(r)});n.getCompilationInfo?.().then(d=>{d.messages.length>0&&console.warn("[pipeline-factory] compilation diagnostics",d.messages.map(u=>({message:u.message,line:u.lineNum,column:u.linePos,type:u.type})))}).catch(d=>{console.warn("[pipeline-factory] compilation info failed",d)});const i=t.createBindGroupLayout({label:"gpu-seed-generate-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),s=t.createBindGroupLayout({label:"gpu-seed-scan-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),e=t.createBindGroupLayout({label:"gpu-seed-scatter-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),c=t.createComputePipeline({label:"gpu-seed-generate-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-generate-pipeline-layout",bindGroupLayouts:[i]}),compute:{module:n,entryPoint:"sha1_generate"}}),f=t.createComputePipeline({label:"gpu-seed-scan-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scan-pipeline-layout",bindGroupLayouts:[s]}),compute:{module:n,entryPoint:"exclusive_scan_groups"}}),a=t.createComputePipeline({label:"gpu-seed-scatter-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scatter-pipeline-layout",bindGroupLayouts:[e]}),compute:{module:n,entryPoint:"scatter_matches"}});return{pipelines:{generate:c,scan:f,scatter:a},layouts:{generate:i,scan:s,scatter:e}}}function ct(t,r){const n=r?.slots??Se,i=r?.workgroupSize??pe;if(n<=0)throw new Error("buffer pool must have at least one slot");const s=Array.from({length:n},()=>({output:null,readback:null,candidate:null,groupCounts:null,groupOffsets:null,matchCount:null,outputSize:0,readbackSize:0,candidateSize:0,groupCountSize:0,groupOffsetSize:0,matchCountSize:0})),e=a=>Math.ceil(a/256)*256;return{get slotCount(){return s.length},acquire:(a,d)=>{if(a<0||a>=s.length)throw new Error(`buffer slot ${a} is out of range`);if(!Number.isFinite(d)||d<=0)throw new Error("messageCount must be a positive integer");const u=s[a],y=d,b=e(Oe+y*le),w=Math.max(1,Math.ceil(d/i)),C=w*i,M=e(C*le),O=e(w*Uint32Array.BYTES_PER_ELEMENT),U=e(Oe);return(!u.output||b>u.outputSize)&&(u.output?.destroy(),u.output=t.createBuffer({label:`gpu-seed-output-${a}`,size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),u.outputSize=b),(!u.readback||b>u.readbackSize)&&(u.readback?.destroy(),u.readback=t.createBuffer({label:`gpu-seed-readback-${a}`,size:b,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.readbackSize=b),(!u.candidate||M>u.candidateSize)&&(u.candidate?.destroy(),u.candidate=t.createBuffer({label:`gpu-seed-candidate-${a}`,size:M,usage:GPUBufferUsage.STORAGE}),u.candidateSize=M),(!u.groupCounts||O>u.groupCountSize)&&(u.groupCounts?.destroy(),u.groupCounts=t.createBuffer({label:`gpu-seed-group-counts-${a}`,size:O,usage:GPUBufferUsage.STORAGE}),u.groupCountSize=O),(!u.groupOffsets||O>u.groupOffsetSize)&&(u.groupOffsets?.destroy(),u.groupOffsets=t.createBuffer({label:`gpu-seed-group-offsets-${a}`,size:O,usage:GPUBufferUsage.STORAGE}),u.groupOffsetSize=O),(!u.matchCount||U>u.matchCountSize)&&(u.matchCount?.destroy(),u.matchCount=t.createBuffer({label:`gpu-seed-match-header-${a}`,size:U,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.matchCountSize=U),{output:u.output,readback:u.readback,candidate:u.candidate,groupCounts:u.groupCounts,groupOffsets:u.groupOffsets,matchCount:u.matchCount,outputSize:u.outputSize,candidateCapacity:C,groupCount:w,maxRecords:y}},dispose:()=>{for(const a of s)a.output?.destroy(),a.readback?.destroy(),a.candidate?.destroy(),a.groupCounts?.destroy(),a.groupOffsets?.destroy(),a.matchCount?.destroy(),a.output=null,a.readback=null,a.candidate=null,a.groupCounts=null,a.groupOffsets=null,a.matchCount=null,a.outputSize=0,a.readbackSize=0,a.candidateSize=0,a.groupCountSize=0,a.groupOffsetSize=0,a.matchCountSize=0}}}function lt(t,r){const n=r?.hostMemoryLimitBytes??De,i=r?.bufferSetCount??Se,s=(()=>{const a=r?.hostMemoryLimitPerSlot;if(typeof a=="number"&&Number.isFinite(a)&&a>0)return a;const d=Math.floor(n/i);return Math.max(1,d)})(),e=r?.workgroupSize??pe;if(n<=0)throw new Error("host memory limit must be positive");if(i<=0)throw new Error("buffer set count must be positive");const c=a=>{const u=t.getDevice().limits,y=Math.max(1,u.maxStorageBufferBindingSize??le),b=Math.max(1,Math.floor(y/le)),w=Math.max(1,Math.floor(s/le)),C=t.getSupportedWorkgroupSize(e),M=u.maxComputeWorkgroupsPerDimension??65535,O=Math.max(1,C*M),U=Math.min(b,w,O);return a<=U?a<=1?1:Math.max(1,Math.min(U,Math.ceil(a/2))):U};return{computePlan:a=>{if(!Number.isFinite(a)||a<0)throw new Error("totalMessages must be a non-negative finite value");if(a===0)return{maxMessagesPerDispatch:0,dispatches:[]};const d=c(a),u=[];let y=a,b=0;for(;y>0;){const w=Math.min(d,y);u.push({baseOffset:b,messageCount:w}),b+=w,y-=w}if(u.length===1&&a>1){const w=u[0],C=Math.ceil(w.messageCount/2),M=w.messageCount-C;M>0&&(u[0]={baseOffset:w.baseOffset,messageCount:C},u.push({baseOffset:w.baseOffset+C,messageCount:M}))}return{maxMessagesPerDispatch:d,dispatches:u}}}}const dt=25,ft=500,gt=1024,be=new Uint32Array([0]),pt=ze;function mt(t){const r=o=>Se,n=De,i=r(),s=(()=>{const o=Math.floor(n/i);return Math.max(1,o)})(),e={workgroupSize:pe,bufferSlotCount:i,hostMemoryLimitBytes:n,hostMemoryLimitPerSlotBytes:s,deviceContext:null,pipelines:null,bindGroupLayouts:null,configBuffer:null,configData:null,bufferPool:null,planner:null,targetBuffer:null,targetBufferCapacity:0,seedCalculator:new nt,isRunning:!1,isPaused:!1,shouldStop:!1,lastProgressUpdateMs:0,timerState:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1}},c=async(o,m,_)=>(async()=>await Promise.resolve(_()))(),f=async()=>{if(e.pipelines&&e.bufferPool&&e.planner&&e.deviceContext)return;const o=await ot(),m=o.getDevice(),_=o.getSupportedWorkgroupSize(e.workgroupSize),{pipelines:S,layouts:R}=it(m,_),P=new Uint32Array(dt),T=te(P.byteLength),g=m.createBuffer({label:"gpu-seed-config-buffer",size:T,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),E=ct(m,{slots:e.bufferSlotCount,workgroupSize:_}),v=lt(o,{workgroupSize:_,bufferSetCount:e.bufferSlotCount,hostMemoryLimitBytes:e.hostMemoryLimitBytes,hostMemoryLimitPerSlot:e.hostMemoryLimitPerSlotBytes});e.deviceContext=o,e.pipelines=S,e.bindGroupLayouts=R,e.configBuffer=g,e.configData=P,e.bufferPool=E,e.planner=v,e.workgroupSize=_},a=o=>{if(!e.deviceContext)throw new Error("WebGPU device is not initialized");const m=e.deviceContext.getDevice(),_=o.length,S=1+_,R=te(S*Uint32Array.BYTES_PER_ELEMENT),P=e.targetBufferCapacity;if(!e.targetBuffer||P<_){e.targetBuffer?.destroy(),e.targetBuffer=m.createBuffer({label:"gpu-seed-target-buffer",size:R,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const v=Math.floor(R/Uint32Array.BYTES_PER_ELEMENT)-1;e.targetBufferCapacity=Math.max(v,_)}const g=new Uint32Array(S);g[0]=_>>>0;for(let v=0;v<_;v+=1)g[1+v]=o[v]>>>0;const E=g.byteLength;m.queue.writeBuffer(e.targetBuffer,0,g.buffer,g.byteOffset,E)},d=async o=>{if(e.isRunning)throw new Error("WebGPU search is already running");if((!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)&&await f(),!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)throw new Error("WebGPU runner failed to initialize");const{context:m,targetSeeds:_,callbacks:S,signal:R}=o;if(m.totalMessages===0){S.onComplete("探索対象の組み合わせが存在しません");return}if(!e.bindGroupLayouts)throw new Error("WebGPU runner missing bind group layout");a(_),e.isRunning=!0,e.isPaused=!1,e.shouldStop=!1,e.lastProgressUpdateMs=Date.now();const P={currentStep:0,totalSteps:m.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0};let T;if(R)if(R.aborted)e.shouldStop=!0;else{const g=()=>{e.shouldStop=!0};R.addEventListener("abort",g),T=()=>R.removeEventListener("abort",g)}j();try{await C(m,P,S);const g=V(),E={...P,elapsedTime:g,estimatedTimeRemaining:0};e.shouldStop?S.onStopped("検索を停止しました",E):(S.onProgress(E),S.onComplete(`検索が完了しました。${P.matchesFound}件ヒットしました。`))}catch(g){const E=g instanceof Error?g.message:"WebGPU検索中に不明なエラーが発生しました",v=g instanceof GPUValidationError?"WEBGPU_VALIDATION_ERROR":void 0;throw S.onError(E,v),g}finally{e.isRunning=!1,e.isPaused=!1,e.shouldStop=!1,F(),T&&T()}},u=()=>{!e.isRunning||e.isPaused||(e.isPaused=!0,F())},y=()=>{!e.isRunning||!e.isPaused||(e.isPaused=!1,x())},b=()=>{e.isRunning&&(e.shouldStop=!0,e.isPaused=!1,x())},w=()=>{e.bufferPool?.dispose(),e.configBuffer?.destroy(),e.configBuffer=null,e.configData=null,e.pipelines=null,e.bindGroupLayouts=null,e.bufferPool=null,e.planner=null,e.deviceContext=null,e.targetBuffer?.destroy(),e.targetBuffer=null,e.targetBufferCapacity=0},C=async(o,m,_)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const R=e.deviceContext.getDevice().queue,P=e.bufferPool.slotCount,T=Array.from({length:P},(l,G)=>P-1-G),g=[],E=[],v=[],N=l=>{v.push(l)},z=()=>new Promise(l=>{if(T.length>0){const G=T.pop();l(G);return}g.push(l)}),A=l=>{const G=g.shift();if(G){G(l);return}T.push(l)};let h=0;for(const l of o.segments){if(e.shouldStop)break;const G=await c("planner.computePlan",{segmentIndex:l.index,totalMessages:l.totalMessages},()=>Promise.resolve(e.planner.computePlan(l.totalMessages)));for(const I of G.dispatches){if(e.shouldStop||(await $(),e.shouldStop))break;const k=await z();if(e.shouldStop){A(k);break}const q={segment:l,dispatchIndex:h,messageCount:I.messageCount,slotIndex:k},H=M(q,I.baseOffset,o,m,_,R,A,N);E.push(H),h+=1}}E.length>0&&await Promise.all(E),v.length>0&&await Promise.all(v)},M=async(o,m,_,S,R,P,T,g)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const E=e.deviceContext.getDevice(),v=e.configBuffer,N=e.configData,z=e.bindGroupLayouts,A=e.pipelines,h=e.targetBuffer,l=e.bufferPool.acquire(o.slotIndex,o.messageCount);let G=!1,I=!1;const k=()=>{G||(G=!0,T(o.slotIndex))},q=Math.ceil(o.messageCount/e.workgroupSize),H=Math.max(1,Math.ceil(l.candidateCapacity/e.workgroupSize)),Y=te(oe*Uint32Array.BYTES_PER_ELEMENT),L={dispatchIndex:o.dispatchIndex,messageCount:o.messageCount,slotIndex:o.slotIndex,workgroupCount:q,scatterWorkgroupCount:H,candidateCapacity:l.candidateCapacity,groupCount:l.groupCount,segmentIndex:o.segment.index,segmentBaseOffset:m};try{await c("dispatch",L,async()=>{P.writeBuffer(l.output,0,be.buffer,be.byteOffset,be.byteLength),W(o.segment,m,o.messageCount,l.groupCount,l.candidateCapacity),P.writeBuffer(v,0,N.buffer,N.byteOffset,N.byteLength);const re=E.createBindGroup({label:`gpu-seed-generate-group-${o.dispatchIndex}`,layout:z.generate,entries:[{binding:0,resource:{buffer:v}},{binding:1,resource:{buffer:h}},{binding:2,resource:{buffer:l.candidate}},{binding:3,resource:{buffer:l.groupCounts}}]}),de=E.createBindGroup({label:`gpu-seed-scan-group-${o.dispatchIndex}`,layout:z.scan,entries:[{binding:0,resource:{buffer:v}},{binding:3,resource:{buffer:l.groupCounts}},{binding:4,resource:{buffer:l.groupOffsets}},{binding:5,resource:{buffer:l.output}}]}),_e=E.createBindGroup({label:`gpu-seed-scatter-group-${o.dispatchIndex}`,layout:z.scatter,entries:[{binding:0,resource:{buffer:v}},{binding:2,resource:{buffer:l.candidate}},{binding:3,resource:{buffer:l.groupCounts}},{binding:4,resource:{buffer:l.groupOffsets}},{binding:5,resource:{buffer:l.output}}]}),K=E.createCommandEncoder({label:`gpu-seed-compute-${o.dispatchIndex}`}),J=K.beginComputePass({label:`gpu-seed-generate-pass-${o.dispatchIndex}`});J.setPipeline(A.generate),J.setBindGroup(0,re),J.dispatchWorkgroups(q),J.end();const X=K.beginComputePass({label:`gpu-seed-scan-pass-${o.dispatchIndex}`});X.setPipeline(A.scan),X.setBindGroup(0,de),X.dispatchWorkgroups(1),X.end();const Q=K.beginComputePass({label:`gpu-seed-scatter-pass-${o.dispatchIndex}`});Q.setPipeline(A.scatter),Q.setBindGroup(0,_e),Q.dispatchWorkgroups(H),Q.end(),K.copyBufferToBuffer(l.output,0,l.matchCount,0,Y);const he=K.finish();await c("dispatch.submit",{...L},async()=>{await c("dispatch.submit.encode",{...L},async()=>{P.submit([he])})});const fe=await c("dispatch.mapMatchCount",{...L,headerCopyBytes:Y},async()=>{await l.matchCount.mapAsync(GPUMapMode.READ,0,Y);const ae=new Uint32Array(l.matchCount.getMappedRange(0,Y))[0]??0;return l.matchCount.unmap(),ae}),Ae=Math.min(fe,l.maxRecords)*ce*Uint32Array.BYTES_PER_ELEMENT,ne=te(oe*Uint32Array.BYTES_PER_ELEMENT+Ae);await c("dispatch.copyResults",{...L,totalCopyBytes:ne},async()=>{const ee=E.createCommandEncoder({label:`gpu-seed-copy-${o.dispatchIndex}`});ee.copyBufferToBuffer(l.output,0,l.readback,0,ne);const ae=ee.finish();await c("dispatch.copyResults.encode",{...L,totalCopyBytes:ne},async()=>{P.submit([ae])})});const ke=(async()=>{try{const{results:ee,clampedMatchCount:ae}=await c("dispatch.mapResults",{...L,totalCopyBytes:ne},async()=>{await l.readback.mapAsync(GPUMapMode.READ,0,ne);const Le=l.readback.getMappedRange(0,ne),ye=new Uint32Array(Le),We=ye[0]??0,Fe=Math.max(0,Math.floor((ye.length-oe)/ce)),ve=Math.min(We,l.maxRecords,Fe),Pe=oe+ve*ce,Ee=new Uint32Array(Pe);return Ee.set(ye.subarray(0,Pe)),l.readback.unmap(),{results:Ee,clampedMatchCount:ve}});try{k(),await c("dispatch.processMatches",{...L,matchCount:ae},()=>O(ee,ae,o,m,_,S,R))}finally{k()}}catch(ee){throw k(),ee}})();I=!0,g(ke)})}finally{I||k()}},O=async(o,m,_,S,R,P,T)=>{const g=_.segment,E=g.rangeSeconds,v=Math.max(E,1),N=Math.max(g.config.vcountCount,1),z=v,A=z*N,h=g.config.timer0Min,l=g.config.vcountMin,G=S;for(let I=0;I<m&&!(e.shouldStop||I%gt===0&&(await $(),e.shouldStop));I+=1){const k=oe+I*ce,q=o[k],H=G+q,Y=o[k+1]>>>0,L=Math.floor(H/A),re=H-L*A,de=Math.floor(re/z),_e=re-de*z,K=h+L,J=l+de,X=new Date(R.startTimestampMs+_e*1e3),Q=e.seedCalculator.generateMessage(R.conditions,K,J,X),{hash:he,seed:fe}=e.seedCalculator.calculateSeed(Q);fe!==Y&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:Y,cpuSeed:fe,messageIndex:H});const Re={seed:Y,datetime:X,timer0:K,vcount:J,conditions:R.conditions,message:Q,sha1Hash:he,isMatch:!0};T.onResult(Re),P.matchesFound+=1}if(_.messageCount>0){const I=_.messageCount-1,k=G+I,q=Math.floor(k/A),H=k-q*A,Y=Math.floor(H/z),L=H-Y*z,re=new Date(R.startTimestampMs+L*1e3).toISOString();P.currentDateTime=re}P.currentStep+=_.messageCount,U(P,T)},U=(o,m)=>{const _=Date.now();if(_-e.lastProgressUpdateMs<ft&&o.currentStep<o.totalSteps)return;const S=V(),R=p(o.currentStep,o.totalSteps,S);m.onProgress({currentStep:o.currentStep,totalSteps:o.totalSteps,elapsedTime:S,estimatedTimeRemaining:R,matchesFound:o.matchesFound,currentDateTime:o.currentDateTime}),e.lastProgressUpdateMs=_},p=(o,m,_)=>{if(o===0||o>=m)return 0;const S=_/o,R=m-o;return Math.round(S*R)},W=(o,m,_,S,R)=>{if(!e.configData)throw new Error("config buffer not prepared");const P=Math.max(o.config.rangeSeconds,1),T=Math.max(o.config.vcountCount,1),g=P,E=g*T,v=Math.floor(m/E),N=m-v*E,z=Math.floor(N/g),A=N-z*g,h=e.configData;h[0]=_>>>0,h[1]=v>>>0,h[2]=z>>>0,h[3]=A>>>0,h[4]=o.config.rangeSeconds>>>0,h[5]=o.config.timer0Min>>>0,h[6]=o.config.timer0Count>>>0,h[7]=o.config.vcountMin>>>0,h[8]=o.config.vcountCount>>>0,h[9]=o.config.startSecondOfDay>>>0,h[10]=o.config.startDayOfWeek>>>0,h[11]=o.config.macLower>>>0,h[12]=o.config.data7Swapped>>>0,h[13]=o.config.keyInputSwapped>>>0,h[14]=o.config.hardwareType>>>0;for(let l=0;l<o.config.nazoSwapped.length;l+=1)h[15+l]=o.config.nazoSwapped[l]>>>0;h[20]=o.config.startYear>>>0,h[21]=o.config.startDayOfYear>>>0,h[22]=S>>>0,h[23]=e.workgroupSize>>>0,h[24]=R>>>0},j=()=>{e.timerState.cumulativeRunTime=0,e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1},F=()=>{e.timerState.isPaused||(e.timerState.cumulativeRunTime+=Date.now()-e.timerState.segmentStartTime,e.timerState.isPaused=!0)},x=()=>{e.timerState.isPaused&&(e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1)},V=()=>e.timerState.isPaused?e.timerState.cumulativeRunTime:e.timerState.cumulativeRunTime+(Date.now()-e.timerState.segmentStartTime),$=async()=>{for(;e.isPaused&&!e.shouldStop;)await se(25)},se=o=>new Promise(m=>setTimeout(m,o)),te=o=>Math.ceil(o/256)*256;return{init:f,run:d,pause:u,resume:y,stop:b,dispose:w}}const Ge=self,D={isRunning:!1,isPaused:!1},me=mt();let ge=null;function B(t){Ge.postMessage(t)}function _t(){B({type:"READY",message:"WebGPU worker initialized"})}function ie(){D.isRunning=!1,D.isPaused=!1,ge=null}function ht(){return pt()?!0:(B({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function yt(t){if(D.isRunning){B({type:"ERROR",error:"Search is already running"});return}if(!t.conditions||!t.targetSeeds){B({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!ht())return;D.isRunning=!0,D.isPaused=!1;let r;try{r=Ye(t.conditions)}catch(s){ie();const e=s instanceof Error?s.message:"検索条件の解析中にエラーが発生しました";B({type:"ERROR",error:e,errorCode:"WEBGPU_CONTEXT_ERROR"});return}ge=new AbortController;const n={onProgress:s=>{B({type:"PROGRESS",progress:s})},onResult:s=>{B({type:"RESULT",result:s})},onComplete:s=>{ie(),B({type:"COMPLETE",message:s})},onError:(s,e)=>{ie(),B({type:"ERROR",error:s,errorCode:e})},onPaused:()=>{D.isPaused=!0,B({type:"PAUSED"})},onResumed:()=>{D.isPaused=!1,B({type:"RESUMED"})},onStopped:(s,e)=>{ie(),B({type:"STOPPED",message:s,progress:e})}},i={context:r,targetSeeds:t.targetSeeds,callbacks:n,signal:ge.signal};try{await me.run(i)}catch(s){if(!D.isRunning)return;ie();const e=s instanceof Error?s.message:"WebGPU search failed with unknown error";B({type:"ERROR",error:e,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function bt(){!D.isRunning||D.isPaused||(me.pause(),D.isPaused=!0,B({type:"PAUSED"}))}function wt(){!D.isRunning||!D.isPaused||(me.resume(),D.isPaused=!1,B({type:"RESUMED"}))}function St(){D.isRunning&&(me.stop(),ge?.abort())}_t();Ge.onmessage=t=>{const r=t.data;switch(r.type){case"START_SEARCH":yt(r);break;case"PAUSE_SEARCH":bt();break;case"RESUME_SEARCH":wt();break;case"STOP_SEARCH":St();break;default:B({type:"ERROR",error:`Unknown request type: ${r.type}`})}};
