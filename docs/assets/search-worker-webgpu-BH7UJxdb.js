const be={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},ve=Date.UTC(2e3,0,1,0,0,0),Ee=100663296,Pe={DS:8,DS_LITE:6,"3DS":9};function Ce(r){const e=fe(r.dateRange.startYear,r.dateRange.startMonth,r.dateRange.startDay,r.dateRange.startHour,r.dateRange.startMinute,r.dateRange.startSecond),t=fe(r.dateRange.endYear,r.dateRange.endMonth,r.dateRange.endDay,r.dateRange.endHour,r.dateRange.endMinute,r.dateRange.endSecond);if(e.getTime()>t.getTime())throw new Error("開始日時が終了日時より後ろに設定されています");const c=xe(r),o=Me(r,c),u=ge(r.dateRange.startYear,r.dateRange.startMonth,r.dateRange.startDay,r.dateRange.startHour,r.dateRange.startMinute,r.dateRange.startSecond),l=ge(r.dateRange.endYear,r.dateRange.endMonth,r.dateRange.endDay,r.dateRange.endHour,r.dateRange.endMinute,r.dateRange.endSecond),f=Math.floor((u-ve)/1e3);if(f<0)throw new Error("2000年より前の日時は指定できません");const a=Math.floor((l-u)/1e3)+1;if(a<=0)throw new Error("探索秒数が0秒以下です");const d=e.getFullYear(),s=Ge(e),b=ze(e),S=e.getDay(),R=Pe[r.hardware],{macLower:P,data7Swapped:x}=Oe(r.macAddress,R),M=de(r.keyInput>>>0),T=Ue(c.nazo),_=[];let G=0;for(let D=0;D<o.length;D+=1){const n=o[D],g=n.timer0Max-n.timer0Min+1,p=a*g,h={startSecondsSince2000:f>>>0,rangeSeconds:a>>>0,timer0Min:n.timer0Min>>>0,timer0Max:n.timer0Max>>>0,timer0Count:g>>>0,vcountMin:n.vcount>>>0,vcountMax:n.vcount>>>0,vcountCount:1,totalMessages:p>>>0,hardwareType:De(r.hardware),macLower:P>>>0,data7Swapped:x>>>0,keyInputSwapped:M>>>0,nazoSwapped:T,startYear:d>>>0,startDayOfYear:s>>>0,startSecondOfDay:b>>>0,startDayOfWeek:S>>>0};_.push({index:D,baseOffset:G,timer0Min:n.timer0Min,timer0Max:n.timer0Max,timer0Count:g,vcount:n.vcount,rangeSeconds:a,totalMessages:p,config:h}),G+=p}const N=_.reduce((D,n)=>D+n.totalMessages,0);return{conditions:r,startDate:e,startTimestampMs:e.getTime(),rangeSeconds:a,totalMessages:N,segments:_}}function fe(r,e,t,c,o,u){return new Date(r,e-1,t,c,o,u)}function ge(r,e,t,c,o,u){return Date.UTC(r,e-1,t,c,o,u,0)}function xe(r){const e=be[r.romVersion];if(!e)throw new Error(`ROMバージョン ${r.romVersion} は未対応です`);const t=e[r.romRegion];if(!t)throw new Error(`ROMリージョン ${r.romRegion} は未対応です`);return{nazo:[...t.nazo],vcountTimerRanges:t.vcountTimerRanges.map(c=>[...c])}}function Me(r,e){const t=[],c=r.timer0VCountConfig.timer0Range.min,o=r.timer0VCountConfig.timer0Range.max;let u=null;for(let l=c;l<=o;l+=1){const f=Te(e,l);u&&u.vcount===f&&l===u.timer0Max+1?u.timer0Max=l:(u&&t.push(u),u={timer0Min:l,timer0Max:l,vcount:f})}return u&&t.push(u),t}function Te(r,e){for(const[t,c,o]of r.vcountTimerRanges)if(e>=c&&e<=o)return t;return r.vcountTimerRanges.length>0?r.vcountTimerRanges[0][0]:96}function Oe(r,e){const t=Be(r),c=(t[4]&255)<<8|t[5]&255,u=((t[0]&255|(t[1]&255)<<8|(t[2]&255)<<16|(t[3]&255)<<24)^Ee^e)>>>0;return{macLower:c,data7Swapped:de(u)}}function Be(r){const e=new Array(6).fill(0);for(let t=0;t<6;t+=1){const c=r[t]??0;e[t]=(Number(c)&255)>>>0}return e}function De(r){switch(r){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function de(r){return((r&255)<<24|(r>>>8&255)<<16|(r>>>16&255)<<8|r>>>24&255)>>>0}function Ue(r){const e=new Uint32Array(r.length);for(let t=0;t<r.length;t+=1)e[t]=de(r[t]>>>0);return e}function Ge(r){const e=new Date(r.getFullYear(),0,1),t=r.getTime()-e.getTime();return Math.floor(t/(1440*60*1e3))+1}function ze(r){return r.getHours()*3600+r.getMinutes()*60+r.getSeconds()}class pe{calculateHash(e){if(e.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const t=1732584193,c=4023233417,o=2562383102,u=271733878,l=3285377520,f=new Array(80);for(let _=0;_<16;_++)f[_]=e[_];for(let _=16;_<80;_++)f[_]=this.leftRotate(f[_-3]^f[_-8]^f[_-14]^f[_-16],1);let a=t,d=c,s=o,b=u,S=l;for(let _=0;_<80;_++){let G;_<20?G=this.leftRotate(a,5)+(d&s|~d&b)+S+f[_]+1518500249&4294967295:_<40?G=this.leftRotate(a,5)+(d^s^b)+S+f[_]+1859775393&4294967295:_<60?G=this.leftRotate(a,5)+(d&s|d&b|s&b)+S+f[_]+2400959708&4294967295:G=this.leftRotate(a,5)+(d^s^b)+S+f[_]+3395469782&4294967295,S=b,b=s,s=this.leftRotate(d,30),d=a,a=G}const R=this.add32(t,a),P=this.add32(c,d),x=this.add32(o,s),M=this.add32(u,b),T=this.add32(l,S);return{h0:R,h1:P,h2:x,h3:M,h4:T}}leftRotate(e,t){return(e<<t|e>>>32-t)>>>0}add32(e,t){return(e+t&4294967295)>>>0}static hashToHex(e,t,c,o,u){return e.toString(16).padStart(8,"0")+t.toString(16).padStart(8,"0")+c.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")+u.toString(16).padStart(8,"0")}}let H=null,X=null;async function Ae(){return H||X||(X=(async()=>{try{const r=await import("./wasm_pkg-HLQ4E-jy.js");let e;if(typeof process<"u"&&!!process.versions?.node){const c=await import("./__vite-browser-external-9wXp6ZBx.js"),u=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");e={module_or_path:c.readFileSync(u)}}else e={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-CDMG1ChF.wasm",import.meta.url)};return await r.default(e),H={IntegratedSeedSearcher:r.IntegratedSeedSearcher,BWGenerationConfig:r.BWGenerationConfig,PokemonGenerator:r.PokemonGenerator,SeedEnumerator:r.SeedEnumerator,EncounterType:r.EncounterType,GameVersion:r.GameVersion,GameMode:r.GameMode,calculate_game_offset:r.calculate_game_offset,sha1_hash_batch:r.sha1_hash_batch},H}catch(r){throw console.error("Failed to load WebAssembly module:",r),H=null,X=null,r}})(),X)}function ke(){if(!H)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return H}function _e(){return H!==null}const Le={DS:8,DS_LITE:6,"3DS":9};class We{sha1;useWasm=!1;constructor(){this.sha1=new pe}async initializeWasm(){try{return await Ae(),this.useWasm=!0,!0}catch(e){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",e),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&_e()}getWasmModule(){return ke()}setUseWasm(e){if(e&&!_e()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=e}getROMParameters(e,t){const c=be[e];if(!c)return console.error(`ROM version not found: ${e}`),null;const o=c[t];return o?{nazo:[...o.nazo],vcountTimerRanges:o.vcountTimerRanges.map(u=>[...u])}:(console.error(`ROM region not found: ${t} for version ${e}`),null)}toLittleEndian32(e){return((e&255)<<24|(e>>>8&255)<<16|(e>>>16&255)<<8|e>>>24&255)>>>0}toLittleEndian16(e){return(e&255)<<8|e>>>8&255}getDayOfWeek(e,t,c){return new Date(e,t-1,c).getDay()}generateMessage(e,t,c,o){const u=this.getROMParameters(e.romVersion,e.romRegion);if(!u)throw new Error(`No parameters found for ${e.romVersion} ${e.romRegion}`);const l=new Array(16).fill(0);for(let m=0;m<5;m++)l[m]=this.toLittleEndian32(u.nazo[m]);l[5]=this.toLittleEndian32(c<<16|t);const f=e.macAddress[4]<<8|e.macAddress[5];l[6]=f;const a=e.macAddress[0]<<0|e.macAddress[1]<<8|e.macAddress[2]<<16|e.macAddress[3]<<24,d=100663296,s=Le[e.hardware],b=a^d^s;l[7]=this.toLittleEndian32(b);const S=o.getFullYear()%100,R=o.getMonth()+1,P=o.getDate(),x=this.getDayOfWeek(o.getFullYear(),R,P),M=Math.floor(S/10)*16+S%10,T=Math.floor(R/10)*16+R%10,_=Math.floor(P/10)*16+P%10,G=Math.floor(x/10)*16+x%10;l[8]=M<<24|T<<16|_<<8|G;const N=o.getHours(),D=o.getMinutes(),n=o.getSeconds(),g=(e.hardware==="DS"||e.hardware==="DS_LITE")&&N>=12?1:0,p=Math.floor(N/10)*16+N%10,h=Math.floor(D/10)*16+D%10,y=Math.floor(n/10)*16+n%10;return l[9]=g<<30|p<<24|h<<16|y<<8|0,l[10]=0,l[11]=0,l[12]=this.toLittleEndian32(e.keyInput),l[13]=2147483648,l[14]=0,l[15]=416,l}calculateSeed(e){const t=this.sha1.calculateHash(e),c=BigInt(this.toLittleEndian32(t.h0)),a=(BigInt(this.toLittleEndian32(t.h1))<<32n|c)*0x5D588B656C078965n+0x269EC3n;return{seed:Number(a>>32n&0xFFFFFFFFn),hash:pe.hashToHex(t.h0,t.h1,t.h2,t.h3,t.h4)}}parseTargetSeeds(e){const t=e.split(`
`).map(l=>l.trim()).filter(l=>l.length>0),c=[],o=[],u=new Set;return t.forEach((l,f)=>{try{let a=l.toLowerCase();if(a.startsWith("0x")&&(a=a.substring(2)),!/^[0-9a-f]{1,8}$/.test(a)){o.push({line:f+1,value:l,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const d=parseInt(a,16);if(u.has(d))return;u.add(d),c.push(d)}catch(a){const d=a instanceof Error?a.message:String(a);o.push({line:f+1,value:l,error:d||"Failed to parse as hexadecimal number."})}}),{validSeeds:c,errors:o}}getVCountForTimer0(e,t){for(const[c,o,u]of e.vcountTimerRanges)if(t>=o&&t<=u)return c;return e.vcountTimerRanges.length>0?e.vcountTimerRanges[0][0]:96}}const ye=Uint32Array.BYTES_PER_ELEMENT,ee=2,re=ee*ye,K=1,me=K*ye,ue=128,Fe=96*1024*1024,he=2,Ie={requiredFeatures:[]};function Se(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function He(r){if(!Se())throw new Error("WebGPU is not available in this environment");const t=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!t)throw new Error("Failed to acquire WebGPU adapter");const c={requiredFeatures:Ie.requiredFeatures,requiredLimits:r?.requiredLimits,label:"seed-search-device"},o=await t.requestDevice(c);let u=!1;const l=o.lost.then(f=>(u=!0,console.warn("[webgpu] device lost:",f.message),f));return{getAdapter:()=>t,getDevice:()=>o,getQueue:()=>o.queue,getLimits:()=>o.limits,isLost:()=>u,waitForLoss:()=>l,getSupportedWorkgroupSize:(f=ue)=>{const a=o.limits,d=a.maxComputeInvocationsPerWorkgroup??f,s=a.maxComputeWorkgroupSizeX??f,b=Math.min(f,d,s);if(b<=0)throw new Error("WebGPU workgroup size limits are invalid");return b}}}var Ne=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
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
`;function Ye(r){return Ne.replace(/WORKGROUP_SIZE_PLACEHOLDER/g,String(r))}function Ve(r,e){const t=r.createShaderModule({label:"gpu-seed-sha1-generated-module",code:Ye(e)});t.getCompilationInfo?.().then(d=>{d.messages.length>0&&console.warn("[pipeline-factory] compilation diagnostics",d.messages.map(s=>({message:s.message,line:s.lineNum,column:s.linePos,type:s.type})))}).catch(d=>{console.warn("[pipeline-factory] compilation info failed",d)});const c=r.createBindGroupLayout({label:"gpu-seed-generate-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),o=r.createBindGroupLayout({label:"gpu-seed-scan-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),u=r.createBindGroupLayout({label:"gpu-seed-scatter-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),l=r.createComputePipeline({label:"gpu-seed-generate-pipeline",layout:r.createPipelineLayout({label:"gpu-seed-generate-pipeline-layout",bindGroupLayouts:[c]}),compute:{module:t,entryPoint:"sha1_generate"}}),f=r.createComputePipeline({label:"gpu-seed-scan-pipeline",layout:r.createPipelineLayout({label:"gpu-seed-scan-pipeline-layout",bindGroupLayouts:[o]}),compute:{module:t,entryPoint:"exclusive_scan_groups"}}),a=r.createComputePipeline({label:"gpu-seed-scatter-pipeline",layout:r.createPipelineLayout({label:"gpu-seed-scatter-pipeline-layout",bindGroupLayouts:[u]}),compute:{module:t,entryPoint:"scatter_matches"}});return{pipelines:{generate:l,scan:f,scatter:a},layouts:{generate:c,scan:o,scatter:u}}}function $e(r,e){const t=e?.slots,c=e?.workgroupSize??ue,o=Array.from({length:t},()=>({output:null,readback:null,candidate:null,groupCounts:null,groupOffsets:null,matchCount:null,outputSize:0,readbackSize:0,candidateSize:0,groupCountSize:0,groupOffsetSize:0,matchCountSize:0})),u=a=>Math.ceil(a/256)*256;return{get slotCount(){return o.length},acquire:(a,d)=>{if(a<0||a>=o.length)throw new Error(`buffer slot ${a} is out of range`);if(!Number.isFinite(d)||d<=0)throw new Error("messageCount must be a positive integer");const s=o[a],b=d,S=u(me+b*re),R=Math.max(1,Math.ceil(d/c)),P=R*c,x=u(P*re),M=u(R*Uint32Array.BYTES_PER_ELEMENT),T=u(me);return(!s.output||S>s.outputSize)&&(s.output?.destroy(),s.output=r.createBuffer({label:`gpu-seed-output-${a}`,size:S,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),s.outputSize=S),(!s.readback||S>s.readbackSize)&&(s.readback?.destroy(),s.readback=r.createBuffer({label:`gpu-seed-readback-${a}`,size:S,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),s.readbackSize=S),(!s.candidate||x>s.candidateSize)&&(s.candidate?.destroy(),s.candidate=r.createBuffer({label:`gpu-seed-candidate-${a}`,size:x,usage:GPUBufferUsage.STORAGE}),s.candidateSize=x),(!s.groupCounts||M>s.groupCountSize)&&(s.groupCounts?.destroy(),s.groupCounts=r.createBuffer({label:`gpu-seed-group-counts-${a}`,size:M,usage:GPUBufferUsage.STORAGE}),s.groupCountSize=M),(!s.groupOffsets||M>s.groupOffsetSize)&&(s.groupOffsets?.destroy(),s.groupOffsets=r.createBuffer({label:`gpu-seed-group-offsets-${a}`,size:M,usage:GPUBufferUsage.STORAGE}),s.groupOffsetSize=M),(!s.matchCount||T>s.matchCountSize)&&(s.matchCount?.destroy(),s.matchCount=r.createBuffer({label:`gpu-seed-match-header-${a}`,size:T,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),s.matchCountSize=T),{output:s.output,readback:s.readback,candidate:s.candidate,groupCounts:s.groupCounts,groupOffsets:s.groupOffsets,matchCount:s.matchCount,outputSize:s.outputSize,candidateCapacity:P,groupCount:R,maxRecords:b}},dispose:()=>{for(const a of o)a.output?.destroy(),a.readback?.destroy(),a.candidate?.destroy(),a.groupCounts?.destroy(),a.groupOffsets?.destroy(),a.matchCount?.destroy(),a.output=null,a.readback=null,a.candidate=null,a.groupCounts=null,a.groupOffsets=null,a.matchCount=null,a.outputSize=0,a.readbackSize=0,a.candidateSize=0,a.groupCountSize=0,a.groupOffsetSize=0,a.matchCountSize=0}}}function qe(r,e){const t=e?.hostMemoryLimitBytes??Fe,c=e?.bufferSetCount,o=e?.workgroupSize??ue,u=typeof e?.maxMessagesOverride=="number"?Math.max(1,Math.floor(e.maxMessagesOverride)):null;if(t<=0)throw new Error("host memory limit must be positive");const l=a=>{if(u!==null)return Math.min(u,a||1);const s=r.getDevice().limits,b=Math.max(1,s.maxStorageBufferBindingSize??re),S=Math.max(1,Math.floor(b/re)),R=Math.max(1,Math.floor(t/(re*c))),P=r.getSupportedWorkgroupSize(o),x=s.maxComputeWorkgroupsPerDimension??65535,M=Math.max(1,P*x),T=Math.min(S,R,M);return a<=T?a<=1?1:Math.max(1,Math.min(T,Math.ceil(a/2))):T};return{computePlan:a=>{if(!Number.isFinite(a)||a<0)throw new Error("totalMessages must be a non-negative finite value");if(a===0)return{maxMessagesPerDispatch:0,dispatches:[]};const d=l(a),s=[];let b=a,S=0;for(;b>0;){const R=Math.min(d,b);s.push({baseOffset:S,messageCount:R}),S+=R,b-=R}if(s.length===1&&a>1){const R=s[0],P=Math.ceil(R.messageCount/2),x=R.messageCount-P;x>0&&(s[0]={baseOffset:R.baseOffset,messageCount:P},s.push({baseOffset:R.baseOffset+P,messageCount:x}))}return{maxMessagesPerDispatch:d,dispatches:s}}}}const Ke=25,Ze=500,je=1024,le=new Uint32Array([0]),Je=Se;function Xe(r){const e={workgroupSize:ue,deviceContext:null,pipelines:null,bindGroupLayouts:null,configBuffer:null,configData:null,bufferPool:null,planner:null,targetBuffer:null,targetBufferCapacity:0,seedCalculator:new We,isRunning:!1,isPaused:!1,shouldStop:!1,lastProgressUpdateMs:0,timerState:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1}},t=async()=>{if(e.pipelines&&e.bufferPool&&e.planner&&e.deviceContext)return;const n=await He(),g=n.getDevice(),p=n.getSupportedWorkgroupSize(e.workgroupSize),{pipelines:h,layouts:y}=Ve(g,p),m=new Uint32Array(Ke),v=D(m.byteLength),i=g.createBuffer({label:"gpu-seed-config-buffer",size:v,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),C=$e(g,{slots:he,workgroupSize:p}),E=qe(n,{workgroupSize:p,bufferSetCount:he});e.deviceContext=n,e.pipelines=h,e.bindGroupLayouts=y,e.configBuffer=i,e.configData=m,e.bufferPool=C,e.planner=E,e.workgroupSize=p},c=n=>{if(!e.deviceContext)throw new Error("WebGPU device is not initialized");const g=e.deviceContext.getDevice(),p=n.length,h=1+p,y=D(h*Uint32Array.BYTES_PER_ELEMENT),m=e.targetBufferCapacity;if(!e.targetBuffer||m<p){e.targetBuffer?.destroy(),e.targetBuffer=g.createBuffer({label:"gpu-seed-target-buffer",size:y,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const E=Math.floor(y/Uint32Array.BYTES_PER_ELEMENT)-1;e.targetBufferCapacity=Math.max(E,p)}const i=new Uint32Array(h);i[0]=p>>>0;for(let E=0;E<p;E+=1)i[1+E]=n[E]>>>0;const C=i.byteLength;g.queue.writeBuffer(e.targetBuffer,0,i.buffer,i.byteOffset,C)},o=async n=>{if(e.isRunning)throw new Error("WebGPU search is already running");if((!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)&&await t(),!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)throw new Error("WebGPU runner failed to initialize");const{context:g,targetSeeds:p,callbacks:h,signal:y}=n;if(g.totalMessages===0){h.onComplete("探索対象の組み合わせが存在しません");return}if(!e.bindGroupLayouts)throw new Error("WebGPU runner missing bind group layout");c(p),e.isRunning=!0,e.isPaused=!1,e.shouldStop=!1,e.lastProgressUpdateMs=Date.now();const m={currentStep:0,totalSteps:g.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0};let v;if(y)if(y.aborted)e.shouldStop=!0;else{const i=()=>{e.shouldStop=!0};y.addEventListener("abort",i),v=()=>y.removeEventListener("abort",i)}x();try{await d(g,m,h);const i=_(),C={...m,elapsedTime:i,estimatedTimeRemaining:0};e.shouldStop?h.onStopped("検索を停止しました",C):(h.onProgress(C),h.onComplete(`検索が完了しました。${m.matchesFound}件ヒットしました。`))}catch(i){const C=i instanceof Error?i.message:"WebGPU検索中に不明なエラーが発生しました",E=i instanceof GPUValidationError?"WEBGPU_VALIDATION_ERROR":void 0;throw h.onError(C,E),i}finally{e.isRunning=!1,e.isPaused=!1,e.shouldStop=!1,M(),v&&v()}},u=()=>{!e.isRunning||e.isPaused||(e.isPaused=!0,M())},l=()=>{!e.isRunning||!e.isPaused||(e.isPaused=!1,T())},f=()=>{e.isRunning&&(e.shouldStop=!0,e.isPaused=!1,T())},a=()=>{e.bufferPool?.dispose(),e.configBuffer?.destroy(),e.configBuffer=null,e.configData=null,e.pipelines=null,e.bindGroupLayouts=null,e.bufferPool=null,e.planner=null,e.deviceContext=null,e.targetBuffer?.destroy(),e.targetBuffer=null,e.targetBufferCapacity=0},d=async(n,g,p)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const y=e.deviceContext.getDevice().queue,m=e.bufferPool.slotCount,v=Array.from({length:m},()=>Promise.resolve());let i=0;for(const C of n.segments){if(e.shouldStop)break;const E=e.planner.computePlan(C.totalMessages);for(const A of E.dispatches){if(e.shouldStop||(await G(),e.shouldStop))break;const U=i%m;await v[U];const L={segment:C,dispatchIndex:i,messageCount:A.messageCount,slotIndex:U};v[U]=s(L,A.baseOffset,n,g,p,y),i+=1}}await Promise.all(v)},s=async(n,g,p,h,y,m)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const v=e.deviceContext.getDevice(),i=e.bufferPool.acquire(n.slotIndex,n.messageCount),C=Math.ceil(n.messageCount/e.workgroupSize),E=Math.max(1,Math.ceil(i.candidateCapacity/e.workgroupSize)),A=D(K*Uint32Array.BYTES_PER_ELEMENT);m.writeBuffer(i.output,0,le.buffer,le.byteOffset,le.byteLength),P(n.segment,g,n.messageCount,i.groupCount,i.candidateCapacity),m.writeBuffer(e.configBuffer,0,e.configData.buffer,e.configData.byteOffset,e.configData.byteLength);const U=v.createBindGroup({label:`gpu-seed-generate-group-${n.dispatchIndex}`,layout:e.bindGroupLayouts.generate,entries:[{binding:0,resource:{buffer:e.configBuffer}},{binding:1,resource:{buffer:e.targetBuffer}},{binding:2,resource:{buffer:i.candidate}},{binding:3,resource:{buffer:i.groupCounts}}]}),L=v.createBindGroup({label:`gpu-seed-scan-group-${n.dispatchIndex}`,layout:e.bindGroupLayouts.scan,entries:[{binding:0,resource:{buffer:e.configBuffer}},{binding:3,resource:{buffer:i.groupCounts}},{binding:4,resource:{buffer:i.groupOffsets}},{binding:5,resource:{buffer:i.output}}]}),w=v.createBindGroup({label:`gpu-seed-scatter-group-${n.dispatchIndex}`,layout:e.bindGroupLayouts.scatter,entries:[{binding:0,resource:{buffer:e.configBuffer}},{binding:2,resource:{buffer:i.candidate}},{binding:3,resource:{buffer:i.groupCounts}},{binding:4,resource:{buffer:i.groupOffsets}},{binding:5,resource:{buffer:i.output}}]}),z=v.createCommandEncoder({label:`gpu-seed-compute-${n.dispatchIndex}`}),Y=z.beginComputePass({label:`gpu-seed-generate-pass-${n.dispatchIndex}`});Y.setPipeline(e.pipelines.generate),Y.setBindGroup(0,U),Y.dispatchWorkgroups(C),Y.end();const k=z.beginComputePass({label:`gpu-seed-scan-pass-${n.dispatchIndex}`});k.setPipeline(e.pipelines.scan),k.setBindGroup(0,L),k.dispatchWorkgroups(1),k.end();const W=z.beginComputePass({label:`gpu-seed-scatter-pass-${n.dispatchIndex}`});W.setPipeline(e.pipelines.scatter),W.setBindGroup(0,w),W.dispatchWorkgroups(E),W.end(),z.copyBufferToBuffer(i.output,0,i.matchCount,0,A),m.submit([z.finish()]),await m.onSubmittedWorkDone(),await i.matchCount.mapAsync(GPUMapMode.READ,0,A);const F=new Uint32Array(i.matchCount.getMappedRange(0,A))[0]??0;i.matchCount.unmap();const $=Math.min(F,i.maxRecords)*ee*Uint32Array.BYTES_PER_ELEMENT,I=D(K*Uint32Array.BYTES_PER_ELEMENT+$),Z=v.createCommandEncoder({label:`gpu-seed-copy-${n.dispatchIndex}`});Z.copyBufferToBuffer(i.output,0,i.readback,0,I),m.submit([Z.finish()]),await m.onSubmittedWorkDone(),await i.readback.mapAsync(GPUMapMode.READ,0,I);const ce=i.readback.getMappedRange(0,I),q=new Uint32Array(ce),ne=q[0]??0,ae=Math.max(0,Math.floor((q.length-K)/ee)),j=Math.min(ne,i.maxRecords,ae),oe=K+j*ee,J=new Uint32Array(oe);J.set(q.subarray(0,oe)),i.readback.unmap(),await b(J,j,n,g,p,h,y)},b=async(n,g,p,h,y,m,v)=>{const i=p.segment,C=i.rangeSeconds,E=Math.max(C,1),A=Math.max(i.config.vcountCount,1),U=E,L=U*A,w=i.config.timer0Min,z=i.config.vcountMin,Y=h;for(let k=0;k<g&&!(e.shouldStop||k%je===0&&(await G(),e.shouldStop));k+=1){const W=K+k*ee,te=n[W],F=Y+te,V=n[W+1]>>>0,$=Math.floor(F/L),I=F-$*L,Z=Math.floor(I/U),ce=I-Z*U,q=w+$,ne=z+Z,ae=new Date(y.startTimestampMs+ce*1e3),j=e.seedCalculator.generateMessage(y.conditions,q,ne,ae),{hash:oe,seed:J}=e.seedCalculator.calculateSeed(j);J!==V&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:V,cpuSeed:J,messageIndex:F});const Re={seed:V,datetime:ae,timer0:q,vcount:ne,conditions:y.conditions,message:j,sha1Hash:oe,isMatch:!0};v.onResult(Re),m.matchesFound+=1}if(p.messageCount>0){const k=p.messageCount-1,W=Y+k,te=Math.floor(W/L),F=W-te*L,V=Math.floor(F/U),$=F-V*U,I=new Date(y.startTimestampMs+$*1e3).toISOString();m.currentDateTime=I}m.currentStep+=p.messageCount,S(m,v)},S=(n,g)=>{const p=Date.now();if(p-e.lastProgressUpdateMs<Ze&&n.currentStep<n.totalSteps)return;const h=_(),y=R(n.currentStep,n.totalSteps,h);g.onProgress({currentStep:n.currentStep,totalSteps:n.totalSteps,elapsedTime:h,estimatedTimeRemaining:y,matchesFound:n.matchesFound,currentDateTime:n.currentDateTime}),e.lastProgressUpdateMs=p},R=(n,g,p)=>{if(n===0||n>=g)return 0;const h=p/n,y=g-n;return Math.round(h*y)},P=(n,g,p,h,y)=>{if(!e.configData)throw new Error("config buffer not prepared");const m=Math.max(n.config.rangeSeconds,1),v=Math.max(n.config.vcountCount,1),i=m,C=i*v,E=Math.floor(g/C),A=g-E*C,U=Math.floor(A/i),L=A-U*i,w=e.configData;w[0]=p>>>0,w[1]=E>>>0,w[2]=U>>>0,w[3]=L>>>0,w[4]=n.config.rangeSeconds>>>0,w[5]=n.config.timer0Min>>>0,w[6]=n.config.timer0Count>>>0,w[7]=n.config.vcountMin>>>0,w[8]=n.config.vcountCount>>>0,w[9]=n.config.startSecondOfDay>>>0,w[10]=n.config.startDayOfWeek>>>0,w[11]=n.config.macLower>>>0,w[12]=n.config.data7Swapped>>>0,w[13]=n.config.keyInputSwapped>>>0,w[14]=n.config.hardwareType>>>0;for(let z=0;z<n.config.nazoSwapped.length;z+=1)w[15+z]=n.config.nazoSwapped[z]>>>0;w[20]=n.config.startYear>>>0,w[21]=n.config.startDayOfYear>>>0,w[22]=h>>>0,w[23]=e.workgroupSize>>>0,w[24]=y>>>0},x=()=>{e.timerState.cumulativeRunTime=0,e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1},M=()=>{e.timerState.isPaused||(e.timerState.cumulativeRunTime+=Date.now()-e.timerState.segmentStartTime,e.timerState.isPaused=!0)},T=()=>{e.timerState.isPaused&&(e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1)},_=()=>e.timerState.isPaused?e.timerState.cumulativeRunTime:e.timerState.cumulativeRunTime+(Date.now()-e.timerState.segmentStartTime),G=async()=>{for(;e.isPaused&&!e.shouldStop;)await N(25)},N=n=>new Promise(g=>setTimeout(g,n)),D=n=>Math.ceil(n/256)*256;return{init:t,run:o,pause:u,resume:l,stop:f,dispose:a}}const we=self,B={isRunning:!1,isPaused:!1},ie=Xe();let se=null;function O(r){we.postMessage(r)}function Qe(){O({type:"READY",message:"WebGPU worker initialized"})}function Q(){B.isRunning=!1,B.isPaused=!1,se=null}function er(){return Je()?!0:(O({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function rr(r){if(B.isRunning){O({type:"ERROR",error:"Search is already running"});return}if(!r.conditions||!r.targetSeeds){O({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!er())return;B.isRunning=!0,B.isPaused=!1;let e;try{e=Ce(r.conditions)}catch(o){Q();const u=o instanceof Error?o.message:"検索条件の解析中にエラーが発生しました";O({type:"ERROR",error:u,errorCode:"WEBGPU_CONTEXT_ERROR"});return}se=new AbortController;const t={onProgress:o=>{O({type:"PROGRESS",progress:o})},onResult:o=>{O({type:"RESULT",result:o})},onComplete:o=>{Q(),O({type:"COMPLETE",message:o})},onError:(o,u)=>{Q(),O({type:"ERROR",error:o,errorCode:u})},onPaused:()=>{B.isPaused=!0,O({type:"PAUSED"})},onResumed:()=>{B.isPaused=!1,O({type:"RESUMED"})},onStopped:(o,u)=>{Q(),O({type:"STOPPED",message:o,progress:u})}},c={context:e,targetSeeds:r.targetSeeds,callbacks:t,signal:se.signal};try{await ie.run(c)}catch(o){if(!B.isRunning)return;Q();const u=o instanceof Error?o.message:"WebGPU search failed with unknown error";O({type:"ERROR",error:u,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function tr(){!B.isRunning||B.isPaused||(ie.pause(),B.isPaused=!0,O({type:"PAUSED"}))}function nr(){!B.isRunning||!B.isPaused||(ie.resume(),B.isPaused=!1,O({type:"RESUMED"}))}function ar(){B.isRunning&&(ie.stop(),se?.abort())}Qe();we.onmessage=r=>{const e=r.data;switch(e.type){case"START_SEARCH":rr(e);break;case"PAUSE_SEARCH":tr();break;case"RESUME_SEARCH":nr();break;case"STOP_SEARCH":ar();break;default:O({type:"ERROR",error:`Unknown request type: ${e.type}`})}};
