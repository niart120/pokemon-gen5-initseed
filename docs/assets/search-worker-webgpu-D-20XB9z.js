const Ue={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},Ye=[["A",0],["B",1],["Select",2],["Start",3],["Right",4],["Left",5],["Up",6],["Down",7],["R",8],["L",9],["X",10],["Y",11]],X=Ye.reduce((t,[n,a])=>(t[n]=a,t),{}),Ve=4095,ze=12287,$e=[1<<X.Up|1<<X.Down,1<<X.Left|1<<X.Right,1<<X.Select|1<<X.Start|1<<X.L|1<<X.R];function Ae(t){return Number.isFinite(t)?t&Ve:0}function Ke(t){const n=Ae(t);for(const a of $e)if((n&a)===a)return!0;return!1}function qe(t){return ze^Ae(t)}const Ze=Date.UTC(2e3,0,1,0,0,0),je=100663296,Xe={DS:8,DS_LITE:6,"3DS":9};function Je(t){const n=[];for(let e=0;e<12;e++)(t&1<<e)!==0&&n.push(e);const a=n.length,i=1<<a,s=[];for(let e=0;e<i;e++){let d=0;for(let r=0;r<a;r++)(e&1<<r)!==0&&(d|=1<<n[r]);if(Ke(d))continue;const c=d^ze;s.push(c)}return s}function Qe(t){const n=Te(t.dateRange.startYear,t.dateRange.startMonth,t.dateRange.startDay,t.dateRange.startHour,t.dateRange.startMinute,t.dateRange.startSecond),a=Te(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond);if(n.getTime()>a.getTime())throw new Error("開始日時が終了日時より後ろに設定されています");const i=et(t),s=tt(t,i),e=xe(t.dateRange.startYear,t.dateRange.startMonth,t.dateRange.startDay,t.dateRange.startHour,t.dateRange.startMinute,t.dateRange.startSecond),d=xe(t.dateRange.endYear,t.dateRange.endMonth,t.dateRange.endDay,t.dateRange.endHour,t.dateRange.endMinute,t.dateRange.endSecond),c=Math.floor((e-Ze)/1e3);if(c<0)throw new Error("2000年より前の日時は指定できません");const r=Math.floor((d-e)/1e3)+1;if(r<=0)throw new Error("探索秒数が0秒以下です");const f=n.getFullYear(),u=ut(n),y=it(n),b=n.getDay(),v=Xe[t.hardware],{macLower:C,data7Swapped:M}=at(t.macAddress,v),T=Je(t.keyInput),B=st(i.nazo),_=[];let W=0;for(const I of T){const K=ve(I>>>0);for(let N=0;N<s.length;N+=1){const U=s[N],Q=U.timer0Max-U.timer0Min+1,Y=r*Q,o={startSecondsSince2000:c>>>0,rangeSeconds:r>>>0,timer0Min:U.timer0Min>>>0,timer0Max:U.timer0Max>>>0,timer0Count:Q>>>0,vcountMin:U.vcount>>>0,vcountMax:U.vcount>>>0,vcountCount:1,totalMessages:Y>>>0,hardwareType:ot(t.hardware),macLower:C>>>0,data7Swapped:M>>>0,keyInputSwapped:K>>>0,nazoSwapped:B,startYear:f>>>0,startDayOfYear:u>>>0,startSecondOfDay:y>>>0,startDayOfWeek:b>>>0};_.push({index:N,baseOffset:W,timer0Min:U.timer0Min,timer0Max:U.timer0Max,timer0Count:Q,vcount:U.vcount,rangeSeconds:r,totalMessages:Y,keyCode:I,config:o}),W+=Y}}const ue=_.reduce((I,K)=>I+K.totalMessages,0);return{conditions:t,startDate:n,startTimestampMs:n.getTime(),rangeSeconds:r,totalMessages:ue,segments:_}}function Te(t,n,a,i,s,e){return new Date(t,n-1,a,i,s,e)}function xe(t,n,a,i,s,e){return Date.UTC(t,n-1,a,i,s,e,0)}function et(t){const n=Ue[t.romVersion];if(!n)throw new Error(`ROMバージョン ${t.romVersion} は未対応です`);const a=n[t.romRegion];if(!a)throw new Error(`ROMリージョン ${t.romRegion} は未対応です`);return{nazo:[...a.nazo],vcountTimerRanges:a.vcountTimerRanges.map(i=>[...i])}}function tt(t,n){const a=[],i=t.timer0VCountConfig.timer0Range.min,s=t.timer0VCountConfig.timer0Range.max;let e=null;for(let d=i;d<=s;d+=1){const c=nt(n,d);e&&e.vcount===c&&d===e.timer0Max+1?e.timer0Max=d:(e&&a.push(e),e={timer0Min:d,timer0Max:d,vcount:c})}return e&&a.push(e),a}function nt(t,n){for(const[a,i,s]of t.vcountTimerRanges)if(n>=i&&n<=s)return a;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}function at(t,n){const a=rt(t),i=(a[4]&255)<<8|a[5]&255,e=((a[0]&255|(a[1]&255)<<8|(a[2]&255)<<16|(a[3]&255)<<24)^je^n)>>>0;return{macLower:i,data7Swapped:ve(e)}}function rt(t){const n=new Array(6).fill(0);for(let a=0;a<6;a+=1){const i=t[a]??0;n[a]=(Number(i)&255)>>>0}return n}function ot(t){switch(t){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function ve(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}function st(t){const n=new Uint32Array(t.length);for(let a=0;a<t.length;a+=1)n[a]=ve(t[a]>>>0);return n}function ut(t){const n=new Date(t.getFullYear(),0,1),a=t.getTime()-n.getTime();return Math.floor(a/(1440*60*1e3))+1}function it(t){return t.getHours()*3600+t.getMinutes()*60+t.getSeconds()}class Oe{calculateHash(n){if(n.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const a=1732584193,i=4023233417,s=2562383102,e=271733878,d=3285377520,c=new Array(80);for(let _=0;_<16;_++)c[_]=n[_];for(let _=16;_<80;_++)c[_]=this.leftRotate(c[_-3]^c[_-8]^c[_-14]^c[_-16],1);let r=a,f=i,u=s,y=e,b=d;for(let _=0;_<80;_++){let W;_<20?W=this.leftRotate(r,5)+(f&u|~f&y)+b+c[_]+1518500249&4294967295:_<40?W=this.leftRotate(r,5)+(f^u^y)+b+c[_]+1859775393&4294967295:_<60?W=this.leftRotate(r,5)+(f&u|f&y|u&y)+b+c[_]+2400959708&4294967295:W=this.leftRotate(r,5)+(f^u^y)+b+c[_]+3395469782&4294967295,b=y,y=u,u=this.leftRotate(f,30),f=r,r=W}const v=this.add32(a,r),C=this.add32(i,f),M=this.add32(s,u),T=this.add32(e,y),B=this.add32(d,b);return{h0:v,h1:C,h2:M,h3:T,h4:B}}leftRotate(n,a){return(n<<a|n>>>32-a)>>>0}add32(n,a){return(n+a&4294967295)>>>0}static hashToHex(n,a,i,s,e){return n.toString(16).padStart(8,"0")+a.toString(16).padStart(8,"0")+i.toString(16).padStart(8,"0")+s.toString(16).padStart(8,"0")+e.toString(16).padStart(8,"0")}}let J=null,ie=null;async function ct(){return J||ie||(ie=(async()=>{try{const t=await import("./wasm_pkg-DM9yDBLH.js");let n;if(typeof process<"u"&&!!process.versions?.node){const i=await import("./__vite-browser-external-9wXp6ZBx.js"),e=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");n={module_or_path:i.readFileSync(e)}}else n={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-IVI303-w.wasm",import.meta.url)};return await t.default(n),J={IntegratedSeedSearcher:t.IntegratedSeedSearcher,BWGenerationConfig:t.BWGenerationConfig,PokemonGenerator:t.PokemonGenerator,SeedEnumerator:t.SeedEnumerator,EncounterType:t.EncounterType,GameVersion:t.GameVersion,GameMode:t.GameMode,calculate_game_offset:t.calculate_game_offset,sha1_hash_batch:t.sha1_hash_batch},J}catch(t){throw console.error("Failed to load WebAssembly module:",t),J=null,ie=null,t}})(),ie)}function lt(){if(!J)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return J}function Be(){return J!==null}const dt={DS:8,DS_LITE:6,"3DS":9};class ft{sha1;useWasm=!1;constructor(){this.sha1=new Oe}async initializeWasm(){try{return await ct(),this.useWasm=!0,!0}catch(n){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",n),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&Be()}getWasmModule(){return lt()}setUseWasm(n){if(n&&!Be()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=n}getROMParameters(n,a){const i=Ue[n];if(!i)return console.error(`ROM version not found: ${n}`),null;const s=i[a];return s?{nazo:[...s.nazo],vcountTimerRanges:s.vcountTimerRanges.map(e=>[...e])}:(console.error(`ROM region not found: ${a} for version ${n}`),null)}toLittleEndian32(n){return((n&255)<<24|(n>>>8&255)<<16|(n>>>16&255)<<8|n>>>24&255)>>>0}toLittleEndian16(n){return(n&255)<<8|n>>>8&255}getDayOfWeek(n,a,i){return new Date(n,a-1,i).getDay()}generateMessage(n,a,i,s,e){const d=this.getROMParameters(n.romVersion,n.romRegion);if(!d)throw new Error(`No parameters found for ${n.romVersion} ${n.romRegion}`);const c=new Array(16).fill(0);for(let g=0;g<5;g++)c[g]=this.toLittleEndian32(d.nazo[g]);c[5]=this.toLittleEndian32(i<<16|a);const r=n.macAddress[4]<<8|n.macAddress[5];c[6]=r;const f=n.macAddress[0]<<0|n.macAddress[1]<<8|n.macAddress[2]<<16|n.macAddress[3]<<24,u=100663296,y=dt[n.hardware],b=f^u^y;c[7]=this.toLittleEndian32(b);const v=s.getFullYear()%100,C=s.getMonth()+1,M=s.getDate(),T=this.getDayOfWeek(s.getFullYear(),C,M),B=Math.floor(v/10)*16+v%10,_=Math.floor(C/10)*16+C%10,W=Math.floor(M/10)*16+M%10,ue=Math.floor(T/10)*16+T%10;c[8]=B<<24|_<<16|W<<8|ue;const I=s.getHours(),K=s.getMinutes(),N=s.getSeconds(),U=(n.hardware==="DS"||n.hardware==="DS_LITE")&&I>=12?1:0,Q=Math.floor(I/10)*16+I%10,Y=Math.floor(K/10)*16+K%10,o=Math.floor(N/10)*16+N%10;c[9]=U<<30|Q<<24|Y<<16|o<<8|0,c[10]=0,c[11]=0;const p=e??qe(n.keyInput);return c[12]=this.toLittleEndian32(p),c[13]=2147483648,c[14]=0,c[15]=416,c}calculateSeed(n){const a=this.sha1.calculateHash(n),i=BigInt(this.toLittleEndian32(a.h0)),e=BigInt(this.toLittleEndian32(a.h1))<<32n|i,r=e*0x5D588B656C078965n+0x269EC3n;return{seed:Number(r>>32n&0xFFFFFFFFn),hash:Oe.hashToHex(a.h0,a.h1,a.h2,a.h3,a.h4),lcgSeed:e}}parseTargetSeeds(n){const a=n.split(`
`).map(d=>d.trim()).filter(d=>d.length>0),i=[],s=[],e=new Set;return a.forEach((d,c)=>{try{let r=d.toLowerCase();if(r.startsWith("0x")&&(r=r.substring(2)),!/^[0-9a-f]{1,8}$/.test(r)){s.push({line:c+1,value:d,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const f=parseInt(r,16);if(e.has(f))return;e.add(f),i.push(f)}catch(r){const f=r instanceof Error?r.message:String(r);s.push({line:c+1,value:d,error:f||"Failed to parse as hexadecimal number."})}}),{validSeeds:i,errors:s}}getVCountForTimer0(n,a){for(const[i,s,e]of n.vcountTimerRanges)if(a>=s&&a<=e)return i;return n.vcountTimerRanges.length>0?n.vcountTimerRanges[0][0]:96}}const ke=Uint32Array.BYTES_PER_ELEMENT,le=2,de=le*ke,se=1,De=se*ke,me=256,Ge=256*1024*1024,Ee=2,gt={requiredFeatures:[]};function Le(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}async function pt(t){if(!Le())throw new Error("WebGPU is not available in this environment");const a=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!a)throw new Error("Failed to acquire WebGPU adapter");const i={requiredFeatures:gt.requiredFeatures,requiredLimits:t?.requiredLimits,label:"seed-search-device"},s=await a.requestDevice(i);let e=!1;const d=s.lost.then(c=>(e=!0,console.warn("[webgpu] device lost:",c.message),c));return{getAdapter:()=>a,getDevice:()=>s,getQueue:()=>s.queue,getLimits:()=>s.limits,isLost:()=>e,waitForLoss:()=>d,getSupportedWorkgroupSize:(c=me)=>{const r=s.limits,f=r.maxComputeInvocationsPerWorkgroup??c,u=r.maxComputeWorkgroupSizeX??c,y=Math.min(c,f,u);if(y<=0)throw new Error("WebGPU workgroup size limits are invalid");return y}}}var mt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;

struct GeneratedConfig {
  message_count : u32,
  base_timer0_index : u32,
  base_vcount_index : u32,
  base_second_offset : u32,
  range_seconds : u32,
  timer0_min : u32,
  timer0_count : u32,
  vcount_min : u32,
  vcount_count : u32,
  start_second_of_day : u32,
  start_day_of_week : u32,
  mac_lower : u32,
  data7_swapped : u32,
  key_input_swapped : u32,
  hardware_type : u32,
  nazo0 : u32,
  nazo1 : u32,
  nazo2 : u32,
  nazo3 : u32,
  nazo4 : u32,
  start_year : u32,
  start_day_of_year : u32,
  groups_per_dispatch : u32,
  configured_workgroup_size : u32,
  candidate_capacity : u32,
};

struct TargetSeedBuffer {
  count : u32,
  values : array<u32>,
};

struct CandidateRecord {
  message_index : u32,
  seed : u32,
};

struct CandidateBuffer {
  records : array<CandidateRecord>,
};

struct GroupCountBuffer {
  values : array<u32>,
};

struct GroupOffsetBuffer {
  values : array<u32>,
};

struct MatchRecord {
  message_index : u32,
  seed : u32,
};

struct MatchOutputBuffer {
  match_count : atomic<u32>,
  records : array<MatchRecord>,
};

struct WideProduct {
  lo : u32,
  hi : u32,
};

struct CarryResult {
  sum : u32,
  carry : u32,
};

const MONTH_LENGTHS_COMMON : array<u32, 12> = array<u32, 12>(
  31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);
const MONTH_LENGTHS_LEAP : array<u32, 12> = array<u32, 12>(
  31u, 29u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);

@group(0) @binding(0) var<storage, read> config : GeneratedConfig;
@group(0) @binding(1) var<storage, read> target_seeds : TargetSeedBuffer;
@group(0) @binding(2) var<storage, read_write> candidate_buffer : CandidateBuffer;
@group(0) @binding(3) var<storage, read_write> group_counts_buffer : GroupCountBuffer;
@group(0) @binding(4) var<storage, read_write> group_offsets_buffer : GroupOffsetBuffer;
@group(0) @binding(5) var<storage, read_write> output_buffer : MatchOutputBuffer;

var<workgroup> scan_values : array<u32, WORKGROUP_SIZE>;
var<workgroup> group_total_matches : u32;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
}

fn swap32(value : u32) -> u32 {
  return ((value & 0x000000FFu) << 24u) |
    ((value & 0x0000FF00u) << 8u) |
    ((value & 0x00FF0000u) >> 8u) |
    ((value & 0xFF000000u) >> 24u);
}

fn to_bcd(value : u32) -> u32 {
  let tens = value / 10u;
  let ones = value - tens * 10u;
  return (tens << 4u) | ones;
}

fn is_leap_year(year : u32) -> bool {
  return (year % 4u == 0u && year % 100u != 0u) || (year % 400u == 0u);
}

fn month_day_from_day_of_year(day_of_year : u32, leap : bool) -> vec2<u32> {
  var remaining = day_of_year;
  var month = 1u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let length = select(MONTH_LENGTHS_COMMON[i], MONTH_LENGTHS_LEAP[i], leap);
    if (remaining <= length) {
      return vec2<u32>(month, remaining);
    }
    remaining = remaining - length;
    month = month + 1u;
  }
  return vec2<u32>(12u, 31u);
}

fn mulExtended(a : u32, b : u32) -> WideProduct {
  let a_lo = a & 0xFFFFu;
  let a_hi = a >> 16u;
  let b_lo = b & 0xFFFFu;
  let b_hi = b >> 16u;

  let low = a_lo * b_lo;
  let mid1 = a_lo * b_hi;
  let mid2 = a_hi * b_lo;
  let high = a_hi * b_hi;

  let carry_mid = (low >> 16u) + (mid1 & 0xFFFFu) + (mid2 & 0xFFFFu);
  let lo = (low & 0xFFFFu) | ((carry_mid & 0xFFFFu) << 16u);
  let hi = high + (mid1 >> 16u) + (mid2 >> 16u) + (carry_mid >> 16u);

  return WideProduct(lo, hi);
}

fn addCarry(a : u32, b : u32) -> CarryResult {
  let sum = a + b;
  let carry = select(0u, 1u, sum < a);
  return CarryResult(sum, carry);
}

fn compute_seed_from_hash(h0 : u32, h1 : u32) -> u32 {
  let le0 = swap32(h0);
  let le1 = swap32(h1);

  let mul_lo : u32 = 0x6C078965u;
  let mul_hi : u32 = 0x5D588B65u;
  let increment : u32 = 0x00269EC3u;

  let prod0 = mulExtended(le0, mul_lo);
  let prod1 = mulExtended(le0, mul_hi);
  let prod2 = mulExtended(le1, mul_lo);
  let inc = addCarry(prod0.lo, increment);

  // Upper 32-bit word of ((le1<<32 | le0) * multiplier + increment)
  var upper_word = prod0.hi;
  upper_word = upper_word + prod1.lo;
  upper_word = upper_word + prod2.lo;
  upper_word = upper_word + inc.carry;

  return upper_word;
}

@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)
fn sha1_generate(
  @builtin(global_invocation_id) global_id : vec3<u32>,
  @builtin(local_invocation_id) local_id : vec3<u32>,
  @builtin(local_invocation_index) local_linear_index : u32,
  @builtin(workgroup_id) workgroup_id : vec3<u32>
) {

  let global_linear_index = global_id.x;
  let is_active = global_linear_index < config.message_count;
  let group_index = workgroup_id.x;
  let configured_workgroup_size = config.configured_workgroup_size;

  var local_message_index : u32 = 0u;
  var seed : u32 = 0u;
  var matched = false;

  if (is_active) {
    let safe_range_seconds = max(config.range_seconds, 1u);
    let safe_vcount_count = max(config.vcount_count, 1u);
    let messages_per_vcount = safe_range_seconds;
    let messages_per_timer0 = messages_per_vcount * safe_vcount_count;

    let local_timer0_index = global_linear_index / messages_per_timer0;
    let local_remainder_after_timer0 = global_linear_index - local_timer0_index * messages_per_timer0;
    let local_vcount_index = local_remainder_after_timer0 / messages_per_vcount;
    let local_second_offset = local_remainder_after_timer0 - local_vcount_index * messages_per_vcount;

    let combined_second_offset = config.base_second_offset + local_second_offset;
    let carry_to_vcount = combined_second_offset / messages_per_vcount;
    let second_offset = combined_second_offset - carry_to_vcount * messages_per_vcount;

    let combined_vcount_index = config.base_vcount_index + local_vcount_index + carry_to_vcount;
    let carry_to_timer0 = combined_vcount_index / safe_vcount_count;
    let vcount_index = combined_vcount_index - carry_to_timer0 * safe_vcount_count;

    let timer0_index = config.base_timer0_index + local_timer0_index + carry_to_timer0;

    let timer0 = config.timer0_min + timer0_index;
    let vcount = config.vcount_min + vcount_index;

    let total_seconds = config.start_second_of_day + second_offset;
    let day_offset = total_seconds / 86400u;
    let seconds_of_day = total_seconds - day_offset * 86400u;

    let hour = seconds_of_day / 3600u;
    let minute = (seconds_of_day % 3600u) / 60u;
    let second = seconds_of_day % 60u;

    var year = config.start_year;
    var day_of_year = config.start_day_of_year + day_offset;
    loop {
      let year_length = select(365u, 366u, is_leap_year(year));
      if (day_of_year <= year_length) {
        break;
      }
      day_of_year = day_of_year - year_length;
      year = year + 1u;
    }

    let leap = is_leap_year(year);
    let month_day = month_day_from_day_of_year(day_of_year, leap);
    let month = month_day.x;
    let day = month_day.y;

    let day_of_week = (config.start_day_of_week + day_offset) % 7u;
    let year_mod = year % 100u;
    let date_word = (to_bcd(year_mod) << 24u) | (to_bcd(month) << 16u) | (to_bcd(day) << 8u) | to_bcd(day_of_week);
    let is_pm = (config.hardware_type <= 1u) && (hour >= 12u);
    let pm_flag = select(0u, 1u, is_pm);
    let time_word = (pm_flag << 30u) | (to_bcd(hour) << 24u) | (to_bcd(minute) << 16u) | (to_bcd(second) << 8u);

    var w : array<u32, 16>;
    w[0] = config.nazo0;
    w[1] = config.nazo1;
    w[2] = config.nazo2;
    w[3] = config.nazo3;
    w[4] = config.nazo4;
    w[5] = swap32((vcount << 16u) | timer0);
    w[6] = config.mac_lower;
    w[7] = config.data7_swapped;
    w[8] = date_word;
    w[9] = time_word;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = config.key_input_swapped;
    w[13] = 0x80000000u;
    w[14] = 0u;
    w[15] = 0x000001A0u;

    var a : u32 = 0x67452301u;
    var b : u32 = 0xEFCDAB89u;
    var c : u32 = 0x98BADCFEu;
    var d : u32 = 0x10325476u;
    var e : u32 = 0xC3D2E1F0u;

    var i : u32 = 0u;
    for (; i < 20u; i = i + 1u) {
      let w_index = i & 15u;
      var w_value : u32;
      if (i < 16u) {
        w_value = w[w_index];
      } else {
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
        let rotated = left_rotate(expanded, 1u);
        w[w_index] = rotated;
        w_value = rotated;
      }

      let temp = left_rotate(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5A827999u + w_value;
      e = d;
      d = c;
      c = left_rotate(b, 30u);
      b = a;
      a = temp;
    }

    for (; i < 40u; i = i + 1u) {
      let w_index = i & 15u;
      var w_value : u32;
      if (i < 16u) {
        w_value = w[w_index];
      } else {
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
        let rotated = left_rotate(expanded, 1u);
        w[w_index] = rotated;
        w_value = rotated;
      }

      let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0x6ED9EBA1u + w_value;
      e = d;
      d = c;
      c = left_rotate(b, 30u);
      b = a;
      a = temp;
    }

    for (; i < 60u; i = i + 1u) {
      let w_index = i & 15u;
      var w_value : u32;
      if (i < 16u) {
        w_value = w[w_index];
      } else {
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
        let rotated = left_rotate(expanded, 1u);
        w[w_index] = rotated;
        w_value = rotated;
      }

      let temp = left_rotate(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + w_value;
      e = d;
      d = c;
      c = left_rotate(b, 30u);
      b = a;
      a = temp;
    }

    for (; i < 80u; i = i + 1u) {
      let w_index = i & 15u;
      var w_value : u32;
      if (i < 16u) {
        w_value = w[w_index];
      } else {
        let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
        let rotated = left_rotate(expanded, 1u);
        w[w_index] = rotated;
        w_value = rotated;
      }

      let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0xCA62C1D6u + w_value;
      e = d;
      d = c;
      c = left_rotate(b, 30u);
      b = a;
      a = temp;
    }

    let h0 = 0x67452301u + a;
    let h1 = 0xEFCDAB89u + b;
    let h2 = 0x98BADCFEu + c;
    let h3 = 0x10325476u + d;
    let h4 = 0xC3D2E1F0u + e;

    seed = compute_seed_from_hash(h0, h1);

    let target_count = target_seeds.count;
    matched = target_count == 0u;
    for (var j = 0u; j < target_count; j = j + 1u) {
      if (target_seeds.values[j] == seed) {
        matched = true;
        break;
      }
    }
  }

  let match_flag = select(0u, 1u, matched);
  scan_values[local_linear_index] = match_flag;
  workgroupBarrier();

  var offset = 1u;
  while (offset < WORKGROUP_SIZE) {
    workgroupBarrier();
    let current_value = scan_values[local_linear_index];
    var addend = 0u;
    if (local_linear_index >= offset) {
      addend = scan_values[local_linear_index - offset];
    }
    workgroupBarrier();
    scan_values[local_linear_index] = current_value + addend;
    offset = offset << 1u;
  }

  workgroupBarrier();
  let inclusive_sum = scan_values[local_linear_index];
  if (local_linear_index == (WORKGROUP_SIZE - 1u)) {
    group_total_matches = inclusive_sum;
  }
  workgroupBarrier();

  if (local_linear_index == 0u) {
    group_counts_buffer.values[group_index] = group_total_matches;
  }

  if (match_flag == 0u) {
    return;
  }

  let record_rank = inclusive_sum - 1u;
  let candidate_index = group_index * configured_workgroup_size + record_rank;
  if (candidate_index >= config.candidate_capacity) {
    return;
  }

  local_message_index = global_linear_index;
  candidate_buffer.records[candidate_index].message_index = local_message_index;
  candidate_buffer.records[candidate_index].seed = seed;
}

@compute @workgroup_size(1)
fn exclusive_scan_groups(@builtin(global_invocation_id) global_id : vec3<u32>) {
  if (global_id.x != 0u) {
    return;
  }

  let group_count = config.groups_per_dispatch;
  var running_total = 0u;
  for (var i = 0u; i < group_count; i = i + 1u) {
    let count = group_counts_buffer.values[i];
    group_offsets_buffer.values[i] = running_total;
    running_total = running_total + count;
  }

  atomicStore(&output_buffer.match_count, running_total);
}

@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)
fn scatter_matches(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let candidate_index = global_id.x;
  if (candidate_index >= config.candidate_capacity) {
    return;
  }

  let configured_workgroup_size = config.configured_workgroup_size;
  let group_index = candidate_index / configured_workgroup_size;
  if (group_index >= config.groups_per_dispatch) {
    return;
  }

  let local_rank = candidate_index - group_index * configured_workgroup_size;
  let group_match_count = group_counts_buffer.values[group_index];
  if (local_rank >= group_match_count) {
    return;
  }

  let base_offset = group_offsets_buffer.values[group_index];
  let final_index = base_offset + local_rank;
  if (final_index >= config.message_count) {
    return;
  }

  let record = candidate_buffer.records[candidate_index];
  output_buffer.records[final_index].message_index = record.message_index;
  output_buffer.records[final_index].seed = record.seed;
}
`;function _t(t){return mt.replace(/WORKGROUP_SIZE_PLACEHOLDER/g,String(t))}function ht(t,n){const a=t.createShaderModule({label:"gpu-seed-sha1-generated-module",code:_t(n)});a.getCompilationInfo?.().then(f=>{f.messages.length>0&&console.warn("[pipeline-factory] compilation diagnostics",f.messages.map(u=>({message:u.message,line:u.lineNum,column:u.linePos,type:u.type})))}).catch(f=>{console.warn("[pipeline-factory] compilation info failed",f)});const i=t.createBindGroupLayout({label:"gpu-seed-generate-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),s=t.createBindGroupLayout({label:"gpu-seed-scan-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),e=t.createBindGroupLayout({label:"gpu-seed-scatter-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),d=t.createComputePipeline({label:"gpu-seed-generate-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-generate-pipeline-layout",bindGroupLayouts:[i]}),compute:{module:a,entryPoint:"sha1_generate"}}),c=t.createComputePipeline({label:"gpu-seed-scan-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scan-pipeline-layout",bindGroupLayouts:[s]}),compute:{module:a,entryPoint:"exclusive_scan_groups"}}),r=t.createComputePipeline({label:"gpu-seed-scatter-pipeline",layout:t.createPipelineLayout({label:"gpu-seed-scatter-pipeline-layout",bindGroupLayouts:[e]}),compute:{module:a,entryPoint:"scatter_matches"}});return{pipelines:{generate:d,scan:c,scatter:r},layouts:{generate:i,scan:s,scatter:e}}}function yt(t,n){const a=n?.slots??Ee,i=n?.workgroupSize??me;if(a<=0)throw new Error("buffer pool must have at least one slot");const s=Array.from({length:a},()=>({output:null,readback:null,candidate:null,groupCounts:null,groupOffsets:null,matchCount:null,outputSize:0,readbackSize:0,candidateSize:0,groupCountSize:0,groupOffsetSize:0,matchCountSize:0})),e=r=>Math.ceil(r/256)*256;return{get slotCount(){return s.length},acquire:(r,f)=>{if(r<0||r>=s.length)throw new Error(`buffer slot ${r} is out of range`);if(!Number.isFinite(f)||f<=0)throw new Error("messageCount must be a positive integer");const u=s[r],y=f,b=e(De+y*de),v=Math.max(1,Math.ceil(f/i)),C=v*i,M=e(C*de),T=e(v*Uint32Array.BYTES_PER_ELEMENT),B=e(De);return(!u.output||b>u.outputSize)&&(u.output?.destroy(),u.output=t.createBuffer({label:`gpu-seed-output-${r}`,size:b,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),u.outputSize=b),(!u.readback||b>u.readbackSize)&&(u.readback?.destroy(),u.readback=t.createBuffer({label:`gpu-seed-readback-${r}`,size:b,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.readbackSize=b),(!u.candidate||M>u.candidateSize)&&(u.candidate?.destroy(),u.candidate=t.createBuffer({label:`gpu-seed-candidate-${r}`,size:M,usage:GPUBufferUsage.STORAGE}),u.candidateSize=M),(!u.groupCounts||T>u.groupCountSize)&&(u.groupCounts?.destroy(),u.groupCounts=t.createBuffer({label:`gpu-seed-group-counts-${r}`,size:T,usage:GPUBufferUsage.STORAGE}),u.groupCountSize=T),(!u.groupOffsets||T>u.groupOffsetSize)&&(u.groupOffsets?.destroy(),u.groupOffsets=t.createBuffer({label:`gpu-seed-group-offsets-${r}`,size:T,usage:GPUBufferUsage.STORAGE}),u.groupOffsetSize=T),(!u.matchCount||B>u.matchCountSize)&&(u.matchCount?.destroy(),u.matchCount=t.createBuffer({label:`gpu-seed-match-header-${r}`,size:B,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),u.matchCountSize=B),{output:u.output,readback:u.readback,candidate:u.candidate,groupCounts:u.groupCounts,groupOffsets:u.groupOffsets,matchCount:u.matchCount,outputSize:u.outputSize,candidateCapacity:C,groupCount:v,maxRecords:y}},dispose:()=>{for(const r of s)r.output?.destroy(),r.readback?.destroy(),r.candidate?.destroy(),r.groupCounts?.destroy(),r.groupOffsets?.destroy(),r.matchCount?.destroy(),r.output=null,r.readback=null,r.candidate=null,r.groupCounts=null,r.groupOffsets=null,r.matchCount=null,r.outputSize=0,r.readbackSize=0,r.candidateSize=0,r.groupCountSize=0,r.groupOffsetSize=0,r.matchCountSize=0}}}function bt(t,n){const a=n?.hostMemoryLimitBytes??Ge,i=n?.bufferSetCount??Ee,s=(()=>{const r=n?.hostMemoryLimitPerSlot;if(typeof r=="number"&&Number.isFinite(r)&&r>0)return r;const f=Math.floor(a/i);return Math.max(1,f)})(),e=n?.workgroupSize??me;if(a<=0)throw new Error("host memory limit must be positive");if(i<=0)throw new Error("buffer set count must be positive");const d=r=>{const u=t.getDevice().limits,y=Math.max(1,u.maxStorageBufferBindingSize??de),b=Math.max(1,Math.floor(y/de)),v=Math.max(1,Math.floor(s/de)),C=t.getSupportedWorkgroupSize(e),M=u.maxComputeWorkgroupsPerDimension??65535,T=Math.max(1,C*M),B=Math.min(b,v,T);return r<=B?r<=1?1:Math.max(1,Math.min(B,Math.ceil(r/2))):B};return{computePlan:r=>{if(!Number.isFinite(r)||r<0)throw new Error("totalMessages must be a non-negative finite value");if(r===0)return{maxMessagesPerDispatch:0,dispatches:[]};const f=d(r),u=[];let y=r,b=0;for(;y>0;){const v=Math.min(f,y);u.push({baseOffset:b,messageCount:v}),b+=v,y-=v}if(u.length===1&&r>1){const v=u[0],C=Math.ceil(v.messageCount/2),M=v.messageCount-C;M>0&&(u[0]={baseOffset:v.baseOffset,messageCount:C},u.push({baseOffset:v.baseOffset+C,messageCount:M}))}return{maxMessagesPerDispatch:f,dispatches:u}}}}const St=25,wt=500,Rt=1024,Re=new Uint32Array([0]),vt=Le;function Et(t){const n=o=>Ee,a=Ge,i=n(),s=(()=>{const o=Math.floor(a/i);return Math.max(1,o)})(),e={workgroupSize:me,bufferSlotCount:i,hostMemoryLimitBytes:a,hostMemoryLimitPerSlotBytes:s,deviceContext:null,pipelines:null,bindGroupLayouts:null,configBuffer:null,configData:null,bufferPool:null,planner:null,targetBuffer:null,targetBufferCapacity:0,seedCalculator:new ft,isRunning:!1,isPaused:!1,shouldStop:!1,lastProgressUpdateMs:0,timerState:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1}},d=async(o,p,g)=>(async()=>await Promise.resolve(g()))(),c=async()=>{if(e.pipelines&&e.bufferPool&&e.planner&&e.deviceContext)return;const o=await pt(),p=o.getDevice(),g=o.getSupportedWorkgroupSize(e.workgroupSize),{pipelines:S,layouts:w}=ht(p,g),E=new Uint32Array(St),x=Y(E.byteLength),m=p.createBuffer({label:"gpu-seed-config-buffer",size:x,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),P=yt(p,{slots:e.bufferSlotCount,workgroupSize:g}),R=bt(o,{workgroupSize:g,bufferSetCount:e.bufferSlotCount,hostMemoryLimitBytes:e.hostMemoryLimitBytes,hostMemoryLimitPerSlot:e.hostMemoryLimitPerSlotBytes});e.deviceContext=o,e.pipelines=S,e.bindGroupLayouts=w,e.configBuffer=m,e.configData=E,e.bufferPool=P,e.planner=R,e.workgroupSize=g},r=o=>{if(!e.deviceContext)throw new Error("WebGPU device is not initialized");const p=e.deviceContext.getDevice(),g=o.length,S=1+g,w=Y(S*Uint32Array.BYTES_PER_ELEMENT),E=e.targetBufferCapacity;if(!e.targetBuffer||E<g){e.targetBuffer?.destroy(),e.targetBuffer=p.createBuffer({label:"gpu-seed-target-buffer",size:w,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const R=Math.floor(w/Uint32Array.BYTES_PER_ELEMENT)-1;e.targetBufferCapacity=Math.max(R,g)}const m=new Uint32Array(S);m[0]=g>>>0;for(let R=0;R<g;R+=1)m[1+R]=o[R]>>>0;const P=m.byteLength;p.queue.writeBuffer(e.targetBuffer,0,m.buffer,m.byteOffset,P)},f=async o=>{if(e.isRunning)throw new Error("WebGPU search is already running");if((!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)&&await c(),!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.deviceContext)throw new Error("WebGPU runner failed to initialize");const{context:p,targetSeeds:g,callbacks:S,signal:w}=o;if(p.totalMessages===0){S.onComplete("探索対象の組み合わせが存在しません");return}if(!e.bindGroupLayouts)throw new Error("WebGPU runner missing bind group layout");r(g),e.isRunning=!0,e.isPaused=!1,e.shouldStop=!1,e.lastProgressUpdateMs=Date.now();const E={currentStep:0,totalSteps:p.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0};let x;if(w)if(w.aborted)e.shouldStop=!0;else{const m=()=>{e.shouldStop=!0};w.addEventListener("abort",m),x=()=>w.removeEventListener("abort",m)}ue();try{await C(p,E,S);const m=N(),P={...E,elapsedTime:m,estimatedTimeRemaining:0};e.shouldStop?S.onStopped("検索を停止しました",P):(S.onProgress(P),S.onComplete(`検索が完了しました。${E.matchesFound}件ヒットしました。`))}catch(m){const P=m instanceof Error?m.message:"WebGPU検索中に不明なエラーが発生しました",R=m instanceof GPUValidationError?"WEBGPU_VALIDATION_ERROR":void 0;throw S.onError(P,R),m}finally{e.isRunning=!1,e.isPaused=!1,e.shouldStop=!1,I(),x&&x()}},u=()=>{!e.isRunning||e.isPaused||(e.isPaused=!0,I())},y=()=>{!e.isRunning||!e.isPaused||(e.isPaused=!1,K())},b=()=>{e.isRunning&&(e.shouldStop=!0,e.isPaused=!1,K())},v=()=>{e.bufferPool?.dispose(),e.configBuffer?.destroy(),e.configBuffer=null,e.configData=null,e.pipelines=null,e.bindGroupLayouts=null,e.bufferPool=null,e.planner=null,e.deviceContext=null,e.targetBuffer?.destroy(),e.targetBuffer=null,e.targetBufferCapacity=0},C=async(o,p,g)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.planner||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const w=e.deviceContext.getDevice().queue,E=e.bufferPool.slotCount,x=Array.from({length:E},(l,A)=>E-1-A),m=[],P=[],R=[],V=l=>{R.push(l)},z=()=>new Promise(l=>{if(x.length>0){const A=x.pop();l(A);return}m.push(l)}),k=l=>{const A=m.shift();if(A){A(l);return}x.push(l)};let h=0;for(const l of o.segments){if(e.shouldStop)break;const A=await d("planner.computePlan",{segmentIndex:l.index,totalMessages:l.totalMessages},()=>Promise.resolve(e.planner.computePlan(l.totalMessages)));for(const F of A.dispatches){if(e.shouldStop||(await U(),e.shouldStop))break;const G=await z();if(e.shouldStop){k(G);break}const q={segment:l,dispatchIndex:h,messageCount:F.messageCount,slotIndex:G},H=M(q,F.baseOffset,o,p,g,w,k,V);P.push(H),h+=1}}P.length>0&&await Promise.all(P),R.length>0&&await Promise.all(R)},M=async(o,p,g,S,w,E,x,m)=>{if(!e.deviceContext||!e.pipelines||!e.bufferPool||!e.configBuffer||!e.configData||!e.targetBuffer||!e.bindGroupLayouts)throw new Error("WebGPU runner is not ready");const P=e.deviceContext.getDevice(),R=e.configBuffer,V=e.configData,z=e.bindGroupLayouts,k=e.pipelines,h=e.targetBuffer,l=e.bufferPool.acquire(o.slotIndex,o.messageCount);let A=!1,F=!1;const G=()=>{A||(A=!0,x(o.slotIndex))},q=Math.ceil(o.messageCount/e.workgroupSize),H=Math.max(1,Math.ceil(l.candidateCapacity/e.workgroupSize)),$=Y(se*Uint32Array.BYTES_PER_ELEMENT),L={dispatchIndex:o.dispatchIndex,messageCount:o.messageCount,slotIndex:o.slotIndex,workgroupCount:q,scatterWorkgroupCount:H,candidateCapacity:l.candidateCapacity,groupCount:l.groupCount,segmentIndex:o.segment.index,segmentBaseOffset:p};try{await d("dispatch",L,async()=>{E.writeBuffer(l.output,0,Re.buffer,Re.byteOffset,Re.byteLength),W(o.segment,p,o.messageCount,l.groupCount,l.candidateCapacity),E.writeBuffer(R,0,V.buffer,V.byteOffset,V.byteLength);const re=P.createBindGroup({label:`gpu-seed-generate-group-${o.dispatchIndex}`,layout:z.generate,entries:[{binding:0,resource:{buffer:R}},{binding:1,resource:{buffer:h}},{binding:2,resource:{buffer:l.candidate}},{binding:3,resource:{buffer:l.groupCounts}}]}),fe=P.createBindGroup({label:`gpu-seed-scan-group-${o.dispatchIndex}`,layout:z.scan,entries:[{binding:0,resource:{buffer:R}},{binding:3,resource:{buffer:l.groupCounts}},{binding:4,resource:{buffer:l.groupOffsets}},{binding:5,resource:{buffer:l.output}}]}),he=P.createBindGroup({label:`gpu-seed-scatter-group-${o.dispatchIndex}`,layout:z.scatter,entries:[{binding:0,resource:{buffer:R}},{binding:2,resource:{buffer:l.candidate}},{binding:3,resource:{buffer:l.groupCounts}},{binding:4,resource:{buffer:l.groupOffsets}},{binding:5,resource:{buffer:l.output}}]}),Z=P.createCommandEncoder({label:`gpu-seed-compute-${o.dispatchIndex}`}),ee=Z.beginComputePass({label:`gpu-seed-generate-pass-${o.dispatchIndex}`});ee.setPipeline(k.generate),ee.setBindGroup(0,re),ee.dispatchWorkgroups(q),ee.end();const te=Z.beginComputePass({label:`gpu-seed-scan-pass-${o.dispatchIndex}`});te.setPipeline(k.scan),te.setBindGroup(0,fe),te.dispatchWorkgroups(1),te.end();const ne=Z.beginComputePass({label:`gpu-seed-scatter-pass-${o.dispatchIndex}`});ne.setPipeline(k.scatter),ne.setBindGroup(0,he),ne.dispatchWorkgroups(H),ne.end(),Z.copyBufferToBuffer(l.output,0,l.matchCount,0,$);const ge=Z.finish();await d("dispatch.submit",{...L},async()=>{await d("dispatch.submit.encode",{...L},async()=>{E.submit([ge])})});const ye=await d("dispatch.mapMatchCount",{...L,headerCopyBytes:$},async()=>{await l.matchCount.mapAsync(GPUMapMode.READ,0,$);const oe=new Uint32Array(l.matchCount.getMappedRange(0,$))[0]??0;return l.matchCount.unmap(),oe}),Se=Math.min(ye,l.maxRecords)*le*Uint32Array.BYTES_PER_ELEMENT,j=Y(se*Uint32Array.BYTES_PER_ELEMENT+Se);await d("dispatch.copyResults",{...L,totalCopyBytes:j},async()=>{const ae=P.createCommandEncoder({label:`gpu-seed-copy-${o.dispatchIndex}`});ae.copyBufferToBuffer(l.output,0,l.readback,0,j);const oe=ae.finish();await d("dispatch.copyResults.encode",{...L,totalCopyBytes:j},async()=>{E.submit([oe])})});const Fe=(async()=>{try{const{results:ae,clampedMatchCount:oe}=await d("dispatch.mapResults",{...L,totalCopyBytes:j},async()=>{await l.readback.mapAsync(GPUMapMode.READ,0,j);const Ie=l.readback.getMappedRange(0,j),we=new Uint32Array(Ie),He=we[0]??0,Ne=Math.max(0,Math.floor((we.length-se)/le)),Pe=Math.min(He,l.maxRecords,Ne),Ce=se+Pe*le,Me=new Uint32Array(Ce);return Me.set(we.subarray(0,Ce)),l.readback.unmap(),{results:Me,clampedMatchCount:Pe}});try{G(),await d("dispatch.processMatches",{...L,matchCount:oe},()=>T(ae,oe,o,p,g,S,w))}finally{G()}}catch(ae){throw G(),ae}})();F=!0,m(Fe)})}finally{F||G()}},T=async(o,p,g,S,w,E,x)=>{const m=g.segment,P=m.rangeSeconds,R=Math.max(P,1),V=Math.max(m.config.vcountCount,1),z=R,k=z*V,h=m.config.timer0Min,l=m.config.vcountMin,A=S;for(let F=0;F<p&&!(e.shouldStop||F%Rt===0&&(await U(),e.shouldStop));F+=1){const G=se+F*le,q=o[G],H=A+q,$=o[G+1]>>>0,L=Math.floor(H/k),re=H-L*k,fe=Math.floor(re/z),he=re-fe*z,Z=h+L,ee=l+fe,te=new Date(w.startTimestampMs+he*1e3),ne=g.segment.keyCode,ge=e.seedCalculator.generateMessage(w.conditions,Z,ee,te,ne),{hash:ye,seed:be,lcgSeed:Se}=e.seedCalculator.calculateSeed(ge);be!==$&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:$,cpuSeed:be,messageIndex:H});const j={seed:$,datetime:te,timer0:Z,vcount:ee,keyCode:ne,conditions:w.conditions,message:ge,sha1Hash:ye,lcgSeed:Se,isMatch:!0};x.onResult(j),E.matchesFound+=1}if(g.messageCount>0){const F=g.messageCount-1,G=A+F,q=Math.floor(G/k),H=G-q*k,$=Math.floor(H/z),L=H-$*z,re=new Date(w.startTimestampMs+L*1e3).toISOString();E.currentDateTime=re}E.currentStep+=g.messageCount,B(E,x)},B=(o,p)=>{const g=Date.now();if(g-e.lastProgressUpdateMs<wt&&o.currentStep<o.totalSteps)return;const S=N(),w=_(o.currentStep,o.totalSteps,S);p.onProgress({currentStep:o.currentStep,totalSteps:o.totalSteps,elapsedTime:S,estimatedTimeRemaining:w,matchesFound:o.matchesFound,currentDateTime:o.currentDateTime}),e.lastProgressUpdateMs=g},_=(o,p,g)=>{if(o===0||o>=p)return 0;const S=g/o,w=p-o;return Math.round(S*w)},W=(o,p,g,S,w)=>{if(!e.configData)throw new Error("config buffer not prepared");const E=Math.max(o.config.rangeSeconds,1),x=Math.max(o.config.vcountCount,1),m=E,P=m*x,R=Math.floor(p/P),V=p-R*P,z=Math.floor(V/m),k=V-z*m,h=e.configData;h[0]=g>>>0,h[1]=R>>>0,h[2]=z>>>0,h[3]=k>>>0,h[4]=o.config.rangeSeconds>>>0,h[5]=o.config.timer0Min>>>0,h[6]=o.config.timer0Count>>>0,h[7]=o.config.vcountMin>>>0,h[8]=o.config.vcountCount>>>0,h[9]=o.config.startSecondOfDay>>>0,h[10]=o.config.startDayOfWeek>>>0,h[11]=o.config.macLower>>>0,h[12]=o.config.data7Swapped>>>0,h[13]=o.config.keyInputSwapped>>>0,h[14]=o.config.hardwareType>>>0;for(let l=0;l<o.config.nazoSwapped.length;l+=1)h[15+l]=o.config.nazoSwapped[l]>>>0;h[20]=o.config.startYear>>>0,h[21]=o.config.startDayOfYear>>>0,h[22]=S>>>0,h[23]=e.workgroupSize>>>0,h[24]=w>>>0},ue=()=>{e.timerState.cumulativeRunTime=0,e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1},I=()=>{e.timerState.isPaused||(e.timerState.cumulativeRunTime+=Date.now()-e.timerState.segmentStartTime,e.timerState.isPaused=!0)},K=()=>{e.timerState.isPaused&&(e.timerState.segmentStartTime=Date.now(),e.timerState.isPaused=!1)},N=()=>e.timerState.isPaused?e.timerState.cumulativeRunTime:e.timerState.cumulativeRunTime+(Date.now()-e.timerState.segmentStartTime),U=async()=>{for(;e.isPaused&&!e.shouldStop;)await Q(25)},Q=o=>new Promise(p=>setTimeout(p,o)),Y=o=>Math.ceil(o/256)*256;return{init:c,run:f,pause:u,resume:y,stop:b,dispose:v}}const We=self,D={isRunning:!1,isPaused:!1},_e=Et();let pe=null;function O(t){We.postMessage(t)}function Pt(){O({type:"READY",message:"WebGPU worker initialized"})}function ce(){D.isRunning=!1,D.isPaused=!1,pe=null}function Ct(){return vt()?!0:(O({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Mt(t){if(D.isRunning){O({type:"ERROR",error:"Search is already running"});return}if(!t.conditions||!t.targetSeeds){O({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!Ct())return;D.isRunning=!0,D.isPaused=!1;let n;try{n=Qe(t.conditions)}catch(s){ce();const e=s instanceof Error?s.message:"検索条件の解析中にエラーが発生しました";O({type:"ERROR",error:e,errorCode:"WEBGPU_CONTEXT_ERROR"});return}pe=new AbortController;const a={onProgress:s=>{O({type:"PROGRESS",progress:s})},onResult:s=>{O({type:"RESULT",result:s})},onComplete:s=>{ce(),O({type:"COMPLETE",message:s})},onError:(s,e)=>{ce(),O({type:"ERROR",error:s,errorCode:e})},onPaused:()=>{D.isPaused=!0,O({type:"PAUSED"})},onResumed:()=>{D.isPaused=!1,O({type:"RESUMED"})},onStopped:(s,e)=>{ce(),O({type:"STOPPED",message:s,progress:e})}},i={context:n,targetSeeds:t.targetSeeds,callbacks:a,signal:pe.signal};try{await _e.run(i)}catch(s){if(!D.isRunning)return;ce();const e=s instanceof Error?s.message:"WebGPU search failed with unknown error";O({type:"ERROR",error:e,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function Tt(){!D.isRunning||D.isPaused||(_e.pause(),D.isPaused=!0,O({type:"PAUSED"}))}function xt(){!D.isRunning||!D.isPaused||(_e.resume(),D.isPaused=!1,O({type:"RESUMED"}))}function Ot(){D.isRunning&&(_e.stop(),pe?.abort())}Pt();We.onmessage=t=>{const n=t.data;switch(n.type){case"START_SEARCH":Mt(n);break;case"PAUSE_SEARCH":Tt();break;case"RESUME_SEARCH":xt();break;case"STOP_SEARCH":Ot();break;default:O({type:"ERROR",error:`Unknown request type: ${n.type}`})}};
