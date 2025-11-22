const _e={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},Se=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],B=Se.reduce((e,[t,r])=>(e[t]=r,e),{}),ke=Se.length,Le=(1<<ke)-1,ye=12287,Ue=[1<<B["[↑]"]|1<<B["[↓]"],1<<B["[←]"]|1<<B["[→]"],1<<B.Select|1<<B.Start|1<<B.L|1<<B.R];function we(e,t){return Number.isFinite(e)?e&Le:0}function Fe(e){const t=we(e);return ye^t}function We(e){const t=we(e);for(const r of Ue)if((t&r)===r)return!0;return!1}function Ie(e){return Fe(e)}const j=1e3,ie=60,xe=60,Re=24,be=ie*xe,ze=be*Re,Ee=ze*j;function Ge(e){const t=e.timeRange;if(!t)throw new Error("timeRange is required for seed search");const r=ae("hour",t.hour,0,Re-1),n=ae("minute",t.minute,0,xe-1),a=ae("second",t.second,0,ie-1),o=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,0,0,0),d=new Date(e.dateRange.endYear,e.dateRange.endMonth-1,e.dateRange.endDay,0,0,0),i=o.getTime(),c=d.getTime();if(i>c)throw new Error("開始日が終了日より後に設定されています");const l=Math.floor((c-i)/Ee)+1;if(l<=0)throw new Error("探索日数が検出できませんでした");const g=r.count*n.count*a.count;if(g<=0)throw new Error("時刻レンジの組み合わせ数が0です");const _=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,r.start,n.start,a.start,0);return{plan:{dayCount:l,combosPerDay:g,hourRangeStart:r.start,hourRangeCount:r.count,minuteRangeStart:n.start,minuteRangeCount:n.count,secondRangeStart:a.start,secondRangeCount:a.count,startDayTimestampMs:i},firstCombinationDate:_}}function ce(e,t){const r=Math.max(e.minuteRangeCount,1),n=Math.max(e.secondRangeCount,1),a=Math.max(e.combosPerDay,1),o=Math.max(0,Math.trunc(t)),d=Math.floor(o/a),i=o-d*a,c=r*n,l=Math.floor(i/c),g=i-l*c,_=Math.floor(g/n),w=g-_*n,b=e.hourRangeStart+l,P=e.minuteRangeStart+_,C=e.secondRangeStart+w,v=e.startDayTimestampMs+d*Ee+b*be*j+P*ie*j+C*j;return new Date(v)}function ae(e,t,r,n){if(!t)throw new Error(`${e} range is required for WebGPU search`);const a=Math.trunc(t.start),o=Math.trunc(t.end);if(Number.isNaN(a)||Number.isNaN(o))throw new Error(`${e} range must be numeric`);if(a<r||o>n)throw new Error(`${e} range must be within ${r} to ${n}`);if(a>o)throw new Error(`${e} range start must be <= end`);return{start:a,end:o,count:o-a+1}}const Ne=100663296,Ye=4294967295;function He(e,t=[],r){const{plan:n,firstCombinationDate:a}=Ge(e),o=qe(r),d=Xe(t),i=Je(e,n,a),c=Ke(i,o),l=c.reduce((g,_)=>g+_.messageCount,0);return{segments:c,targetSeeds:d,timePlan:n,summary:{totalMessages:l,totalSegments:c.length,targetSeedCount:d.length,rangeSeconds:i.rangeSeconds},limits:o,conditions:e}}function Ke(e,t){const r=[];if(e.rangeSeconds<=0)return r;const n=Math.max(1,t.workgroupSize*t.maxWorkgroupsPerDispatch),a=Math.min(t.maxMessagesPerDispatch,n),o={startDayOfWeek:e.startDayOfWeek,macLower:e.macLower,data7Swapped:e.data7Swapped,hardwareType:e.hardwareType,nazoSwapped:e.nazoSwapped,startYear:e.startYear,startDayOfYear:e.startDayOfYear,hourRangeStart:e.hourRangeStart,hourRangeCount:e.hourRangeCount,minuteRangeStart:e.minuteRangeStart,minuteRangeCount:e.minuteRangeCount,secondRangeStart:e.secondRangeStart,secondRangeCount:e.secondRangeCount};let d=0,i=0;for(const c of e.keyCodes){const l=ee(c>>>0);for(const g of e.timer0Segments)for(let _=g.timer0Min;_<=g.timer0Max;_+=1){const w=g.vcount;let b=e.rangeSeconds,P=0;const C=ee((w&65535)<<16|_&65535);for(;b>0;){const E=Math.min(b,a),v=$e(E,t),s=()=>Ze({...o,timer0VcountSwapped:C,keyInputSwapped:l});r.push({id:`seg-${i}`,keyCode:c,timer0:_,vcount:w,messageCount:E,baseSecondOffset:P,globalMessageOffset:d,workgroupCount:v,getUniformWords:s}),b-=E,P+=E,d+=E,i+=1}}}return r}function $e(e,t){const r=Math.max(1,Math.ceil(e/t.workgroupSize)),n=Math.max(1,t.maxWorkgroupsPerDispatch);return Math.min(r,n)}function qe(e){if(!e?.limits)throw new Error("Seed search job limits are required for WebGPU execution");return Ve(e.limits)}function Ve(e){const t=N(e.workgroupSize,"workgroupSize"),r=N(e.maxWorkgroupsPerDispatch,"maxWorkgroupsPerDispatch"),n=N(e.candidateCapacityPerDispatch,"candidateCapacityPerDispatch"),a=N(e.maxMessagesPerDispatch,"maxMessagesPerDispatch"),o=N(e.maxDispatchesInFlight,"maxDispatchesInFlight"),d=Math.max(1,Math.floor(Ye/Math.max(1,t))),i=Math.min(r,d),c=Math.max(1,t*i),l=Math.min(a,c);return{workgroupSize:t,maxWorkgroupsPerDispatch:i,candidateCapacityPerDispatch:n,maxMessagesPerDispatch:l,maxDispatchesInFlight:o}}function Je(e,t,r){const n=Qe(e),a=et(e.keyInput);if(a.length===0)throw new Error("入力されたキー条件から生成できる組み合わせがありません");const o=tt(e,n);if(o.length===0)throw new Error("timer0の範囲が正しく設定されていません");const d=st(n.nazo),{macLower:i,data7Swapped:c}=nt(e.macAddress,je[e.hardware]),l=t.dayCount*t.combosPerDay;if(l<=0)throw new Error("探索対象の秒数が0以下です");return{rangeSeconds:l,timer0Segments:o,keyCodes:a,nazoSwapped:d,macLower:i,data7Swapped:c,hardwareType:ot(e.hardware),startYear:r.getFullYear(),startDayOfYear:ut(r),startDayOfWeek:r.getDay(),hourRangeStart:t.hourRangeStart,hourRangeCount:t.hourRangeCount,minuteRangeStart:t.minuteRangeStart,minuteRangeCount:t.minuteRangeCount,secondRangeStart:t.secondRangeStart,secondRangeCount:t.secondRangeCount}}function Xe(e){if(!e||e.length===0)return new Uint32Array(0);const t=[];for(const r of e)typeof r!="number"||!Number.isFinite(r)||t.push(r>>>0);return Uint32Array.from(t)}function N(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function Ze(e){const t=new Uint32Array(20);return t[0]=e.timer0VcountSwapped>>>0,t[1]=e.macLower>>>0,t[2]=e.data7Swapped>>>0,t[3]=e.keyInputSwapped>>>0,t[4]=e.hardwareType>>>0,t[5]=e.startYear>>>0,t[6]=e.startDayOfYear>>>0,t[7]=e.startDayOfWeek>>>0,t[8]=e.hourRangeStart>>>0,t[9]=e.hourRangeCount>>>0,t[10]=e.minuteRangeStart>>>0,t[11]=e.minuteRangeCount>>>0,t[12]=e.secondRangeStart>>>0,t[13]=e.secondRangeCount>>>0,t[14]=(e.nazoSwapped[0]??0)>>>0,t[15]=(e.nazoSwapped[1]??0)>>>0,t[16]=(e.nazoSwapped[2]??0)>>>0,t[17]=(e.nazoSwapped[3]??0)>>>0,t[18]=(e.nazoSwapped[4]??0)>>>0,t[19]=0,t}const je={DS:8,DS_LITE:6,"3DS":9};function Qe(e){const t=_e[e.romVersion];if(!t)throw new Error(`ROMバージョン ${e.romVersion} は未対応です`);const r=t[e.romRegion];if(!r)throw new Error(`ROMリージョン ${e.romRegion} は未対応です`);return{nazo:[...r.nazo],vcountTimerRanges:r.vcountTimerRanges.map(n=>[...n])}}function et(e){const t=[];for(let a=0;a<12;a+=1)(e&1<<a)!==0&&t.push(a);const r=[],n=1<<t.length;for(let a=0;a<n;a+=1){let o=0;for(let d=0;d<t.length;d+=1)(a&1<<d)!==0&&(o|=1<<t[d]);We(o)||r.push((o^ye)>>>0)}return r}function tt(e,t){const r=[],{timer0VCountConfig:{useAutoConfiguration:n,timer0Range:{min:a,max:o},vcountRange:{min:d,max:i}}}=e;if(!n){for(let l=d;l<=i;l+=1)r.push({timer0Min:a,timer0Max:o,vcount:l});return r}let c=null;for(let l=a;l<=o;l+=1){const g=rt(t,l);c&&c.vcount===g&&l===c.timer0Max+1?c.timer0Max=l:(c&&r.push(c),c={timer0Min:l,timer0Max:l,vcount:g})}return c&&r.push(c),r}function rt(e,t){for(const[r,n,a]of e.vcountTimerRanges)if(t>=n&&t<=a)return r;return e.vcountTimerRanges.length>0?e.vcountTimerRanges[0][0]:96}function nt(e,t){const r=at(e),n=(r[4]&255)<<8|r[5]&255,o=((r[0]&255|(r[1]&255)<<8|(r[2]&255)<<16|(r[3]&255)<<24)^Ne^t)>>>0;return{macLower:n,data7Swapped:ee(o)}}function at(e){const t=new Array(6).fill(0);for(let r=0;r<6;r+=1){const n=e[r]??0;t[r]=(Number(n)&255)>>>0}return t}function ot(e){switch(e){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function ee(e){return((e&255)<<24|(e>>>8&255)<<16|(e>>>16&255)<<8|e>>>24&255)>>>0}function st(e){const t=new Uint32Array(e.length);for(let r=0;r<e.length;r+=1)t[r]=ee(e[r]>>>0);return t}function ut(e){const t=new Date(e.getFullYear(),0,1),r=e.getTime()-t.getTime();return Math.floor(r/(1440*60*1e3))+1}class de{calculateHash(t){if(t.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const r=1732584193,n=4023233417,a=2562383102,o=271733878,d=3285377520,i=new Array(80);for(let s=0;s<16;s++)i[s]=t[s];for(let s=16;s<80;s++)i[s]=this.leftRotate(i[s-3]^i[s-8]^i[s-14]^i[s-16],1);let c=r,l=n,g=a,_=o,w=d;for(let s=0;s<80;s++){let p;s<20?p=this.leftRotate(c,5)+(l&g|~l&_)+w+i[s]+1518500249&4294967295:s<40?p=this.leftRotate(c,5)+(l^g^_)+w+i[s]+1859775393&4294967295:s<60?p=this.leftRotate(c,5)+(l&g|l&_|g&_)+w+i[s]+2400959708&4294967295:p=this.leftRotate(c,5)+(l^g^_)+w+i[s]+3395469782&4294967295,w=_,_=g,g=this.leftRotate(l,30),l=c,c=p}const b=this.add32(r,c),P=this.add32(n,l),C=this.add32(a,g),E=this.add32(o,_),v=this.add32(d,w);return{h0:b,h1:P,h2:C,h3:E,h4:v}}leftRotate(t,r){return(t<<r|t>>>32-r)>>>0}add32(t,r){return(t+r&4294967295)>>>0}static hashToHex(t,r,n,a,o){return t.toString(16).padStart(8,"0")+r.toString(16).padStart(8,"0")+n.toString(16).padStart(8,"0")+a.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")}}let k=null,Y=null;async function it(){return k||Y||(Y=(async()=>{try{const e=await import("./wasm_pkg-DRWLiY4b.js");let t;if(typeof process<"u"&&!!process.versions?.node){const n=await import("./__vite-browser-external-9wXp6ZBx.js"),o=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");t={module_or_path:n.readFileSync(o)}}else t={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-D27IxIOn.wasm",import.meta.url)};return await e.default(t),k={IntegratedSeedSearcher:e.IntegratedSeedSearcher,BWGenerationConfig:e.BWGenerationConfig,PokemonGenerator:e.PokemonGenerator,SeedEnumerator:e.SeedEnumerator,EncounterType:e.EncounterType,GameVersion:e.GameVersion,GameMode:e.GameMode,calculate_game_offset:e.calculate_game_offset,sha1_hash_batch:e.sha1_hash_batch},k}catch(e){throw console.error("Failed to load WebAssembly module:",e),k=null,Y=null,e}})(),Y)}function ct(){if(!k)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return k}function le(){return k!==null}const dt={DS:8,DS_LITE:6,"3DS":9};class lt{sha1;useWasm=!1;constructor(){this.sha1=new de}async initializeWasm(){try{return await it(),this.useWasm=!0,!0}catch(t){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",t),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&le()}getWasmModule(){return ct()}setUseWasm(t){if(t&&!le()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=t}getROMParameters(t,r){const n=_e[t];if(!n)return console.error(`ROM version not found: ${t}`),null;const a=n[r];return a?{nazo:[...a.nazo],vcountTimerRanges:a.vcountTimerRanges.map(o=>[...o])}:(console.error(`ROM region not found: ${r} for version ${t}`),null)}toLittleEndian32(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}toLittleEndian16(t){return(t&255)<<8|t>>>8&255}getDayOfWeek(t,r,n){return new Date(t,r-1,n).getDay()}generateMessage(t,r,n,a,o){const d=this.getROMParameters(t.romVersion,t.romRegion);if(!d)throw new Error(`No parameters found for ${t.romVersion} ${t.romRegion}`);const i=new Array(16).fill(0);for(let R=0;R<5;R++)i[R]=this.toLittleEndian32(d.nazo[R]);i[5]=this.toLittleEndian32(n<<16|r);const c=t.macAddress[4]<<8|t.macAddress[5];i[6]=c;const l=t.macAddress[0]<<0|t.macAddress[1]<<8|t.macAddress[2]<<16|t.macAddress[3]<<24,g=100663296,_=dt[t.hardware],w=l^g^_;i[7]=this.toLittleEndian32(w);const b=a.getFullYear()%100,P=a.getMonth()+1,C=a.getDate(),E=this.getDayOfWeek(a.getFullYear(),P,C),v=Math.floor(b/10)*16+b%10,s=Math.floor(P/10)*16+P%10,p=Math.floor(C/10)*16+C%10,u=Math.floor(E/10)*16+E%10;i[8]=v<<24|s<<16|p<<8|u;const f=a.getHours(),h=a.getMinutes(),m=a.getSeconds(),S=(t.hardware==="DS"||t.hardware==="DS_LITE")&&f>=12?1:0,y=Math.floor(f/10)*16+f%10,x=Math.floor(h/10)*16+h%10,O=Math.floor(m/10)*16+m%10;i[9]=S<<30|y<<24|x<<16|O<<8|0,i[10]=0,i[11]=0;const D=o??Ie(t.keyInput);return i[12]=this.toLittleEndian32(D),i[13]=2147483648,i[14]=0,i[15]=416,i}calculateSeed(t){const r=this.sha1.calculateHash(t),n=BigInt(this.toLittleEndian32(r.h0)),o=BigInt(this.toLittleEndian32(r.h1))<<32n|n,c=o*0x5D588B656C078965n+0x269EC3n;return{seed:Number(c>>32n&0xFFFFFFFFn),hash:de.hashToHex(r.h0,r.h1,r.h2,r.h3,r.h4),lcgSeed:o}}parseTargetSeeds(t){const r=t.split(`
`).map(d=>d.trim()).filter(d=>d.length>0),n=[],a=[],o=new Set;return r.forEach((d,i)=>{try{let c=d.toLowerCase();if(c.startsWith("0x")&&(c=c.substring(2)),!/^[0-9a-f]{1,8}$/.test(c)){a.push({line:i+1,value:d,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const l=parseInt(c,16);if(o.has(l))return;o.add(l),n.push(l)}catch(c){const l=c instanceof Error?c.message:String(c);a.push({line:i+1,value:d,error:l||"Failed to parse as hexadecimal number."})}}),{validSeeds:n,errors:a}}getVCountForTimer0(t,r){for(const[n,a,o]of t.vcountTimerRanges)if(r>=a&&r<=o)return n;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}}const $=2,Q=1,fe={requiredFeatures:[],powerPreference:"high-performance"},Pe={workgroupSize:256,candidateCapacityPerDispatch:4096},ft=$*Uint32Array.BYTES_PER_ELEMENT,mt=4294967295,me={mobile:1,integrated:2,discrete:4,unknown:1},pt=1,pe=8;function Ce(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}const gt=Ce;async function De(e){if(!Ce())throw new Error("WebGPU is not available in this environment");const r=await navigator.gpu.requestAdapter({powerPreference:fe.powerPreference});if(!r)throw new Error("Failed to acquire WebGPU adapter");const n={requiredFeatures:fe.requiredFeatures,requiredLimits:e?.requiredLimits,label:"seed-search-device"},[a,o]=await Promise.all([r.requestDevice(n),yt(r)]);let d=!1;const i=a.lost.then(g=>(d=!0,console.warn("[webgpu] device lost:",g.message),g)),c=ht(r,a),l=a.limits;return{getAdapter:()=>r,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>l,getCapabilities:()=>c,getGpuProfile:()=>o,deriveSearchJobLimits:g=>_t(c.limits,o,g),isLost:()=>d,waitForLoss:()=>i,getSupportedWorkgroupSize:g=>Me(c.limits,g)}}function ht(e,t){const r=new Set;return e.features.forEach(n=>r.add(n)),{limits:t.limits,features:r}}function _t(e,t,r){const n={...Pe,...r},a=Tt(n),o=Me(e,a.workgroupSize),d=te(e.maxComputeWorkgroupsPerDimension),i=a.maxWorkgroupsPerDispatch??d,c=Math.max(1,Math.floor(mt/Math.max(1,o))),l=q(Math.min(i,d,c),"maxWorkgroupsPerDispatch"),g=o*l,_=a.maxMessagesPerDispatch??g,w=q(Math.min(_,g),"maxMessagesPerDispatch"),b=Math.max(1,Math.floor(te(e.maxStorageBufferBindingSize)/ft)),P=a.candidateCapacityPerDispatch??b,C=q(Math.min(P,b),"candidateCapacityPerDispatch"),E=St(t,a);return{workgroupSize:o,maxWorkgroupsPerDispatch:l,maxMessagesPerDispatch:w,candidateCapacityPerDispatch:C,maxDispatchesInFlight:E}}function Me(e,t){const r=Pe.workgroupSize,n=typeof t=="number"&&Number.isFinite(t)&&t>0?Math.floor(t):r,a=te(e.maxComputeWorkgroupSizeX),o=te(e.maxComputeInvocationsPerWorkgroup),d=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(n,d))}function te(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function q(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function St(e,t){if(typeof t.maxDispatchesInFlight=="number")return q(Math.min(t.maxDispatchesInFlight,pe),"maxDispatchesInFlight");const r=e.isFallbackAdapter?pt:me[e.kind]??me.unknown;return q(Math.min(r,pe),"maxDispatchesInFlight")}async function yt(e){const t=wt(),n=!!e.isFallbackAdapter,a=Pt();if(a){const o={description:a.renderer};return{kind:a.kind,source:"webgl",userAgent:t,adapterInfo:o,isFallbackAdapter:n}}return n?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}}function wt(){return typeof navigator>"u"?"":navigator.userAgent||""}const xt=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],Rt=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"],bt=["intel","iris","uhd","hd graphics","radeon graphics","apple"];function oe(e,t){return t.some(r=>e.includes(r))}function Et(e){if(!e)return;const t=e.toLowerCase();if(oe(t,xt))return"mobile";if(oe(t,Rt))return"discrete";if(oe(t,bt))return"integrated"}function Pt(){const e=Ct();if(!e)return;const t=Et(e);if(t)return{kind:t,renderer:e}}function Ct(){const e=Dt();if(e)try{const t=Mt(e);if(!t)return;const r=t.getExtension("WEBGL_debug_renderer_info");if(!r)return;const n=t.getParameter(r.UNMASKED_RENDERER_WEBGL),a=t.getExtension("WEBGL_lose_context");return a&&a.loseContext(),typeof n=="string"?n:void 0}catch(t){console.warn("[webgpu] webgl renderer detection failed:",t);return}}function Dt(){if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);if(typeof document<"u"&&typeof document.createElement=="function"){const e=document.createElement("canvas");return e.width=1,e.height=1,e}}function Mt(e){const t=e,r=t.getContext;if(typeof r!="function")return null;const n=a=>r.call(t,a)??null;return n("webgl2")??n("webgl")}function Tt(e,t){return e}var vt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
\r
struct DispatchState {\r
  message_count : u32,\r
  base_second_offset : u32,\r
  candidate_capacity : u32,\r
  padding : u32,\r
};\r
\r
struct SearchConstants {\r
  timer0_vcount_swapped : u32,\r
  mac_lower : u32,\r
  data7_swapped : u32,\r
  key_input_swapped : u32,\r
  hardware_type : u32,\r
  start_year : u32,\r
  start_day_of_year : u32,\r
  start_day_of_week : u32,\r
  hour_range_start : u32,\r
  hour_range_count : u32,\r
  minute_range_start : u32,\r
  minute_range_count : u32,\r
  second_range_start : u32,\r
  second_range_count : u32,\r
  nazo0 : u32,\r
  nazo1 : u32,\r
  nazo2 : u32,\r
  nazo3 : u32,\r
  nazo4 : u32,\r
  reserved0 : u32,\r
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
const BCD_LOOKUP : array<u32, 100> = array<u32, 100>(\r
  0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u, 0x09u,\r
  0x10u, 0x11u, 0x12u, 0x13u, 0x14u, 0x15u, 0x16u, 0x17u, 0x18u, 0x19u,\r
  0x20u, 0x21u, 0x22u, 0x23u, 0x24u, 0x25u, 0x26u, 0x27u, 0x28u, 0x29u,\r
  0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u, 0x38u, 0x39u,\r
  0x40u, 0x41u, 0x42u, 0x43u, 0x44u, 0x45u, 0x46u, 0x47u, 0x48u, 0x49u,\r
  0x50u, 0x51u, 0x52u, 0x53u, 0x54u, 0x55u, 0x56u, 0x57u, 0x58u, 0x59u,\r
  0x60u, 0x61u, 0x62u, 0x63u, 0x64u, 0x65u, 0x66u, 0x67u, 0x68u, 0x69u,\r
  0x70u, 0x71u, 0x72u, 0x73u, 0x74u, 0x75u, 0x76u, 0x77u, 0x78u, 0x79u,\r
  0x80u, 0x81u, 0x82u, 0x83u, 0x84u, 0x85u, 0x86u, 0x87u, 0x88u, 0x89u,\r
  0x90u, 0x91u, 0x92u, 0x93u, 0x94u, 0x95u, 0x96u, 0x97u, 0x98u, 0x99u\r
);\r
\r
@group(0) @binding(0) var<storage, read> state : DispatchState;\r
@group(0) @binding(1) var<uniform> constants : SearchConstants;\r
@group(0) @binding(2) var<storage, read> target_seeds : TargetSeedBuffer;\r
@group(0) @binding(3) var<storage, read_write> output_buffer : MatchOutputBuffer;\r
\r
fn left_rotate(value : u32, amount : u32) -> u32 {\r
  return (value << amount) | (value >> (32u - amount));\r
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
  let le0 = ((h0 & 0x000000FFu) << 24u) |\r
    ((h0 & 0x0000FF00u) << 8u) |\r
    ((h0 & 0x00FF0000u) >> 8u) |\r
    ((h0 & 0xFF000000u) >> 24u);\r
  let le1 = ((h1 & 0x000000FFu) << 24u) |\r
    ((h1 & 0x0000FF00u) << 8u) |\r
    ((h1 & 0x00FF0000u) >> 8u) |\r
    ((h1 & 0xFF000000u) >> 24u);\r
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
  let is_active = global_linear_index < state.message_count;\r
  var seed : u32 = 0u;\r
  var matched = false;\r
\r
  if (is_active) {\r
    let safe_hour_count = max(constants.hour_range_count, 1u);\r
    let safe_minute_count = max(constants.minute_range_count, 1u);\r
    let safe_second_count = max(constants.second_range_count, 1u);\r
    let combos_per_day = safe_hour_count * safe_minute_count * safe_second_count;\r
    let total_second_offset = state.base_second_offset + global_linear_index;\r
\r
    let day_offset = total_second_offset / combos_per_day;\r
    let remainder_after_day = total_second_offset - day_offset * combos_per_day;\r
\r
    let entries_per_hour = safe_minute_count * safe_second_count;\r
    let hour_index = remainder_after_day / entries_per_hour;\r
    let remainder_after_hour = remainder_after_day - hour_index * entries_per_hour;\r
    let minute_index = remainder_after_hour / safe_second_count;\r
    let second_index = remainder_after_hour - minute_index * safe_second_count;\r
\r
    let hour = constants.hour_range_start + hour_index;\r
    let minute = constants.minute_range_start + minute_index;\r
    let second = constants.second_range_start + second_index;\r
\r
    var year = constants.start_year;\r
    var day_of_year = constants.start_day_of_year + day_offset;\r
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
    let day_of_week = (constants.start_day_of_week + day_offset) % 7u;\r
    let year_mod = year % 100u;\r
    let date_word = (BCD_LOOKUP[year_mod] << 24u) |\r
      (BCD_LOOKUP[month] << 16u) |\r
      (BCD_LOOKUP[day] << 8u) |\r
      BCD_LOOKUP[day_of_week];\r
    let is_pm = (constants.hardware_type <= 1u) && (hour >= 12u);\r
    let pm_flag = select(0u, 1u, is_pm);\r
    let time_word = (pm_flag << 30u) |\r
      (BCD_LOOKUP[hour] << 24u) |\r
      (BCD_LOOKUP[minute] << 16u) |\r
      (BCD_LOOKUP[second] << 8u);\r
\r
    var w : array<u32, 16>;\r
    w[0] = constants.nazo0;\r
    w[1] = constants.nazo1;\r
    w[2] = constants.nazo2;\r
    w[3] = constants.nazo3;\r
    w[4] = constants.nazo4;\r
    w[5] = constants.timer0_vcount_swapped;\r
    w[6] = constants.mac_lower;\r
    w[7] = constants.data7_swapped;\r
    w[8] = date_word;\r
    w[9] = time_word;\r
    w[10] = 0u;\r
    w[11] = 0u;\r
    w[12] = constants.key_input_swapped;\r
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
  if (record_index >= state.candidate_capacity) {\r
    atomicSub(&output_buffer.match_count, 1u);\r
    return;\r
  }\r
\r
  output_buffer.records[record_index].message_index = global_linear_index;\r
  output_buffer.records[record_index].seed = seed;\r
}`;const Ot=/WORKGROUP_SIZE_PLACEHOLDER/g,At="seed-search-kernel-module",Bt="seed-search-kernel",kt="seed-search-kernel-layout",Lt="seed-search-kernel-bind-layout";function Ut(e){return vt.replace(Ot,String(e))}function Ft(e,t){const r=e.createShaderModule({label:At,code:Ut(t)});r.getCompilationInfo?.().then(d=>{d.messages.length>0&&console.warn("[seed-search-kernel] compilation diagnostics",d.messages.map(i=>({message:i.message,line:i.lineNum,column:i.linePos,type:i.type})))}).catch(d=>{console.warn("[seed-search-kernel] compilation info failed",d)});const n=e.createBindGroupLayout({label:Lt,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),a=e.createPipelineLayout({label:kt,bindGroupLayouts:[n]});return{pipeline:e.createComputePipeline({label:Bt,layout:a,compute:{module:r,entryPoint:"sha1_generate"}}),layout:n}}const Wt=4,ge=256,se=new Uint32Array([0]);function Te(e,t){const r={context:t??null,pipeline:null,bindGroupLayout:null,targetBuffer:null,targetCapacity:0,workgroupSize:0,candidateCapacity:0,currentLimits:null,dispatchSlots:[],availableSlots:[],slotWaiters:[],desiredDispatchSlots:1},n=async(s,p)=>{r.context||(r.context=await De());const u=r.context.getDevice(),f=r.context.getSupportedWorkgroupSize(s.workgroupSize),h=Math.max(1,p?.dispatchSlots??r.desiredDispatchSlots??1),m=!r.currentLimits||r.workgroupSize!==f||r.candidateCapacity!==s.candidateCapacityPerDispatch;if(!r.pipeline||m){const{pipeline:y,layout:x}=Ft(u,f);r.pipeline=y,r.bindGroupLayout=x}r.workgroupSize=f,r.candidateCapacity=s.candidateCapacityPerDispatch,r.currentLimits=s,r.desiredDispatchSlots=h,a(u,h,s.candidateCapacityPerDispatch),r.currentLimits=s},a=(s,p,u)=>{for(const f of r.dispatchSlots)c(s,f,u);for(;r.dispatchSlots.length<p;){const f=r.dispatchSlots.length,h=o(s,f,u);r.dispatchSlots.push(h)}for(;r.dispatchSlots.length>p;){const f=r.dispatchSlots.pop();f&&l(f)}r.availableSlots=[...r.dispatchSlots],r.slotWaiters.length=0},o=(s,p,u)=>{const f=new Uint32Array(Wt),h=H(f.byteLength),m=s.createBuffer({label:`seed-search-dispatch-state-${p}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),{matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}=i(s,u,p);return{id:p,dispatchStateBuffer:m,dispatchStateData:f,uniformBuffer:null,uniformCapacityWords:0,matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}},d=(s,p,u)=>{const f=H(u*Uint32Array.BYTES_PER_ELEMENT);(!p.uniformBuffer||p.uniformCapacityWords<u)&&(p.uniformBuffer?.destroy(),p.uniformBuffer=s.createBuffer({label:`seed-search-uniform-${p.id}`,size:f,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),p.uniformCapacityWords=u)},i=(s,p,u)=>{const f=Q+p*$,h=H(f*Uint32Array.BYTES_PER_ELEMENT),m=s.createBuffer({label:`seed-search-output-${u}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),S=s.createBuffer({label:`seed-search-readback-${u}`,size:h,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return{matchOutputBuffer:m,readbackBuffer:S,matchBufferSize:h}},c=(s,p,u)=>{const f=Q+u*$,h=H(f*Uint32Array.BYTES_PER_ELEMENT);if(p.matchBufferSize===h)return;p.matchOutputBuffer.destroy(),p.readbackBuffer.destroy();const m=i(s,u,p.id);p.matchOutputBuffer=m.matchOutputBuffer,p.readbackBuffer=m.readbackBuffer,p.matchBufferSize=m.matchBufferSize},l=s=>{s.dispatchStateBuffer.destroy(),s.uniformBuffer?.destroy(),s.matchOutputBuffer.destroy(),s.readbackBuffer.destroy()},g=()=>r.availableSlots.length>0?Promise.resolve(r.availableSlots.pop()):new Promise(s=>{r.slotWaiters.push(s)}),_=s=>{const p=r.slotWaiters.shift();if(p){p(s);return}r.availableSlots.push(s)};return{ensureConfigured:n,setTargetSeeds:s=>{if(!r.context)throw new Error("SeedSearchEngine is not configured yet");const p=r.context.getDevice(),u=s.length,f=1+u,h=H(f*Uint32Array.BYTES_PER_ELEMENT);(!r.targetBuffer||r.targetCapacity<u)&&(r.targetBuffer?.destroy(),r.targetBuffer=p.createBuffer({label:"seed-search-target-seeds",size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.targetCapacity=u);const m=new Uint32Array(f);m[0]=u>>>0;for(let S=0;S<u;S+=1)m[1+S]=s[S]>>>0;p.queue.writeBuffer(r.targetBuffer,0,m.buffer,m.byteOffset,m.byteLength)},executeSegment:async s=>{if(!r.context||!r.pipeline||!r.bindGroupLayout)throw new Error("SeedSearchEngine is not ready");if(!r.targetBuffer)throw new Error("Target seed buffer is not prepared");if(r.dispatchSlots.length===0)throw new Error("Dispatch slots are not configured");const p=r.context.getDevice(),u=p.queue,f=Math.max(1,s.workgroupCount),h=f,m=await g();try{const S=X();u.writeBuffer(m.matchOutputBuffer,0,se.buffer,se.byteOffset,se.byteLength);const y=m.dispatchStateData;y[0]=s.messageCount>>>0,y[1]=s.baseSecondOffset>>>0,y[2]=r.candidateCapacity>>>0,y[3]=0,u.writeBuffer(m.dispatchStateBuffer,0,y.buffer,y.byteOffset,y.byteLength);const x=s.getUniformWords();d(p,m,x.length),u.writeBuffer(m.uniformBuffer,0,x.buffer,x.byteOffset,x.byteLength);const O=X(),D=p.createBindGroup({label:`seed-search-bind-group-${s.id}-slot-${m.id}`,layout:r.bindGroupLayout,entries:[{binding:0,resource:{buffer:m.dispatchStateBuffer}},{binding:1,resource:{buffer:m.uniformBuffer}},{binding:2,resource:{buffer:r.targetBuffer}},{binding:3,resource:{buffer:m.matchOutputBuffer}}]}),R=p.createCommandEncoder({label:`seed-search-encoder-${s.id}`}),A=R.beginComputePass({label:`seed-search-pass-${s.id}`});A.setPipeline(r.pipeline),A.setBindGroup(0,D),A.dispatchWorkgroups(f),A.end(),R.copyBufferToBuffer(m.matchOutputBuffer,0,m.readbackBuffer,0,m.matchBufferSize);const L=R.finish();u.submit([L]),await m.readbackBuffer.mapAsync(GPUMapMode.READ,0,m.matchBufferSize);const I=X(),U=m.readbackBuffer.getMappedRange(0,m.matchBufferSize),F=new Uint32Array(U.slice(0));m.readbackBuffer.unmap();const z=X(),V=F[0]??0,G=Math.min(V,r.candidateCapacity),ne=Math.min(F.length,Q+G*$),J={words:F.slice(0,ne),matchCount:G};return e?.onDispatchComplete?.({segmentId:s.id,messageCount:s.messageCount,workgroupCount:h,matchCount:G,candidateCapacity:r.candidateCapacity,timings:{totalMs:z-S,setupMs:O-S,gpuMs:I-O,readbackMs:z-I},timestampMs:z}),J}finally{_(m)}},dispose:()=>{for(const s of r.dispatchSlots)l(s);r.dispatchSlots=[],r.availableSlots=[],r.slotWaiters.length=0,r.targetBuffer?.destroy(),r.context=null,r.pipeline=null,r.bindGroupLayout=null,r.targetBuffer=null,r.targetCapacity=0,r.currentLimits=null},getWorkgroupSize:()=>r.workgroupSize,getCandidateCapacity:()=>r.candidateCapacity,getSupportedLimits:()=>r.context?.getLimits()??null}}function H(e){return Math.ceil(e/ge)*ge}function X(){return typeof performance<"u"?performance.now():Date.now()}const It=1024,zt=500;function Gt(e){const t=new lt,r=e??Te(),n={isRunning:!1,isPaused:!1,shouldStop:!1,job:null,progress:null,callbacks:null,timer:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1},lastProgressUpdate:0},a=async(u,f,h)=>{if(n.isRunning)throw new Error("Seed search is already running");n.isRunning=!0,n.isPaused=!1,n.shouldStop=h?.aborted??!1,n.job=u,n.callbacks=f,n.lastProgressUpdate=0,n.progress={currentStep:0,totalSteps:u.summary.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0,currentDateTime:u.timePlan?new Date(u.timePlan.startDayTimestampMs).toISOString():void 0},b(),_(!1);let m;if(h){const S=()=>{n.shouldStop=!0};h.addEventListener("abort",S),m=()=>h.removeEventListener("abort",S),n.abortCleanup=m}try{if(u.summary.totalMessages===0){f.onComplete("探索対象の組み合わせが存在しません");return}const S=Math.max(1,Math.min(u.limits.maxDispatchesInFlight??1,u.segments.length||1));await r.ensureConfigured(u.limits,{dispatchSlots:S}),r.setTargetSeeds(u.targetSeeds);const y=new Set,x=new Set,O=D=>{let R;R=(async()=>{if(n.shouldStop)return;const{words:L,matchCount:I}=await r.executeSegment(D);if(n.shouldStop)return;let U;U=(async()=>{await d(D,L,I)})().finally(()=>x.delete(U)),x.add(U)})().finally(()=>y.delete(R)),y.add(R)};for(const D of u.segments){if(n.shouldStop||(await g(),n.shouldStop))break;O(D),y.size>=S&&await Promise.race(y)}y.size>0&&await Promise.all(y),x.size>0&&await Promise.all(x),o()}catch(S){const y=S instanceof Error?S.message:"WebGPU検索中に不明なエラーが発生しました",x=globalThis.GPUValidationError,O=x&&S instanceof x?"WEBGPU_VALIDATION_ERROR":void 0;throw f.onError(y,O),S}finally{m?.(),n.abortCleanup=void 0,n.isRunning=!1,n.isPaused=!1,E(),n.job=null,n.callbacks=null,n.progress=null,n.shouldStop=!1,n.lastProgressUpdate=0}},o=()=>{const u=n.callbacks,f=n.progress;if(!(!u||!f)){if(w(f),n.shouldStop){u.onStopped("検索を停止しました",f);return}u.onProgress(f),u.onComplete(`検索が完了しました。${f.matchesFound}件ヒットしました。`)}},d=async(u,f,h)=>{const m=n.job,S=n.callbacks,y=n.progress;if(!m||!S||!y)return;const x=Q,O=$;for(let D=0;D<h&&!(n.shouldStop||D%It===0&&(await g(),n.shouldStop));D+=1){const R=x+D*O,A=f[R],L=f[R+1]>>>0,I=u.globalMessageOffset+A,U=u.baseSecondOffset+A,F=u.timer0,z=u.vcount,V=ce(m.timePlan,U),G=t.generateMessage(m.conditions,F,z,V,u.keyCode),{hash:ne,seed:J,lcgSeed:Ae}=t.calculateSeed(G);J!==L&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:L,cpuSeed:J,messageIndex:I});const Be={seed:L,datetime:V,timer0:F,vcount:z,keyCode:u.keyCode,conditions:m.conditions,message:G,sha1Hash:ne,lcgSeed:Ae,isMatch:!0};S.onResult(Be),y.matchesFound+=1}if(u.messageCount>0){const D=u.messageCount-1,R=u.baseSecondOffset+D;y.currentDateTime=ce(m.timePlan,R).toISOString()}y.currentStep+=u.messageCount,_(!0)},i=()=>{!n.isRunning||n.isPaused||(n.isPaused=!0,P(),n.callbacks?.onPaused())},c=()=>{!n.isRunning||!n.isPaused||(n.isPaused=!1,C(),n.callbacks?.onResumed())},l=()=>{n.isRunning&&(n.shouldStop=!0,n.isPaused=!1,C())},g=async()=>{for(;n.isPaused&&!n.shouldStop;)await p(25)},_=u=>{const f=n.callbacks,h=n.progress;if(!f||!h)return;const m=Date.now();u&&h.currentStep<h.totalSteps&&m-n.lastProgressUpdate<zt||(w(h),f.onProgress(h),n.lastProgressUpdate=m)},w=u=>{const f=v();u.elapsedTime=f,u.estimatedTimeRemaining=s(u.currentStep,u.totalSteps,f)},b=()=>{n.timer.cumulativeRunTime=0,n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1},P=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},C=()=>{n.timer.isPaused&&(n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1)},E=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},v=()=>n.timer.isPaused?n.timer.cumulativeRunTime:n.timer.cumulativeRunTime+(Date.now()-n.timer.segmentStartTime),s=(u,f,h)=>{if(u===0||u>=f)return 0;const m=h/u;return Math.round(m*(f-u))},p=u=>new Promise(f=>setTimeout(f,u));return{run:a,pause:i,resume:c,stop:l}}const ve=self,Nt=3n,he=0x100000000n,T={isRunning:!1,isPaused:!1};let W=null,re=null,ue=null,Z=null;function Yt(e,t){if(t<=0||e.maxMessagesPerDispatch<=0)return e.candidateCapacityPerDispatch;const r=Nt*BigInt(e.maxMessagesPerDispatch)*BigInt(t),n=Number((r+he-1n)/he);return Math.max(1,n)}async function Oe(){return ue||(ue=De()),ue}async function Ht(){if(W)return W;const e=await Oe(),t=Te(void 0,e);return W=Gt(t),W}async function Kt(){return Z||(Z=(await Oe()).deriveSearchJobLimits(),Z)}function M(e){ve.postMessage(e)}function $t(){M({type:"READY",message:"WebGPU worker initialized"})}function K(){T.isRunning=!1,T.isPaused=!1,re=null}function qt(){return gt()?!0:(M({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Vt(e){if(T.isRunning){M({type:"ERROR",error:"Search is already running"});return}if(!e.conditions||!e.targetSeeds){M({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!qt())return;T.isRunning=!0,T.isPaused=!1;let t,r;try{const[a,o]=await Promise.all([Kt(),Ht()]),d=Yt(a,e.targetSeeds.length),i={...a,candidateCapacityPerDispatch:Math.min(a.candidateCapacityPerDispatch,d)};t=He(e.conditions,e.targetSeeds,{limits:i}),r=o}catch(a){K();const o=a instanceof Error?a.message:"検索条件の解析中にエラーが発生しました";M({type:"ERROR",error:o,errorCode:"WEBGPU_CONTEXT_ERROR"});return}re=new AbortController;const n={onProgress:a=>{M({type:"PROGRESS",progress:a})},onResult:a=>{M({type:"RESULT",result:a})},onComplete:a=>{K(),M({type:"COMPLETE",message:a})},onError:(a,o)=>{K(),M({type:"ERROR",error:a,errorCode:o})},onPaused:()=>{T.isPaused=!0,M({type:"PAUSED"})},onResumed:()=>{T.isPaused=!1,M({type:"RESUMED"})},onStopped:(a,o)=>{K(),M({type:"STOPPED",message:a,progress:o})}};try{await r.run(t,n,re.signal)}catch(a){if(!T.isRunning)return;K();const o=a instanceof Error?a.message:"WebGPU search failed with unknown error";M({type:"ERROR",error:o,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function Jt(){!T.isRunning||T.isPaused||W?.pause()}function Xt(){!T.isRunning||!T.isPaused||W?.resume()}function Zt(){T.isRunning&&(W?.stop(),re?.abort())}$t();ve.onmessage=e=>{const t=e.data;switch(t.type){case"START_SEARCH":Vt(t);break;case"PAUSE_SEARCH":Jt();break;case"RESUME_SEARCH":Xt();break;case"STOP_SEARCH":Zt();break;default:M({type:"ERROR",error:`Unknown request type: ${t.type}`})}};
