const _e={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},Se=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],k=Se.reduce((e,[t,n])=>(e[t]=n,e),{}),ke=Se.length,Le=(1<<ke)-1,ye=12287,Ue=[1<<k["[↑]"]|1<<k["[↓]"],1<<k["[←]"]|1<<k["[→]"],1<<k.Select|1<<k.Start|1<<k.L|1<<k.R];function we(e,t){return Number.isFinite(e)?e&Le:0}function Fe(e){const t=we(e);return ye^t}function We(e){const t=we(e);for(const n of Ue)if((t&n)===n)return!0;return!1}function Ie(e){return Fe(e)}const Z=1e3,ie=60,xe=60,Re=24,be=ie*xe,ze=be*Re,Ee=ze*Z;function Ge(e){const t=e.timeRange;if(!t)throw new Error("timeRange is required for seed search");const n=ae("hour",t.hour,0,Re-1),r=ae("minute",t.minute,0,xe-1),a=ae("second",t.second,0,ie-1),o=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,0,0,0),d=new Date(e.dateRange.endYear,e.dateRange.endMonth-1,e.dateRange.endDay,0,0,0),i=o.getTime(),c=d.getTime();if(i>c)throw new Error("開始日が終了日より後に設定されています");const l=Math.floor((c-i)/Ee)+1;if(l<=0)throw new Error("探索日数が検出できませんでした");const p=n.count*r.count*a.count;if(p<=0)throw new Error("時刻レンジの組み合わせ数が0です");const _=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,n.start,r.start,a.start,0);return{plan:{dayCount:l,combosPerDay:p,hourRangeStart:n.start,hourRangeCount:n.count,minuteRangeStart:r.start,minuteRangeCount:r.count,secondRangeStart:a.start,secondRangeCount:a.count,startDayTimestampMs:i},firstCombinationDate:_}}function ce(e,t){const n=Math.max(e.minuteRangeCount,1),r=Math.max(e.secondRangeCount,1),a=Math.max(e.combosPerDay,1),o=Math.max(0,Math.trunc(t)),d=Math.floor(o/a),i=o-d*a,c=n*r,l=Math.floor(i/c),p=i-l*c,_=Math.floor(p/r),w=p-_*r,b=e.hourRangeStart+l,P=e.minuteRangeStart+_,C=e.secondRangeStart+w,v=e.startDayTimestampMs+d*Ee+b*be*Z+P*ie*Z+C*Z;return new Date(v)}function ae(e,t,n,r){if(!t)throw new Error(`${e} range is required for WebGPU search`);const a=Math.trunc(t.start),o=Math.trunc(t.end);if(Number.isNaN(a)||Number.isNaN(o))throw new Error(`${e} range must be numeric`);if(a<n||o>r)throw new Error(`${e} range must be within ${n} to ${r}`);if(a>o)throw new Error(`${e} range start must be <= end`);return{start:a,end:o,count:o-a+1}}const Ne=100663296,Ye=4294967295;function He(e,t=[],n){const{plan:r,firstCombinationDate:a}=Ge(e),o=qe(n),d=Xe(t),i=Je(e,r,a),c=Ke(i,o),l=c.reduce((p,_)=>p+_.messageCount,0);return{segments:c,targetSeeds:d,timePlan:r,summary:{totalMessages:l,totalSegments:c.length,targetSeedCount:d.length,rangeSeconds:i.rangeSeconds},limits:o,conditions:e}}function Ke(e,t){const n=[];if(e.rangeSeconds<=0)return n;const r=Math.max(1,t.workgroupSize*t.maxWorkgroupsPerDispatch),a=Math.min(t.maxMessagesPerDispatch,r),o={startDayOfWeek:e.startDayOfWeek,macLower:e.macLower,data7Swapped:e.data7Swapped,hardwareType:e.hardwareType,nazoSwapped:e.nazoSwapped,startYear:e.startYear,startDayOfYear:e.startDayOfYear,hourRangeStart:e.hourRangeStart,hourRangeCount:e.hourRangeCount,minuteRangeStart:e.minuteRangeStart,minuteRangeCount:e.minuteRangeCount,secondRangeStart:e.secondRangeStart,secondRangeCount:e.secondRangeCount};let d=0,i=0;for(const c of e.keyCodes){const l=Q(c>>>0);for(const p of e.timer0Segments)for(let _=p.timer0Min;_<=p.timer0Max;_+=1){const w=p.vcount;let b=e.rangeSeconds,P=0;const C=Q((w&65535)<<16|_&65535);for(;b>0;){const E=Math.min(b,a),v=$e(E,t),s=()=>Ze({...o,timer0VcountSwapped:C,keyInputSwapped:l});n.push({id:`seg-${i}`,keyCode:c,timer0:_,vcount:w,messageCount:E,baseSecondOffset:P,globalMessageOffset:d,workgroupCount:v,getUniformWords:s}),b-=E,P+=E,d+=E,i+=1}}}return n}function $e(e,t){const n=Math.max(1,Math.ceil(e/t.workgroupSize)),r=Math.max(1,t.maxWorkgroupsPerDispatch);return Math.min(n,r)}function qe(e){if(!e?.limits)throw new Error("Seed search job limits are required for WebGPU execution");return Ve(e.limits)}function Ve(e){const t=G(e.workgroupSize,"workgroupSize"),n=G(e.maxWorkgroupsPerDispatch,"maxWorkgroupsPerDispatch"),r=G(e.candidateCapacityPerDispatch,"candidateCapacityPerDispatch"),a=G(e.maxMessagesPerDispatch,"maxMessagesPerDispatch"),o=G(e.maxDispatchesInFlight,"maxDispatchesInFlight"),d=Math.max(1,Math.floor(Ye/Math.max(1,t))),i=Math.min(n,d),c=Math.max(1,t*i),l=Math.min(a,c);return{workgroupSize:t,maxWorkgroupsPerDispatch:i,candidateCapacityPerDispatch:r,maxMessagesPerDispatch:l,maxDispatchesInFlight:o}}function Je(e,t,n){const r=Qe(e),a=et(e.keyInput);if(a.length===0)throw new Error("入力されたキー条件から生成できる組み合わせがありません");const o=tt(e,r);if(o.length===0)throw new Error("timer0の範囲が正しく設定されていません");const d=st(r.nazo),{macLower:i,data7Swapped:c}=rt(e.macAddress,je[e.hardware]),l=t.dayCount*t.combosPerDay;if(l<=0)throw new Error("探索対象の秒数が0以下です");return{rangeSeconds:l,timer0Segments:o,keyCodes:a,nazoSwapped:d,macLower:i,data7Swapped:c,hardwareType:ot(e.hardware),startYear:n.getFullYear(),startDayOfYear:ut(n),startDayOfWeek:n.getDay(),hourRangeStart:t.hourRangeStart,hourRangeCount:t.hourRangeCount,minuteRangeStart:t.minuteRangeStart,minuteRangeCount:t.minuteRangeCount,secondRangeStart:t.secondRangeStart,secondRangeCount:t.secondRangeCount}}function Xe(e){if(!e||e.length===0)return new Uint32Array(0);const t=[];for(const n of e)typeof n!="number"||!Number.isFinite(n)||t.push(n>>>0);return Uint32Array.from(t)}function G(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function Ze(e){const t=new Uint32Array(20);return t[0]=e.timer0VcountSwapped>>>0,t[1]=e.macLower>>>0,t[2]=e.data7Swapped>>>0,t[3]=e.keyInputSwapped>>>0,t[4]=e.hardwareType>>>0,t[5]=e.startYear>>>0,t[6]=e.startDayOfYear>>>0,t[7]=e.startDayOfWeek>>>0,t[8]=e.hourRangeStart>>>0,t[9]=e.hourRangeCount>>>0,t[10]=e.minuteRangeStart>>>0,t[11]=e.minuteRangeCount>>>0,t[12]=e.secondRangeStart>>>0,t[13]=e.secondRangeCount>>>0,t[14]=(e.nazoSwapped[0]??0)>>>0,t[15]=(e.nazoSwapped[1]??0)>>>0,t[16]=(e.nazoSwapped[2]??0)>>>0,t[17]=(e.nazoSwapped[3]??0)>>>0,t[18]=(e.nazoSwapped[4]??0)>>>0,t[19]=0,t}const je={DS:8,DS_LITE:6,"3DS":9};function Qe(e){const t=_e[e.romVersion];if(!t)throw new Error(`ROMバージョン ${e.romVersion} は未対応です`);const n=t[e.romRegion];if(!n)throw new Error(`ROMリージョン ${e.romRegion} は未対応です`);return{nazo:[...n.nazo],vcountTimerRanges:n.vcountTimerRanges.map(r=>[...r])}}function et(e){const t=[];for(let a=0;a<12;a+=1)(e&1<<a)!==0&&t.push(a);const n=[],r=1<<t.length;for(let a=0;a<r;a+=1){let o=0;for(let d=0;d<t.length;d+=1)(a&1<<d)!==0&&(o|=1<<t[d]);We(o)||n.push((o^ye)>>>0)}return n}function tt(e,t){const n=[],{timer0VCountConfig:{useAutoConfiguration:r,timer0Range:{min:a,max:o},vcountRange:{min:d,max:i}}}=e;if(!r){for(let l=d;l<=i;l+=1)n.push({timer0Min:a,timer0Max:o,vcount:l});return n}let c=null;for(let l=a;l<=o;l+=1){const p=nt(t,l);c&&c.vcount===p&&l===c.timer0Max+1?c.timer0Max=l:(c&&n.push(c),c={timer0Min:l,timer0Max:l,vcount:p})}return c&&n.push(c),n}function nt(e,t){for(const[n,r,a]of e.vcountTimerRanges)if(t>=r&&t<=a)return n;return e.vcountTimerRanges.length>0?e.vcountTimerRanges[0][0]:96}function rt(e,t){const n=at(e),r=(n[4]&255)<<8|n[5]&255,o=((n[0]&255|(n[1]&255)<<8|(n[2]&255)<<16|(n[3]&255)<<24)^Ne^t)>>>0;return{macLower:r,data7Swapped:Q(o)}}function at(e){const t=new Array(6).fill(0);for(let n=0;n<6;n+=1){const r=e[n]??0;t[n]=(Number(r)&255)>>>0}return t}function ot(e){switch(e){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function Q(e){return((e&255)<<24|(e>>>8&255)<<16|(e>>>16&255)<<8|e>>>24&255)>>>0}function st(e){const t=new Uint32Array(e.length);for(let n=0;n<e.length;n+=1)t[n]=Q(e[n]>>>0);return t}function ut(e){const t=new Date(e.getFullYear(),0,1),n=e.getTime()-t.getTime();return Math.floor(n/(1440*60*1e3))+1}class de{calculateHash(t){if(t.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const n=1732584193,r=4023233417,a=2562383102,o=271733878,d=3285377520,i=new Array(80);for(let s=0;s<16;s++)i[s]=t[s];for(let s=16;s<80;s++)i[s]=this.leftRotate(i[s-3]^i[s-8]^i[s-14]^i[s-16],1);let c=n,l=r,p=a,_=o,w=d;for(let s=0;s<80;s++){let g;s<20?g=this.leftRotate(c,5)+(l&p|~l&_)+w+i[s]+1518500249&4294967295:s<40?g=this.leftRotate(c,5)+(l^p^_)+w+i[s]+1859775393&4294967295:s<60?g=this.leftRotate(c,5)+(l&p|l&_|p&_)+w+i[s]+2400959708&4294967295:g=this.leftRotate(c,5)+(l^p^_)+w+i[s]+3395469782&4294967295,w=_,_=p,p=this.leftRotate(l,30),l=c,c=g}const b=this.add32(n,c),P=this.add32(r,l),C=this.add32(a,p),E=this.add32(o,_),v=this.add32(d,w);return{h0:b,h1:P,h2:C,h3:E,h4:v}}leftRotate(t,n){return(t<<n|t>>>32-n)>>>0}add32(t,n){return(t+n&4294967295)>>>0}static hashToHex(t,n,r,a,o){return t.toString(16).padStart(8,"0")+n.toString(16).padStart(8,"0")+r.toString(16).padStart(8,"0")+a.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")}}let L=null,N=null;async function it(){return L||N||(N=(async()=>{try{const e=await import("./wasm_pkg-Df3C8dbS.js");let t;if(typeof process<"u"&&!!process.versions?.node){const r=await import("./__vite-browser-external-9wXp6ZBx.js"),o=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");t={module_or_path:r.readFileSync(o)}}else t={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-s2Bl4JAe.wasm",import.meta.url)};return await e.default(t),L={IntegratedSeedSearcher:e.IntegratedSeedSearcher,BWGenerationConfig:e.BWGenerationConfig,PokemonGenerator:e.PokemonGenerator,SeedEnumerator:e.SeedEnumerator,EncounterType:e.EncounterType,GameVersion:e.GameVersion,GameMode:e.GameMode,EggSeedEnumeratorJs:e.EggSeedEnumeratorJs,ParentsIVsJs:e.ParentsIVsJs,GenerationConditionsJs:e.GenerationConditionsJs,EverstonePlanJs:e.EverstonePlanJs,IndividualFilterJs:e.IndividualFilterJs,TrainerIds:e.TrainerIds,GenderRatio:e.GenderRatio,StatRange:e.StatRange,EggBootTimingSearchResult:e.EggBootTimingSearchResult,EggBootTimingSearchIterator:e.EggBootTimingSearchIterator,generate_egg_key_codes:e.generate_egg_key_codes,calculate_game_offset:e.calculate_game_offset,sha1_hash_batch:e.sha1_hash_batch},L}catch(e){throw console.error("Failed to load WebAssembly module:",e),L=null,N=null,e}})(),N)}function ct(){if(!L)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return L}function le(){return L!==null}const dt={DS:8,DS_LITE:6,"3DS":9};class lt{sha1;useWasm=!1;constructor(){this.sha1=new de}async initializeWasm(){try{return await it(),this.useWasm=!0,!0}catch(t){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",t),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&le()}getWasmModule(){return ct()}setUseWasm(t){if(t&&!le()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=t}getROMParameters(t,n){const r=_e[t];if(!r)return console.error(`ROM version not found: ${t}`),null;const a=r[n];return a?{nazo:[...a.nazo],vcountTimerRanges:a.vcountTimerRanges.map(o=>[...o])}:(console.error(`ROM region not found: ${n} for version ${t}`),null)}toLittleEndian32(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}toLittleEndian16(t){return(t&255)<<8|t>>>8&255}getDayOfWeek(t,n,r){return new Date(t,n-1,r).getDay()}generateMessage(t,n,r,a,o){const d=this.getROMParameters(t.romVersion,t.romRegion);if(!d)throw new Error(`No parameters found for ${t.romVersion} ${t.romRegion}`);const i=new Array(16).fill(0);for(let R=0;R<5;R++)i[R]=this.toLittleEndian32(d.nazo[R]);i[5]=this.toLittleEndian32(r<<16|n);const c=t.macAddress[4]<<8|t.macAddress[5];i[6]=c;const l=t.macAddress[0]<<0|t.macAddress[1]<<8|t.macAddress[2]<<16|t.macAddress[3]<<24,p=100663296,_=dt[t.hardware],w=l^p^_;i[7]=this.toLittleEndian32(w);const b=a.getFullYear()%100,P=a.getMonth()+1,C=a.getDate(),E=this.getDayOfWeek(a.getFullYear(),P,C),v=Math.floor(b/10)*16+b%10,s=Math.floor(P/10)*16+P%10,g=Math.floor(C/10)*16+C%10,u=Math.floor(E/10)*16+E%10;i[8]=v<<24|s<<16|g<<8|u;const f=a.getHours(),h=a.getMinutes(),m=a.getSeconds(),S=(t.hardware==="DS"||t.hardware==="DS_LITE")&&f>=12?1:0,y=Math.floor(f/10)*16+f%10,x=Math.floor(h/10)*16+h%10,B=Math.floor(m/10)*16+m%10;i[9]=S<<30|y<<24|x<<16|B<<8|0,i[10]=0,i[11]=0;const D=o??Ie(t.keyInput);return i[12]=this.toLittleEndian32(D),i[13]=2147483648,i[14]=0,i[15]=416,i}calculateSeed(t){const n=this.sha1.calculateHash(t),r=BigInt(this.toLittleEndian32(n.h0)),o=BigInt(this.toLittleEndian32(n.h1))<<32n|r,c=o*0x5D588B656C078965n+0x269EC3n;return{seed:Number(c>>32n&0xFFFFFFFFn),hash:de.hashToHex(n.h0,n.h1,n.h2,n.h3,n.h4),lcgSeed:o}}parseTargetSeeds(t){const n=t.split(`
`).map(d=>d.trim()).filter(d=>d.length>0),r=[],a=[],o=new Set;return n.forEach((d,i)=>{try{let c=d.toLowerCase();if(c.startsWith("0x")&&(c=c.substring(2)),!/^[0-9a-f]{1,8}$/.test(c)){a.push({line:i+1,value:d,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const l=parseInt(c,16);if(o.has(l))return;o.add(l),r.push(l)}catch(c){const l=c instanceof Error?c.message:String(c);a.push({line:i+1,value:d,error:l||"Failed to parse as hexadecimal number."})}}),{validSeeds:r,errors:a}}getVCountForTimer0(t,n){for(const[r,a,o]of t.vcountTimerRanges)if(n>=a&&n<=o)return r;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0][0]:96}}const K=2,j=1,fe={requiredFeatures:[],powerPreference:"high-performance"},Pe={workgroupSize:256,candidateCapacityPerDispatch:4096},ft=K*Uint32Array.BYTES_PER_ELEMENT,mt=4294967295,me={mobile:1,integrated:2,discrete:4,unknown:1},gt=1,ge=8;function Ce(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}const pt=Ce;async function De(e){if(!Ce())throw new Error("WebGPU is not available in this environment");const n=await navigator.gpu.requestAdapter({powerPreference:fe.powerPreference});if(!n)throw new Error("Failed to acquire WebGPU adapter");const r={requiredFeatures:fe.requiredFeatures,requiredLimits:e?.requiredLimits,label:"seed-search-device"},[a,o]=await Promise.all([n.requestDevice(r),yt(n)]);let d=!1;const i=a.lost.then(p=>(d=!0,console.warn("[webgpu] device lost:",p.message),p)),c=ht(n,a),l=a.limits;return{getAdapter:()=>n,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>l,getCapabilities:()=>c,getGpuProfile:()=>o,deriveSearchJobLimits:p=>_t(c.limits,o,p),isLost:()=>d,waitForLoss:()=>i,getSupportedWorkgroupSize:p=>Te(c.limits,p)}}function ht(e,t){const n=new Set;return e.features.forEach(r=>n.add(r)),{limits:t.limits,features:n}}function _t(e,t,n){const r={...Pe,...n},a=Mt(r),o=Te(e,a.workgroupSize),d=ee(e.maxComputeWorkgroupsPerDimension),i=a.maxWorkgroupsPerDispatch??d,c=Math.max(1,Math.floor(mt/Math.max(1,o))),l=$(Math.min(i,d,c),"maxWorkgroupsPerDispatch"),p=o*l,_=a.maxMessagesPerDispatch??p,w=$(Math.min(_,p),"maxMessagesPerDispatch"),b=Math.max(1,Math.floor(ee(e.maxStorageBufferBindingSize)/ft)),P=a.candidateCapacityPerDispatch??b,C=$(Math.min(P,b),"candidateCapacityPerDispatch"),E=St(t,a);return{workgroupSize:o,maxWorkgroupsPerDispatch:l,maxMessagesPerDispatch:w,candidateCapacityPerDispatch:C,maxDispatchesInFlight:E}}function Te(e,t){const n=Pe.workgroupSize,r=typeof t=="number"&&Number.isFinite(t)&&t>0?Math.floor(t):n,a=ee(e.maxComputeWorkgroupSizeX),o=ee(e.maxComputeInvocationsPerWorkgroup),d=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(r,d))}function ee(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function $(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function St(e,t){if(typeof t.maxDispatchesInFlight=="number")return $(Math.min(t.maxDispatchesInFlight,ge),"maxDispatchesInFlight");const n=e.isFallbackAdapter?gt:me[e.kind]??me.unknown;return $(Math.min(n,ge),"maxDispatchesInFlight")}async function yt(e){const t=wt(),r=!!e.isFallbackAdapter,a=Pt();if(a){const o={description:a.renderer};return{kind:a.kind,source:"webgl",userAgent:t,adapterInfo:o,isFallbackAdapter:r}}return r?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:void 0,isFallbackAdapter:r}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:void 0,isFallbackAdapter:r}}function wt(){return typeof navigator>"u"?"":navigator.userAgent||""}const xt=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],Rt=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"],bt=["intel","iris","uhd","hd graphics","radeon graphics","apple"];function oe(e,t){return t.some(n=>e.includes(n))}function Et(e){if(!e)return;const t=e.toLowerCase();if(oe(t,xt))return"mobile";if(oe(t,Rt))return"discrete";if(oe(t,bt))return"integrated"}function Pt(){const e=Ct();if(!e)return;const t=Et(e);if(t)return{kind:t,renderer:e}}function Ct(){const e=Dt();if(e)try{const t=Tt(e);if(!t)return;const n=t.getExtension("WEBGL_debug_renderer_info");if(!n)return;const r=t.getParameter(n.UNMASKED_RENDERER_WEBGL),a=t.getExtension("WEBGL_lose_context");return a&&a.loseContext(),typeof r=="string"?r:void 0}catch(t){console.warn("[webgpu] webgl renderer detection failed:",t);return}}function Dt(){if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);if(typeof document<"u"&&typeof document.createElement=="function"){const e=document.createElement("canvas");return e.width=1,e.height=1,e}}function Tt(e){const t=e,n=t.getContext;if(typeof n!="function")return null;const r=a=>n.call(t,a)??null;return r("webgl2")??r("webgl")}function Mt(e,t){return e}var vt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;

struct DispatchState {
  message_count : u32,
  base_second_offset : u32,
  candidate_capacity : u32,
  padding : u32,
};

struct SearchConstants {
  timer0_vcount_swapped : u32,
  mac_lower : u32,
  data7_swapped : u32,
  key_input_swapped : u32,
  hardware_type : u32,
  start_year : u32,
  start_day_of_year : u32,
  start_day_of_week : u32,
  hour_range_start : u32,
  hour_range_count : u32,
  minute_range_start : u32,
  minute_range_count : u32,
  second_range_start : u32,
  second_range_count : u32,
  nazo0 : u32,
  nazo1 : u32,
  nazo2 : u32,
  nazo3 : u32,
  nazo4 : u32,
  reserved0 : u32,
};

struct TargetSeedBuffer {
  count : u32,
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

const BCD_LOOKUP : array<u32, 100> = array<u32, 100>(
  0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u, 0x09u,
  0x10u, 0x11u, 0x12u, 0x13u, 0x14u, 0x15u, 0x16u, 0x17u, 0x18u, 0x19u,
  0x20u, 0x21u, 0x22u, 0x23u, 0x24u, 0x25u, 0x26u, 0x27u, 0x28u, 0x29u,
  0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u, 0x38u, 0x39u,
  0x40u, 0x41u, 0x42u, 0x43u, 0x44u, 0x45u, 0x46u, 0x47u, 0x48u, 0x49u,
  0x50u, 0x51u, 0x52u, 0x53u, 0x54u, 0x55u, 0x56u, 0x57u, 0x58u, 0x59u,
  0x60u, 0x61u, 0x62u, 0x63u, 0x64u, 0x65u, 0x66u, 0x67u, 0x68u, 0x69u,
  0x70u, 0x71u, 0x72u, 0x73u, 0x74u, 0x75u, 0x76u, 0x77u, 0x78u, 0x79u,
  0x80u, 0x81u, 0x82u, 0x83u, 0x84u, 0x85u, 0x86u, 0x87u, 0x88u, 0x89u,
  0x90u, 0x91u, 0x92u, 0x93u, 0x94u, 0x95u, 0x96u, 0x97u, 0x98u, 0x99u
);

@group(0) @binding(0) var<storage, read> state : DispatchState;
@group(0) @binding(1) var<uniform> constants : SearchConstants;
@group(0) @binding(2) var<storage, read> target_seeds : TargetSeedBuffer;
@group(0) @binding(3) var<storage, read_write> output_buffer : MatchOutputBuffer;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
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
  let le0 = ((h0 & 0x000000FFu) << 24u) |
    ((h0 & 0x0000FF00u) << 8u) |
    ((h0 & 0x00FF0000u) >> 8u) |
    ((h0 & 0xFF000000u) >> 24u);
  let le1 = ((h1 & 0x000000FFu) << 24u) |
    ((h1 & 0x0000FF00u) << 8u) |
    ((h1 & 0x00FF0000u) >> 8u) |
    ((h1 & 0xFF000000u) >> 24u);

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
  @builtin(global_invocation_id) global_id : vec3<u32>
) {

  let global_linear_index = global_id.x;
  let is_active = global_linear_index < state.message_count;
  var seed : u32 = 0u;
  var matched = false;

  if (is_active) {
    let safe_hour_count = max(constants.hour_range_count, 1u);
    let safe_minute_count = max(constants.minute_range_count, 1u);
    let safe_second_count = max(constants.second_range_count, 1u);
    let combos_per_day = safe_hour_count * safe_minute_count * safe_second_count;
    let total_second_offset = state.base_second_offset + global_linear_index;

    let day_offset = total_second_offset / combos_per_day;
    let remainder_after_day = total_second_offset - day_offset * combos_per_day;

    let entries_per_hour = safe_minute_count * safe_second_count;
    let hour_index = remainder_after_day / entries_per_hour;
    let remainder_after_hour = remainder_after_day - hour_index * entries_per_hour;
    let minute_index = remainder_after_hour / safe_second_count;
    let second_index = remainder_after_hour - minute_index * safe_second_count;

    let hour = constants.hour_range_start + hour_index;
    let minute = constants.minute_range_start + minute_index;
    let second = constants.second_range_start + second_index;

    var year = constants.start_year;
    var day_of_year = constants.start_day_of_year + day_offset;
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

    let day_of_week = (constants.start_day_of_week + day_offset) % 7u;
    let year_mod = year % 100u;
    let date_word = (BCD_LOOKUP[year_mod] << 24u) |
      (BCD_LOOKUP[month] << 16u) |
      (BCD_LOOKUP[day] << 8u) |
      BCD_LOOKUP[day_of_week];
    let is_pm = (constants.hardware_type <= 1u) && (hour >= 12u);
    let pm_flag = select(0u, 1u, is_pm);
    let time_word = (pm_flag << 30u) |
      (BCD_LOOKUP[hour] << 24u) |
      (BCD_LOOKUP[minute] << 16u) |
      (BCD_LOOKUP[second] << 8u);

    var w : array<u32, 16>;
    w[0] = constants.nazo0;
    w[1] = constants.nazo1;
    w[2] = constants.nazo2;
    w[3] = constants.nazo3;
    w[4] = constants.nazo4;
    w[5] = constants.timer0_vcount_swapped;
    w[6] = constants.mac_lower;
    w[7] = constants.data7_swapped;
    w[8] = date_word;
    w[9] = time_word;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = constants.key_input_swapped;
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

  if (!matched) {
    return;
  }

  let record_index = atomicAdd(&output_buffer.match_count, 1u);
  if (record_index >= state.candidate_capacity) {
    atomicSub(&output_buffer.match_count, 1u);
    return;
  }

  output_buffer.records[record_index].message_index = global_linear_index;
  output_buffer.records[record_index].seed = seed;
}`;const Bt=/WORKGROUP_SIZE_PLACEHOLDER/g,Ot="seed-search-kernel-module",At="seed-search-kernel",kt="seed-search-kernel-layout",Lt="seed-search-kernel-bind-layout";function Ut(e){return vt.replace(Bt,String(e))}function Ft(e,t){const n=e.createShaderModule({label:Ot,code:Ut(t)});n.getCompilationInfo?.().then(d=>{d.messages.length>0&&console.warn("[seed-search-kernel] compilation diagnostics",d.messages.map(i=>({message:i.message,line:i.lineNum,column:i.linePos,type:i.type})))}).catch(d=>{console.warn("[seed-search-kernel] compilation info failed",d)});const r=e.createBindGroupLayout({label:Lt,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),a=e.createPipelineLayout({label:kt,bindGroupLayouts:[r]});return{pipeline:e.createComputePipeline({label:At,layout:a,compute:{module:n,entryPoint:"sha1_generate"}}),layout:r}}const Wt=4,pe=256,se=new Uint32Array([0]);function Me(e,t){const n={context:t??null,pipeline:null,bindGroupLayout:null,targetBuffer:null,targetCapacity:0,workgroupSize:0,candidateCapacity:0,currentLimits:null,dispatchSlots:[],availableSlots:[],slotWaiters:[],desiredDispatchSlots:1},r=async(s,g)=>{n.context||(n.context=await De());const u=n.context.getDevice(),f=n.context.getSupportedWorkgroupSize(s.workgroupSize),h=Math.max(1,g?.dispatchSlots??n.desiredDispatchSlots??1),m=!n.currentLimits||n.workgroupSize!==f||n.candidateCapacity!==s.candidateCapacityPerDispatch;if(!n.pipeline||m){const{pipeline:y,layout:x}=Ft(u,f);n.pipeline=y,n.bindGroupLayout=x}n.workgroupSize=f,n.candidateCapacity=s.candidateCapacityPerDispatch,n.currentLimits=s,n.desiredDispatchSlots=h,a(u,h,s.candidateCapacityPerDispatch),n.currentLimits=s},a=(s,g,u)=>{for(const f of n.dispatchSlots)c(s,f,u);for(;n.dispatchSlots.length<g;){const f=n.dispatchSlots.length,h=o(s,f,u);n.dispatchSlots.push(h)}for(;n.dispatchSlots.length>g;){const f=n.dispatchSlots.pop();f&&l(f)}n.availableSlots=[...n.dispatchSlots],n.slotWaiters.length=0},o=(s,g,u)=>{const f=new Uint32Array(Wt),h=Y(f.byteLength),m=s.createBuffer({label:`seed-search-dispatch-state-${g}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),{matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}=i(s,u,g);return{id:g,dispatchStateBuffer:m,dispatchStateData:f,uniformBuffer:null,uniformCapacityWords:0,matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}},d=(s,g,u)=>{const f=Y(u*Uint32Array.BYTES_PER_ELEMENT);(!g.uniformBuffer||g.uniformCapacityWords<u)&&(g.uniformBuffer?.destroy(),g.uniformBuffer=s.createBuffer({label:`seed-search-uniform-${g.id}`,size:f,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),g.uniformCapacityWords=u)},i=(s,g,u)=>{const f=j+g*K,h=Y(f*Uint32Array.BYTES_PER_ELEMENT),m=s.createBuffer({label:`seed-search-output-${u}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),S=s.createBuffer({label:`seed-search-readback-${u}`,size:h,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return{matchOutputBuffer:m,readbackBuffer:S,matchBufferSize:h}},c=(s,g,u)=>{const f=j+u*K,h=Y(f*Uint32Array.BYTES_PER_ELEMENT);if(g.matchBufferSize===h)return;g.matchOutputBuffer.destroy(),g.readbackBuffer.destroy();const m=i(s,u,g.id);g.matchOutputBuffer=m.matchOutputBuffer,g.readbackBuffer=m.readbackBuffer,g.matchBufferSize=m.matchBufferSize},l=s=>{s.dispatchStateBuffer.destroy(),s.uniformBuffer?.destroy(),s.matchOutputBuffer.destroy(),s.readbackBuffer.destroy()},p=()=>n.availableSlots.length>0?Promise.resolve(n.availableSlots.pop()):new Promise(s=>{n.slotWaiters.push(s)}),_=s=>{const g=n.slotWaiters.shift();if(g){g(s);return}n.availableSlots.push(s)};return{ensureConfigured:r,setTargetSeeds:s=>{if(!n.context)throw new Error("SeedSearchEngine is not configured yet");const g=n.context.getDevice(),u=s.length,f=1+u,h=Y(f*Uint32Array.BYTES_PER_ELEMENT);(!n.targetBuffer||n.targetCapacity<u)&&(n.targetBuffer?.destroy(),n.targetBuffer=g.createBuffer({label:"seed-search-target-seeds",size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.targetCapacity=u);const m=new Uint32Array(f);m[0]=u>>>0;for(let S=0;S<u;S+=1)m[1+S]=s[S]>>>0;g.queue.writeBuffer(n.targetBuffer,0,m.buffer,m.byteOffset,m.byteLength)},executeSegment:async s=>{if(!n.context||!n.pipeline||!n.bindGroupLayout)throw new Error("SeedSearchEngine is not ready");if(!n.targetBuffer)throw new Error("Target seed buffer is not prepared");if(n.dispatchSlots.length===0)throw new Error("Dispatch slots are not configured");const g=n.context.getDevice(),u=g.queue,f=Math.max(1,s.workgroupCount),h=f,m=await p();try{const S=J();u.writeBuffer(m.matchOutputBuffer,0,se.buffer,se.byteOffset,se.byteLength);const y=m.dispatchStateData;y[0]=s.messageCount>>>0,y[1]=s.baseSecondOffset>>>0,y[2]=n.candidateCapacity>>>0,y[3]=0,u.writeBuffer(m.dispatchStateBuffer,0,y.buffer,y.byteOffset,y.byteLength);const x=s.getUniformWords();d(g,m,x.length),u.writeBuffer(m.uniformBuffer,0,x.buffer,x.byteOffset,x.byteLength);const B=J(),D=g.createBindGroup({label:`seed-search-bind-group-${s.id}-slot-${m.id}`,layout:n.bindGroupLayout,entries:[{binding:0,resource:{buffer:m.dispatchStateBuffer}},{binding:1,resource:{buffer:m.uniformBuffer}},{binding:2,resource:{buffer:n.targetBuffer}},{binding:3,resource:{buffer:m.matchOutputBuffer}}]}),R=g.createCommandEncoder({label:`seed-search-encoder-${s.id}`}),O=R.beginComputePass({label:`seed-search-pass-${s.id}`});O.setPipeline(n.pipeline),O.setBindGroup(0,D),O.dispatchWorkgroups(f),O.end(),R.copyBufferToBuffer(m.matchOutputBuffer,0,m.readbackBuffer,0,m.matchBufferSize);const U=R.finish();u.submit([U]),await m.readbackBuffer.mapAsync(GPUMapMode.READ,0,m.matchBufferSize);const A=J(),ne=m.readbackBuffer.getMappedRange(0,m.matchBufferSize),W=new Uint32Array(ne.slice(0));m.readbackBuffer.unmap();const I=J(),q=W[0]??0,z=Math.min(q,n.candidateCapacity),re=Math.min(W.length,j+z*K),V={words:W.slice(0,re),matchCount:z};return e?.onDispatchComplete?.({segmentId:s.id,messageCount:s.messageCount,workgroupCount:h,matchCount:z,candidateCapacity:n.candidateCapacity,timings:{totalMs:I-S,setupMs:B-S,gpuMs:A-B,readbackMs:I-A},timestampMs:I}),V}finally{_(m)}},dispose:()=>{for(const s of n.dispatchSlots)l(s);n.dispatchSlots=[],n.availableSlots=[],n.slotWaiters.length=0,n.targetBuffer?.destroy(),n.context=null,n.pipeline=null,n.bindGroupLayout=null,n.targetBuffer=null,n.targetCapacity=0,n.currentLimits=null},getWorkgroupSize:()=>n.workgroupSize,getCandidateCapacity:()=>n.candidateCapacity,getSupportedLimits:()=>n.context?.getLimits()??null}}function Y(e){return Math.ceil(e/pe)*pe}function J(){return typeof performance<"u"?performance.now():Date.now()}const It=1024,zt=500;function Gt(e){const t=new lt,n=e??Me(),r={isRunning:!1,isPaused:!1,shouldStop:!1,job:null,progress:null,callbacks:null,timer:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1},lastProgressUpdate:0},a=async(u,f,h)=>{if(r.isRunning)throw new Error("Seed search is already running");r.isRunning=!0,r.isPaused=!1,r.shouldStop=h?.aborted??!1,r.job=u,r.callbacks=f,r.lastProgressUpdate=0,r.progress={currentStep:0,totalSteps:u.summary.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0,currentDateTime:u.timePlan?new Date(u.timePlan.startDayTimestampMs).toISOString():void 0},b(),_(!1);let m;if(h){const S=()=>{r.shouldStop=!0};h.addEventListener("abort",S),m=()=>h.removeEventListener("abort",S),r.abortCleanup=m}try{if(u.summary.totalMessages===0){f.onComplete("探索対象の組み合わせが存在しません");return}const S=Math.max(1,Math.min(u.limits.maxDispatchesInFlight??1,u.segments.length||1));await n.ensureConfigured(u.limits,{dispatchSlots:S}),n.setTargetSeeds(u.targetSeeds);const y=new Set,x=new Set,B=D=>{const R=(async()=>{if(r.shouldStop)return;const{words:O,matchCount:U}=await n.executeSegment(D);if(r.shouldStop)return;const A=(async()=>{await d(D,O,U)})();x.add(A),A.finally(()=>x.delete(A))})();y.add(R),R.finally(()=>y.delete(R))};for(const D of u.segments){if(r.shouldStop||(await p(),r.shouldStop))break;B(D),y.size>=S&&await Promise.race(y)}y.size>0&&await Promise.all(y),x.size>0&&await Promise.all(x),o()}catch(S){const y=S instanceof Error?S.message:"WebGPU検索中に不明なエラーが発生しました",x=globalThis.GPUValidationError,B=x&&S instanceof x?"WEBGPU_VALIDATION_ERROR":void 0;throw f.onError(y,B),S}finally{m?.(),r.abortCleanup=void 0,r.isRunning=!1,r.isPaused=!1,E(),r.job=null,r.callbacks=null,r.progress=null,r.shouldStop=!1,r.lastProgressUpdate=0}},o=()=>{const u=r.callbacks,f=r.progress;if(!(!u||!f)){if(w(f),r.shouldStop){u.onStopped("検索を停止しました",f);return}u.onProgress(f),u.onComplete(`検索が完了しました。${f.matchesFound}件ヒットしました。`)}},d=async(u,f,h)=>{const m=r.job,S=r.callbacks,y=r.progress;if(!m||!S||!y)return;const x=j,B=K;for(let D=0;D<h&&!(r.shouldStop||D%It===0&&(await p(),r.shouldStop));D+=1){const R=x+D*B,O=f[R],U=f[R+1]>>>0,A=u.globalMessageOffset+O,ne=u.baseSecondOffset+O,W=u.timer0,I=u.vcount,q=ce(m.timePlan,ne),z=t.generateMessage(m.conditions,W,I,q,u.keyCode),{hash:re,seed:V,lcgSeed:Oe}=t.calculateSeed(z);V!==U&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:U,cpuSeed:V,messageIndex:A});const Ae={seed:U,datetime:q,timer0:W,vcount:I,keyCode:u.keyCode,conditions:m.conditions,message:z,sha1Hash:re,lcgSeed:Oe,isMatch:!0};S.onResult(Ae),y.matchesFound+=1}if(u.messageCount>0){const D=u.messageCount-1,R=u.baseSecondOffset+D;y.currentDateTime=ce(m.timePlan,R).toISOString()}y.currentStep+=u.messageCount,_(!0)},i=()=>{!r.isRunning||r.isPaused||(r.isPaused=!0,P(),r.callbacks?.onPaused())},c=()=>{!r.isRunning||!r.isPaused||(r.isPaused=!1,C(),r.callbacks?.onResumed())},l=()=>{r.isRunning&&(r.shouldStop=!0,r.isPaused=!1,C())},p=async()=>{for(;r.isPaused&&!r.shouldStop;)await g(25)},_=u=>{const f=r.callbacks,h=r.progress;if(!f||!h)return;const m=Date.now();u&&h.currentStep<h.totalSteps&&m-r.lastProgressUpdate<zt||(w(h),f.onProgress(h),r.lastProgressUpdate=m)},w=u=>{const f=v();u.elapsedTime=f,u.estimatedTimeRemaining=s(u.currentStep,u.totalSteps,f)},b=()=>{r.timer.cumulativeRunTime=0,r.timer.segmentStartTime=Date.now(),r.timer.isPaused=!1},P=()=>{r.timer.isPaused||(r.timer.cumulativeRunTime+=Date.now()-r.timer.segmentStartTime,r.timer.isPaused=!0)},C=()=>{r.timer.isPaused&&(r.timer.segmentStartTime=Date.now(),r.timer.isPaused=!1)},E=()=>{r.timer.isPaused||(r.timer.cumulativeRunTime+=Date.now()-r.timer.segmentStartTime,r.timer.isPaused=!0)},v=()=>r.timer.isPaused?r.timer.cumulativeRunTime:r.timer.cumulativeRunTime+(Date.now()-r.timer.segmentStartTime),s=(u,f,h)=>{if(u===0||u>=f)return 0;const m=h/u;return Math.round(m*(f-u))},g=u=>new Promise(f=>setTimeout(f,u));return{run:a,pause:i,resume:c,stop:l}}const ve=self,Nt=3n,he=0x100000000n,M={isRunning:!1,isPaused:!1};let F=null,te=null,ue=null,X=null;function Yt(e,t){if(t<=0||e.maxMessagesPerDispatch<=0)return e.candidateCapacityPerDispatch;const n=Nt*BigInt(e.maxMessagesPerDispatch)*BigInt(t),r=Number((n+he-1n)/he);return Math.max(1,r)}async function Be(){return ue||(ue=De()),ue}async function Ht(){if(F)return F;const e=await Be(),t=Me(void 0,e);return F=Gt(t),F}async function Kt(){return X||(X=(await Be()).deriveSearchJobLimits(),X)}function T(e){ve.postMessage(e)}function $t(){T({type:"READY",message:"WebGPU worker initialized"})}function H(){M.isRunning=!1,M.isPaused=!1,te=null}function qt(){return pt()?!0:(T({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Vt(e){if(M.isRunning){T({type:"ERROR",error:"Search is already running"});return}if(!e.conditions||!e.targetSeeds){T({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!qt())return;M.isRunning=!0,M.isPaused=!1;let t,n;try{const[a,o]=await Promise.all([Kt(),Ht()]),d=Yt(a,e.targetSeeds.length),i={...a,candidateCapacityPerDispatch:Math.min(a.candidateCapacityPerDispatch,d)};t=He(e.conditions,e.targetSeeds,{limits:i}),n=o}catch(a){H();const o=a instanceof Error?a.message:"検索条件の解析中にエラーが発生しました";T({type:"ERROR",error:o,errorCode:"WEBGPU_CONTEXT_ERROR"});return}te=new AbortController;const r={onProgress:a=>{T({type:"PROGRESS",progress:a})},onResult:a=>{T({type:"RESULT",result:a})},onComplete:a=>{H(),T({type:"COMPLETE",message:a})},onError:(a,o)=>{H(),T({type:"ERROR",error:a,errorCode:o})},onPaused:()=>{M.isPaused=!0,T({type:"PAUSED"})},onResumed:()=>{M.isPaused=!1,T({type:"RESUMED"})},onStopped:(a,o)=>{H(),T({type:"STOPPED",message:a,progress:o})}};try{await n.run(t,r,te.signal)}catch(a){if(!M.isRunning)return;H();const o=a instanceof Error?a.message:"WebGPU search failed with unknown error";T({type:"ERROR",error:o,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function Jt(){!M.isRunning||M.isPaused||F?.pause()}function Xt(){!M.isRunning||!M.isPaused||F?.resume()}function Zt(){M.isRunning&&(F?.stop(),te?.abort())}$t();ve.onmessage=e=>{const t=e.data;switch(t.type){case"START_SEARCH":Vt(t);break;case"PAUSE_SEARCH":Jt();break;case"RESUME_SEARCH":Xt();break;case"STOP_SEARCH":Zt();break;default:T({type:"ERROR",error:`Unknown request type: ${t.type}`})}};
