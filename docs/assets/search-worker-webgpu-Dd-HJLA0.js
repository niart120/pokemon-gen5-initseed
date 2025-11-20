const ye={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[[96,3193,3194]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3204,3205]]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[[96,3195,3196]]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[[95,3191,3192]]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[[95,3187,3188]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[96,3206,3207]]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[[95,3178,3179]]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[[95,3175,3177]]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[[96,3195,3196]]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[[96,3198,3200]]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[[96,3194,3195]]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[[95,3182,3183]]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[[95,3184,3185]]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[[96,3195,3196]]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[[130,4354,4360]]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[[130,4335,4340]]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[[130,4354,4360]]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[[129,4325,4328],[130,4329,4332]]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[[130,4340,4344]]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[[130,4353,4358]]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[[130,4359,4361],[131,4361,4365]]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[[130,4341,4347]]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[[129,4324,4329]]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[[130,4338,4342]]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[[130,4325,4333]]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[[130,4332,4336]]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[[130,4335,4340]]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[[130,4351,4356]]}}},we=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],T=we.reduce((e,[r,t])=>(e[r]=t,e),{}),Ie=we.length,Ue=(1<<Ie)-1,Se=12287,ze=[1<<T["[↑]"]|1<<T["[↓]"],1<<T["[←]"]|1<<T["[→]"],1<<T.Select|1<<T.Start|1<<T.L|1<<T.R];function be(e,r){return Number.isFinite(e)?e&Ue:0}function Fe(e){const r=be(e);return Se^r}function Ge(e){const r=be(e);for(const t of ze)if((r&t)===t)return!0;return!1}function Ne(e){return Fe(e)}const V=1e3,ne=60,Re=60,xe=24,Pe=ne*Re,Ye=Pe*xe,Ee=Ye*V;function He(e){const r=e.timeRange;if(!r)throw new Error("timeRange is required for seed search");const t=J("hour",r.hour,0,xe-1),n=J("minute",r.minute,0,Re-1),a=J("second",r.second,0,ne-1),o=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,0,0,0),s=new Date(e.dateRange.endYear,e.dateRange.endMonth-1,e.dateRange.endDay,0,0,0),i=o.getTime(),u=s.getTime();if(i>u)throw new Error("開始日が終了日より後に設定されています");const c=Math.floor((u-i)/Ee)+1;if(c<=0)throw new Error("探索日数が検出できませんでした");const f=t.count*n.count*a.count;if(f<=0)throw new Error("時刻レンジの組み合わせ数が0です");const d=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,t.start,n.start,a.start,0);return{plan:{dayCount:c,combosPerDay:f,hourRangeStart:t.start,hourRangeCount:t.count,minuteRangeStart:n.start,minuteRangeCount:n.count,secondRangeStart:a.start,secondRangeCount:a.count,startDayTimestampMs:i},firstCombinationDate:d}}function fe(e,r){const t=Math.max(e.minuteRangeCount,1),n=Math.max(e.secondRangeCount,1),a=Math.max(e.combosPerDay,1),o=Math.max(0,Math.trunc(r)),s=Math.floor(o/a),i=o-s*a,u=t*n,c=Math.floor(i/u),f=i-c*u,d=Math.floor(f/n),m=f-d*n,g=e.hourRangeStart+c,y=e.minuteRangeStart+d,w=e.secondRangeStart+m,S=e.startDayTimestampMs+s*Ee+g*Pe*V+y*ne*V+w*V;return new Date(S)}function J(e,r,t,n){if(!r)throw new Error(`${e} range is required for WebGPU search`);const a=Math.trunc(r.start),o=Math.trunc(r.end);if(Number.isNaN(a)||Number.isNaN(o))throw new Error(`${e} range must be numeric`);if(a<t||o>n)throw new Error(`${e} range must be within ${t} to ${n}`);if(a>o)throw new Error(`${e} range start must be <= end`);return{start:a,end:o,count:o-a+1}}const Ve=100663296,Ke=4294967295;function qe(e,r=[],t){const{plan:n,firstCombinationDate:a}=He(e),o=je(t),s=Qe(r),i=Ze(e,n,a),u=$e(i,o),c=u.reduce((f,d)=>f+d.messageCount,0);return{segments:u,targetSeeds:s,timePlan:n,summary:{totalMessages:c,totalSegments:u.length,targetSeedCount:s.length,rangeSeconds:i.rangeSeconds},limits:o,conditions:e}}function $e(e,r){const t=[];if(e.rangeSeconds<=0)return t;const n=Math.max(1,r.workgroupSize*r.maxWorkgroupsPerDispatch*r.maxWorkgroupsPerDispatchY),a=Math.min(r.maxMessagesPerDispatch,n);let o=0,s=0;for(const i of e.keyCodes){const u=ae(i>>>0);for(const c of e.timer0Segments){const f=c.timer0Max-c.timer0Min+1;if(f<=0)continue;const d=1;let g=e.rangeSeconds*f,y=0;for(;g>0;){const w=Math.min(g,a),p=Xe(w,r),S=et(e.rangeSeconds,d,y),_=tt({messageCount:w,baseTimer0Index:S.baseTimer0Index,baseVcountIndex:S.baseVcountIndex,baseSecondOffset:S.baseSecondOffset,rangeSeconds:e.rangeSeconds,timer0Min:c.timer0Min,timer0Count:f,vcountMin:c.vcount,vcountCount:d,startSecondOfDay:e.startSecondOfDay,startDayOfWeek:e.startDayOfWeek,macLower:e.macLower,data7Swapped:e.data7Swapped,keyInputSwapped:u,hardwareType:e.hardwareType,nazoSwapped:e.nazoSwapped,startYear:e.startYear,startDayOfYear:e.startDayOfYear,dayCount:e.dayCount,hourRangeStart:e.hourRangeStart,hourRangeCount:e.hourRangeCount,minuteRangeStart:e.minuteRangeStart,minuteRangeCount:e.minuteRangeCount,secondRangeStart:e.secondRangeStart,secondRangeCount:e.secondRangeCount,groupsPerDispatch:p.x,workgroupsPerDispatchY:p.y,workgroupSize:r.workgroupSize,candidateCapacity:r.candidateCapacityPerDispatch});t.push({id:`seg-${s}`,keyCode:i,timer0Min:c.timer0Min,timer0Max:c.timer0Max,timer0Count:f,vcountMin:c.vcount,vcountCount:d,rangeSeconds:e.rangeSeconds,messageCount:w,localMessageOffset:y,globalMessageOffset:o,baseTimer0Index:S.baseTimer0Index,baseVcountIndex:S.baseVcountIndex,baseSecondOffset:S.baseSecondOffset,workgroupCount:p.total,workgroupCountX:p.x,workgroupCountY:p.y,configWords:_}),g-=w,y+=w,o+=w,s+=1}}}return t}function Xe(e,r){const t=Math.max(1,Math.ceil(e/r.workgroupSize)),n=Math.max(1,r.maxWorkgroupsPerDispatch),a=Math.max(1,r.maxWorkgroupsPerDispatchY),o=Math.min(t,n),s=Math.max(1,Math.ceil(t/o)),i=Math.min(s,a),u=Math.max(1,Math.ceil(t/i)),c=Math.min(u,n),f=c*i;return{x:c,y:i,total:f}}function je(e){if(!e?.limits)throw new Error("Seed search job limits are required for WebGPU execution");return Je(e.limits)}function Je(e){const r=U(e.workgroupSize,"workgroupSize"),t=U(e.maxWorkgroupsPerDispatch,"maxWorkgroupsPerDispatch"),n=U(e.maxWorkgroupsPerDispatchY,"maxWorkgroupsPerDispatchY"),a=U(e.candidateCapacityPerDispatch,"candidateCapacityPerDispatch"),o=U(e.maxMessagesPerDispatch,"maxMessagesPerDispatch"),s=Math.max(1,Math.floor(Ke/Math.max(1,r*t))),i=Math.min(n,s),u=Math.max(1,r*t*i),c=Math.min(o,u);return{workgroupSize:r,maxWorkgroupsPerDispatch:t,maxWorkgroupsPerDispatchY:i,candidateCapacityPerDispatch:a,maxMessagesPerDispatch:c}}function Ze(e,r,t){const n=nt(e),a=at(e.keyInput);if(a.length===0)throw new Error("入力されたキー条件から生成できる組み合わせがありません");const o=ot(e,n);if(o.length===0)throw new Error("timer0の範囲が正しく設定されていません");const s=dt(n.nazo),{macLower:i,data7Swapped:u}=it(e.macAddress,rt[e.hardware]),c=r.dayCount*r.combosPerDay;if(c<=0)throw new Error("探索対象の秒数が0以下です");return{rangeSeconds:c,timer0Segments:o,keyCodes:a,nazoSwapped:s,macLower:i,data7Swapped:u,hardwareType:ct(e.hardware),startYear:t.getFullYear(),startDayOfYear:lt(t),startSecondOfDay:ft(t),startDayOfWeek:t.getDay(),dayCount:r.dayCount,hourRangeStart:r.hourRangeStart,hourRangeCount:r.hourRangeCount,minuteRangeStart:r.minuteRangeStart,minuteRangeCount:r.minuteRangeCount,secondRangeStart:r.secondRangeStart,secondRangeCount:r.secondRangeCount}}function Qe(e){if(!e||e.length===0)return new Uint32Array(0);const r=[];for(const t of e)typeof t!="number"||!Number.isFinite(t)||r.push(t>>>0);return Uint32Array.from(r)}function U(e,r){if(!Number.isFinite(e)||e<=0)throw new Error(`${r} must be a positive finite number`);return Math.floor(e)}function et(e,r,t){const n=Math.max(e,1),a=Math.max(r,1),o=n,s=o*a,i=Math.floor(t/s),u=t-i*s,c=Math.floor(u/o),f=u-c*o;return{baseTimer0Index:i,baseVcountIndex:c,baseSecondOffset:f}}function tt(e){const r=new Uint32Array(33);r[0]=e.messageCount>>>0,r[1]=e.baseTimer0Index>>>0,r[2]=e.baseVcountIndex>>>0,r[3]=e.baseSecondOffset>>>0,r[4]=e.rangeSeconds>>>0,r[5]=e.timer0Min>>>0,r[6]=e.timer0Count>>>0,r[7]=e.vcountMin>>>0,r[8]=e.vcountCount>>>0,r[9]=e.startSecondOfDay>>>0,r[10]=e.startDayOfWeek>>>0,r[11]=e.macLower>>>0,r[12]=e.data7Swapped>>>0,r[13]=e.keyInputSwapped>>>0,r[14]=e.hardwareType>>>0;for(let t=0;t<e.nazoSwapped.length&&15+t<20;t+=1)r[15+t]=e.nazoSwapped[t]>>>0;return r[20]=e.startYear>>>0,r[21]=e.startDayOfYear>>>0,r[22]=e.groupsPerDispatch>>>0,r[23]=e.workgroupSize>>>0,r[24]=e.candidateCapacity>>>0,r[25]=e.dayCount>>>0,r[26]=e.hourRangeStart>>>0,r[27]=e.hourRangeCount>>>0,r[28]=e.minuteRangeStart>>>0,r[29]=e.minuteRangeCount>>>0,r[30]=e.secondRangeStart>>>0,r[31]=e.secondRangeCount>>>0,r[32]=e.workgroupsPerDispatchY>>>0,r}const rt={DS:8,DS_LITE:6,"3DS":9};function nt(e){const r=ye[e.romVersion];if(!r)throw new Error(`ROMバージョン ${e.romVersion} は未対応です`);const t=r[e.romRegion];if(!t)throw new Error(`ROMリージョン ${e.romRegion} は未対応です`);return{nazo:[...t.nazo],vcountTimerRanges:t.vcountTimerRanges.map(n=>[...n])}}function at(e){const r=[];for(let a=0;a<12;a+=1)(e&1<<a)!==0&&r.push(a);const t=[],n=1<<r.length;for(let a=0;a<n;a+=1){let o=0;for(let s=0;s<r.length;s+=1)(a&1<<s)!==0&&(o|=1<<r[s]);Ge(o)||t.push((o^Se)>>>0)}return t}function ot(e,r){const t=[],n=e.timer0VCountConfig.timer0Range.min,a=e.timer0VCountConfig.timer0Range.max;let o=null;for(let s=n;s<=a;s+=1){const i=st(r,s);o&&o.vcount===i&&s===o.timer0Max+1?o.timer0Max=s:(o&&t.push(o),o={timer0Min:s,timer0Max:s,vcount:i})}return o&&t.push(o),t}function st(e,r){for(const[t,n,a]of e.vcountTimerRanges)if(r>=n&&r<=a)return t;return e.vcountTimerRanges.length>0?e.vcountTimerRanges[0][0]:96}function it(e,r){const t=ut(e),n=(t[4]&255)<<8|t[5]&255,o=((t[0]&255|(t[1]&255)<<8|(t[2]&255)<<16|(t[3]&255)<<24)^Ve^r)>>>0;return{macLower:n,data7Swapped:ae(o)}}function ut(e){const r=new Array(6).fill(0);for(let t=0;t<6;t+=1){const n=e[t]??0;r[t]=(Number(n)&255)>>>0}return r}function ct(e){switch(e){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function ae(e){return((e&255)<<24|(e>>>8&255)<<16|(e>>>16&255)<<8|e>>>24&255)>>>0}function dt(e){const r=new Uint32Array(e.length);for(let t=0;t<e.length;t+=1)r[t]=ae(e[t]>>>0);return r}function lt(e){const r=new Date(e.getFullYear(),0,1),t=e.getTime()-r.getTime();return Math.floor(t/(1440*60*1e3))+1}function ft(e){return e.getHours()*3600+e.getMinutes()*60+e.getSeconds()}class me{calculateHash(r){if(r.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const t=1732584193,n=4023233417,a=2562383102,o=271733878,s=3285377520,i=new Array(80);for(let _=0;_<16;_++)i[_]=r[_];for(let _=16;_<80;_++)i[_]=this.leftRotate(i[_-3]^i[_-8]^i[_-14]^i[_-16],1);let u=t,c=n,f=a,d=o,m=s;for(let _=0;_<80;_++){let P;_<20?P=this.leftRotate(u,5)+(c&f|~c&d)+m+i[_]+1518500249&4294967295:_<40?P=this.leftRotate(u,5)+(c^f^d)+m+i[_]+1859775393&4294967295:_<60?P=this.leftRotate(u,5)+(c&f|c&d|f&d)+m+i[_]+2400959708&4294967295:P=this.leftRotate(u,5)+(c^f^d)+m+i[_]+3395469782&4294967295,m=d,d=f,f=this.leftRotate(c,30),c=u,u=P}const g=this.add32(t,u),y=this.add32(n,c),w=this.add32(a,f),p=this.add32(o,d),S=this.add32(s,m);return{h0:g,h1:y,h2:w,h3:p,h4:S}}leftRotate(r,t){return(r<<t|r>>>32-t)>>>0}add32(r,t){return(r+t&4294967295)>>>0}static hashToHex(r,t,n,a,o){return r.toString(16).padStart(8,"0")+t.toString(16).padStart(8,"0")+n.toString(16).padStart(8,"0")+a.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")}}let k=null,z=null;async function mt(){return k||z||(z=(async()=>{try{const e=await import("./wasm_pkg-DRWLiY4b.js");let r;if(typeof process<"u"&&!!process.versions?.node){const n=await import("./__vite-browser-external-9wXp6ZBx.js"),o=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");r={module_or_path:n.readFileSync(o)}}else r={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-D27IxIOn.wasm",import.meta.url)};return await e.default(r),k={IntegratedSeedSearcher:e.IntegratedSeedSearcher,BWGenerationConfig:e.BWGenerationConfig,PokemonGenerator:e.PokemonGenerator,SeedEnumerator:e.SeedEnumerator,EncounterType:e.EncounterType,GameVersion:e.GameVersion,GameMode:e.GameMode,calculate_game_offset:e.calculate_game_offset,sha1_hash_batch:e.sha1_hash_batch},k}catch(e){throw console.error("Failed to load WebAssembly module:",e),k=null,z=null,e}})(),z)}function gt(){if(!k)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return k}function ge(){return k!==null}const pt={DS:8,DS_LITE:6,"3DS":9};class _t{sha1;useWasm=!1;constructor(){this.sha1=new me}async initializeWasm(){try{return await mt(),this.useWasm=!0,!0}catch(r){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",r),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&ge()}getWasmModule(){return gt()}setUseWasm(r){if(r&&!ge()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=r}getROMParameters(r,t){const n=ye[r];if(!n)return console.error(`ROM version not found: ${r}`),null;const a=n[t];return a?{nazo:[...a.nazo],vcountTimerRanges:a.vcountTimerRanges.map(o=>[...o])}:(console.error(`ROM region not found: ${t} for version ${r}`),null)}toLittleEndian32(r){return((r&255)<<24|(r>>>8&255)<<16|(r>>>16&255)<<8|r>>>24&255)>>>0}toLittleEndian16(r){return(r&255)<<8|r>>>8&255}getDayOfWeek(r,t,n){return new Date(r,t-1,n).getDay()}generateMessage(r,t,n,a,o){const s=this.getROMParameters(r.romVersion,r.romRegion);if(!s)throw new Error(`No parameters found for ${r.romVersion} ${r.romRegion}`);const i=new Array(16).fill(0);for(let O=0;O<5;O++)i[O]=this.toLittleEndian32(s.nazo[O]);i[5]=this.toLittleEndian32(n<<16|t);const u=r.macAddress[4]<<8|r.macAddress[5];i[6]=u;const c=r.macAddress[0]<<0|r.macAddress[1]<<8|r.macAddress[2]<<16|r.macAddress[3]<<24,f=100663296,d=pt[r.hardware],m=c^f^d;i[7]=this.toLittleEndian32(m);const g=a.getFullYear()%100,y=a.getMonth()+1,w=a.getDate(),p=this.getDayOfWeek(a.getFullYear(),y,w),S=Math.floor(g/10)*16+g%10,_=Math.floor(y/10)*16+y%10,P=Math.floor(w/10)*16+w%10,l=Math.floor(p/10)*16+p%10;i[8]=S<<24|_<<16|P<<8|l;const h=a.getHours(),b=a.getMinutes(),R=a.getSeconds(),x=(r.hardware==="DS"||r.hardware==="DS_LITE")&&h>=12?1:0,v=Math.floor(h/10)*16+h%10,C=Math.floor(b/10)*16+b%10,D=Math.floor(R/10)*16+R%10;i[9]=x<<30|v<<24|C<<16|D<<8|0,i[10]=0,i[11]=0;const X=o??Ne(r.keyInput);return i[12]=this.toLittleEndian32(X),i[13]=2147483648,i[14]=0,i[15]=416,i}calculateSeed(r){const t=this.sha1.calculateHash(r),n=BigInt(this.toLittleEndian32(t.h0)),o=BigInt(this.toLittleEndian32(t.h1))<<32n|n,u=o*0x5D588B656C078965n+0x269EC3n;return{seed:Number(u>>32n&0xFFFFFFFFn),hash:me.hashToHex(t.h0,t.h1,t.h2,t.h3,t.h4),lcgSeed:o}}parseTargetSeeds(r){const t=r.split(`
`).map(s=>s.trim()).filter(s=>s.length>0),n=[],a=[],o=new Set;return t.forEach((s,i)=>{try{let u=s.toLowerCase();if(u.startsWith("0x")&&(u=u.substring(2)),!/^[0-9a-f]{1,8}$/.test(u)){a.push({line:i+1,value:s,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const c=parseInt(u,16);if(o.has(c))return;o.add(c),n.push(c)}catch(u){const c=u instanceof Error?u.message:String(u);a.push({line:i+1,value:s,error:c||"Failed to parse as hexadecimal number."})}}),{validSeeds:n,errors:a}}getVCountForTimer0(r,t){for(const[n,a,o]of r.vcountTimerRanges)if(t>=a&&t<=o)return n;return r.vcountTimerRanges.length>0?r.vcountTimerRanges[0][0]:96}}const K=2,re=1,pe={requiredFeatures:[],powerPreference:"high-performance"},Ce={workgroupSize:256,maxWorkgroupsPerDispatchY:256,candidateCapacityPerDispatch:4096},ht=K*Uint32Array.BYTES_PER_ELEMENT,yt=4294967295;function Me(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}const wt=Me;async function ve(e){if(!Me())throw new Error("WebGPU is not available in this environment");const t=await navigator.gpu.requestAdapter({powerPreference:pe.powerPreference});if(!t)throw new Error("Failed to acquire WebGPU adapter");const n={requiredFeatures:pe.requiredFeatures,requiredLimits:e?.requiredLimits,label:"seed-search-device"},[a,o]=await Promise.all([t.requestDevice(n),Rt(t)]);let s=!1;const i=a.lost.then(f=>(s=!0,console.warn("[webgpu] device lost:",f.message),f)),u=St(t,a),c=a.limits;return{getAdapter:()=>t,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>c,getCapabilities:()=>u,getGpuProfile:()=>o,deriveSearchJobLimits:f=>bt(u.limits,o,f),isLost:()=>s,waitForLoss:()=>i,getSupportedWorkgroupSize:f=>De(u.limits,f)}}function St(e,r){const t=new Set;return e.features.forEach(n=>t.add(n)),{limits:r.limits,features:t}}function bt(e,r,t){const n={...Ce,...t},a=Tt(n,r),o=De(e,a.workgroupSize),s=q(e.maxComputeWorkgroupsPerDimension),i=a.maxWorkgroupsPerDispatch??s,u=N(Math.min(i,s),"maxWorkgroupsPerDispatch"),c=Math.max(1,Math.floor(yt/Math.max(1,o*u))),f=a.maxWorkgroupsPerDispatchY??s,d=N(Math.min(f,s,c),"maxWorkgroupsPerDispatchY"),m=Math.max(1,u*d),g=o*m,y=a.maxMessagesPerDispatch??g,w=N(Math.min(y,g),"maxMessagesPerDispatch"),p=Math.max(1,Math.floor(q(e.maxStorageBufferBindingSize)/ht)),S=a.candidateCapacityPerDispatch??p,_=N(Math.min(S,p),"candidateCapacityPerDispatch");return{workgroupSize:o,maxWorkgroupsPerDispatch:u,maxWorkgroupsPerDispatchY:d,maxMessagesPerDispatch:w,candidateCapacityPerDispatch:_}}function De(e,r){const t=Ce.workgroupSize,n=typeof r=="number"&&Number.isFinite(r)&&r>0?Math.floor(r):t,a=q(e.maxComputeWorkgroupSizeX),o=q(e.maxComputeInvocationsPerWorkgroup),s=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(n,s))}function q(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function N(e,r){if(!Number.isFinite(e)||e<=0)throw new Error(`${r} must be a positive finite number`);return Math.floor(e)}async function Rt(e){const r=e,t=Pt(),n=await xt(r),o=!!e.isFallbackAdapter;if(Et(t))return{kind:"mobile",source:"user-agent",userAgent:t,adapterInfo:n,isFallbackAdapter:o};const s=Dt(n);return s?{kind:s,source:"adapter-info",userAgent:t,adapterInfo:n,isFallbackAdapter:o}:o?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:n,isFallbackAdapter:o}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:n,isFallbackAdapter:o}}async function xt(e){const r=e.requestAdapterInfo;if(typeof r=="function")try{return await r.call(e)}catch(t){console.warn("[webgpu] requestAdapterInfo failed:",t);return}}function Pt(){return typeof navigator>"u"?"":navigator.userAgent||""}function Et(e){return e?/Android|iPhone|iPad|iPod|Mobile|Silk|Kindle|Opera Mini|Opera Mobi/i.test(e):!1}const Ct=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],Mt=["intel","iris","uhd","hd graphics","radeon graphics","apple"],vt=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"];function Dt(e){if(!e)return;const r=[e.vendor,e.architecture,e.device,e.description].filter(Boolean).join(" ").toLowerCase();if(r){if(Z(r,Ct))return"mobile";if(Z(r,Mt))return"integrated";if(Z(r,vt))return"discrete"}}function Z(e,r){return r.some(t=>e.includes(t))}function Tt(e,r){return r.kind==="mobile"||r.kind==="integrated"?{...e,maxWorkgroupsPerDispatchY:1}:e}var kt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
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
  workgroups_per_dispatch_y : u32,\r
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
  let invocations_per_y = max(config.groups_per_dispatch, 1u) * max(config.configured_workgroup_size, 1u);\r
  let global_linear_index = global_id.x + global_id.y * invocations_per_y;\r
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
}`;const At=/WORKGROUP_SIZE_PLACEHOLDER/g,Ot="seed-search-kernel-module",Bt="seed-search-kernel",Lt="seed-search-kernel-layout",Wt="seed-search-kernel-bind-layout";function It(e){return kt.replace(At,String(e))}function Ut(e,r){const t=e.createShaderModule({label:Ot,code:It(r)});t.getCompilationInfo?.().then(s=>{s.messages.length>0&&console.warn("[seed-search-kernel] compilation diagnostics",s.messages.map(i=>({message:i.message,line:i.lineNum,column:i.linePos,type:i.type})))}).catch(s=>{console.warn("[seed-search-kernel] compilation info failed",s)});const n=e.createBindGroupLayout({label:Wt,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),a=e.createPipelineLayout({label:Lt,bindGroupLayouts:[n]});return{pipeline:e.createComputePipeline({label:Bt,layout:a,compute:{module:t,entryPoint:"sha1_generate"}}),layout:n}}const zt=33,_e=256,Q=new Uint32Array([0]);function Te(e,r){const t={context:r??null,pipeline:null,bindGroupLayout:null,configBuffer:null,configData:null,matchOutputBuffer:null,readbackBuffer:null,matchBufferSize:0,targetBuffer:null,targetCapacity:0,workgroupSize:0,candidateCapacity:0,currentLimits:null},n=async d=>{t.context||(t.context=await ve());const m=t.context.getDevice(),g=t.context.getSupportedWorkgroupSize(d.workgroupSize),y=!t.currentLimits||t.workgroupSize!==g||t.candidateCapacity!==d.candidateCapacityPerDispatch;if(!t.pipeline||y){const{pipeline:p,layout:S}=Ut(m,g);t.pipeline=p,t.bindGroupLayout=S,a(m),o(m,d.candidateCapacityPerDispatch),t.workgroupSize=g,t.candidateCapacity=d.candidateCapacityPerDispatch,t.currentLimits=d;return}(!t.matchOutputBuffer||!t.readbackBuffer)&&o(m,d.candidateCapacityPerDispatch),(!t.configBuffer||!t.configData)&&a(m),t.currentLimits=d},a=d=>{const m=new Uint32Array(zt),g=ee(m.byteLength);t.configBuffer?.destroy(),t.configBuffer=d.createBuffer({label:"seed-search-config",size:g,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.configData=m},o=(d,m)=>{const g=re+m*K,y=ee(g*Uint32Array.BYTES_PER_ELEMENT);t.matchOutputBuffer?.destroy(),t.matchOutputBuffer=d.createBuffer({label:"seed-search-output",size:y,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),t.readbackBuffer?.destroy(),t.readbackBuffer=d.createBuffer({label:"seed-search-readback",size:y,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),t.matchBufferSize=y};return{ensureConfigured:n,setTargetSeeds:d=>{if(!t.context)throw new Error("SeedSearchEngine is not configured yet");const m=t.context.getDevice(),g=d.length,y=1+g,w=ee(y*Uint32Array.BYTES_PER_ELEMENT);(!t.targetBuffer||t.targetCapacity<g)&&(t.targetBuffer?.destroy(),t.targetBuffer=m.createBuffer({label:"seed-search-target-seeds",size:w,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.targetCapacity=g);const p=new Uint32Array(y);p[0]=g>>>0;for(let S=0;S<g;S+=1)p[1+S]=d[S]>>>0;m.queue.writeBuffer(t.targetBuffer,0,p.buffer,p.byteOffset,p.byteLength)},executeSegment:async d=>{if(!t.context||!t.pipeline||!t.bindGroupLayout)throw new Error("SeedSearchEngine is not ready");if(!t.configBuffer||!t.configData||!t.matchOutputBuffer||!t.readbackBuffer)throw new Error("SeedSearchEngine buffers are not ready");if(!t.targetBuffer)throw new Error("Target seed buffer is not prepared");const m=t.context.getDevice(),g=m.queue,y=Math.max(1,d.workgroupCountX??1),w=Math.max(1,d.workgroupCountY??1);Y(),g.writeBuffer(t.matchOutputBuffer,0,Q.buffer,Q.byteOffset,Q.byteLength);const p=t.configData;p.set(d.configWords),p[0]=d.messageCount>>>0,p[22]=y>>>0,p[23]=t.workgroupSize>>>0,p[24]=t.candidateCapacity>>>0,p[32]=w>>>0,g.writeBuffer(t.configBuffer,0,p.buffer,p.byteOffset,p.byteLength),Y();const S=m.createBindGroup({label:`seed-search-bind-group-${d.id}`,layout:t.bindGroupLayout,entries:[{binding:0,resource:{buffer:t.configBuffer}},{binding:1,resource:{buffer:t.targetBuffer}},{binding:2,resource:{buffer:t.matchOutputBuffer}}]}),_=m.createCommandEncoder({label:`seed-search-encoder-${d.id}`}),P=_.beginComputePass({label:`seed-search-pass-${d.id}`});P.setPipeline(t.pipeline),P.setBindGroup(0,S),P.dispatchWorkgroups(y,w),P.end(),_.copyBufferToBuffer(t.matchOutputBuffer,0,t.readbackBuffer,0,t.matchBufferSize);const l=_.finish();g.submit([l]),await g.onSubmittedWorkDone(),Y(),await t.readbackBuffer.mapAsync(GPUMapMode.READ,0,t.matchBufferSize);const h=t.readbackBuffer.getMappedRange(0,t.matchBufferSize),b=new Uint32Array(h.slice(0));t.readbackBuffer.unmap(),Y();const R=b[0]??0,x=Math.min(R,t.candidateCapacity),v=Math.min(b.length,re+x*K);return{words:b.slice(0,v),matchCount:x}},dispose:()=>{t.configBuffer?.destroy(),t.matchOutputBuffer?.destroy(),t.readbackBuffer?.destroy(),t.targetBuffer?.destroy(),t.context=null,t.pipeline=null,t.bindGroupLayout=null,t.configBuffer=null,t.configData=null,t.matchOutputBuffer=null,t.readbackBuffer=null,t.targetBuffer=null,t.targetCapacity=0,t.currentLimits=null},getWorkgroupSize:()=>t.workgroupSize,getCandidateCapacity:()=>t.candidateCapacity}}function ee(e){return Math.ceil(e/_e)*_e}function Y(){return typeof performance<"u"?performance.now():Date.now()}const Ft=1024,Gt=500;function Nt(e){const r=new _t,t=e??Te(),n={isRunning:!1,isPaused:!1,shouldStop:!1,job:null,progress:null,callbacks:null,timer:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1},lastProgressUpdate:0},a=async(l,h,b)=>{if(n.isRunning)throw new Error("Seed search is already running");n.isRunning=!0,n.isPaused=!1,n.shouldStop=b?.aborted??!1,n.job=l,n.callbacks=h,n.lastProgressUpdate=0,n.progress={currentStep:0,totalSteps:l.summary.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0,currentDateTime:l.timePlan?new Date(l.timePlan.startDayTimestampMs).toISOString():void 0},g(),d(!1);let R;if(b){const x=()=>{n.shouldStop=!0};b.addEventListener("abort",x),R=()=>b.removeEventListener("abort",x),n.abortCleanup=R}try{if(l.summary.totalMessages===0){h.onComplete("探索対象の組み合わせが存在しません");return}await t.ensureConfigured(l.limits),t.setTargetSeeds(l.targetSeeds);for(const x of l.segments){if(n.shouldStop||(await f(),n.shouldStop))break;const{words:v,matchCount:C}=await t.executeSegment(x);if(await s(x,v,C),n.shouldStop)break}o()}catch(x){const v=x instanceof Error?x.message:"WebGPU検索中に不明なエラーが発生しました",C=globalThis.GPUValidationError,D=C&&x instanceof C?"WEBGPU_VALIDATION_ERROR":void 0;throw h.onError(v,D),x}finally{R?.(),n.abortCleanup=void 0,n.isRunning=!1,n.isPaused=!1,p(),n.job=null,n.callbacks=null,n.progress=null,n.shouldStop=!1,n.lastProgressUpdate=0}},o=()=>{const l=n.callbacks,h=n.progress;if(!(!l||!h)){if(m(h),n.shouldStop){l.onStopped("検索を停止しました",h);return}l.onProgress(h),l.onComplete(`検索が完了しました。${h.matchesFound}件ヒットしました。`)}},s=async(l,h,b)=>{const R=n.job,x=n.callbacks,v=n.progress;if(!R||!x||!v)return;const C=Math.max(l.rangeSeconds,1),D=C*Math.max(l.vcountCount,1),X=re,O=K;for(let B=0;B<b&&!(n.shouldStop||B%Ft===0&&(await f(),n.shouldStop));B+=1){const W=X+B*O,j=h[W],L=h[W+1]>>>0,I=l.globalMessageOffset+j,G=Math.floor(I/D),oe=I-G*D,se=Math.floor(oe/C),Oe=oe-se*C,ie=l.timer0Min+G,ue=l.vcountMin+se,ce=fe(R.timePlan,Oe),de=r.generateMessage(R.conditions,ie,ue,ce,l.keyCode),{hash:Be,seed:le,lcgSeed:Le}=r.calculateSeed(de);le!==L&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:L,cpuSeed:le,messageIndex:I});const We={seed:L,datetime:ce,timer0:ie,vcount:ue,keyCode:l.keyCode,conditions:R.conditions,message:de,sha1Hash:Be,lcgSeed:Le,isMatch:!0};x.onResult(We),v.matchesFound+=1}if(l.messageCount>0){const B=l.messageCount-1,W=l.globalMessageOffset+B,j=Math.floor(W/D),L=W-j*D,I=Math.floor(L/C),G=L-I*C;v.currentDateTime=fe(R.timePlan,G).toISOString()}v.currentStep+=l.messageCount,d(!0)},i=()=>{!n.isRunning||n.isPaused||(n.isPaused=!0,y(),n.callbacks?.onPaused())},u=()=>{!n.isRunning||!n.isPaused||(n.isPaused=!1,w(),n.callbacks?.onResumed())},c=()=>{n.isRunning&&(n.shouldStop=!0,n.isPaused=!1,w())},f=async()=>{for(;n.isPaused&&!n.shouldStop;)await P(25)},d=l=>{const h=n.callbacks,b=n.progress;if(!h||!b)return;const R=Date.now();l&&b.currentStep<b.totalSteps&&R-n.lastProgressUpdate<Gt||(m(b),h.onProgress(b),n.lastProgressUpdate=R)},m=l=>{const h=S();l.elapsedTime=h,l.estimatedTimeRemaining=_(l.currentStep,l.totalSteps,h)},g=()=>{n.timer.cumulativeRunTime=0,n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1},y=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},w=()=>{n.timer.isPaused&&(n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1)},p=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},S=()=>n.timer.isPaused?n.timer.cumulativeRunTime:n.timer.cumulativeRunTime+(Date.now()-n.timer.segmentStartTime),_=(l,h,b)=>{if(l===0||l>=h)return 0;const R=b/l;return Math.round(R*(h-l))},P=l=>new Promise(h=>setTimeout(h,l));return{run:a,pause:i,resume:u,stop:c}}const ke=self,Yt=3n,he=0x100000000n,M={isRunning:!1,isPaused:!1};let A=null,$=null,te=null,H=null;function Ht(e,r){if(r<=0||e.maxMessagesPerDispatch<=0)return e.candidateCapacityPerDispatch;const t=Yt*BigInt(e.maxMessagesPerDispatch)*BigInt(r),n=Number((t+he-1n)/he);return Math.max(1,n)}async function Ae(){return te||(te=ve()),te}async function Vt(){if(A)return A;const e=await Ae(),r=Te(void 0,e);return A=Nt(r),A}async function Kt(){return H||(H=(await Ae()).deriveSearchJobLimits(),H)}function E(e){ke.postMessage(e)}function qt(){E({type:"READY",message:"WebGPU worker initialized"})}function F(){M.isRunning=!1,M.isPaused=!1,$=null}function $t(){return wt()?!0:(E({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Xt(e){if(M.isRunning){E({type:"ERROR",error:"Search is already running"});return}if(!e.conditions||!e.targetSeeds){E({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!$t())return;M.isRunning=!0,M.isPaused=!1;let r,t;try{const[a,o]=await Promise.all([Kt(),Vt()]),s=Ht(a,e.targetSeeds.length),i={...a,candidateCapacityPerDispatch:Math.min(a.candidateCapacityPerDispatch,s)};r=qe(e.conditions,e.targetSeeds,{limits:i}),t=o}catch(a){F();const o=a instanceof Error?a.message:"検索条件の解析中にエラーが発生しました";E({type:"ERROR",error:o,errorCode:"WEBGPU_CONTEXT_ERROR"});return}$=new AbortController;const n={onProgress:a=>{E({type:"PROGRESS",progress:a})},onResult:a=>{E({type:"RESULT",result:a})},onComplete:a=>{F(),E({type:"COMPLETE",message:a})},onError:(a,o)=>{F(),E({type:"ERROR",error:a,errorCode:o})},onPaused:()=>{M.isPaused=!0,E({type:"PAUSED"})},onResumed:()=>{M.isPaused=!1,E({type:"RESUMED"})},onStopped:(a,o)=>{F(),E({type:"STOPPED",message:a,progress:o})}};try{await t.run(r,n,$.signal)}catch(a){if(!M.isRunning)return;F();const o=a instanceof Error?a.message:"WebGPU search failed with unknown error";E({type:"ERROR",error:o,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function jt(){!M.isRunning||M.isPaused||A?.pause()}function Jt(){!M.isRunning||!M.isPaused||A?.resume()}function Zt(){M.isRunning&&(A?.stop(),$?.abort())}qt();ke.onmessage=e=>{const r=e.data;switch(r.type){case"START_SEARCH":Xt(r);break;case"PAUSE_SEARCH":jt();break;case"RESUME_SEARCH":Jt();break;case"STOP_SEARCH":Zt();break;default:E({type:"ERROR",error:`Unknown request type: ${r.type}`})}};
