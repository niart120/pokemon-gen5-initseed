const Se={B:{JPN:{nazo:[35741456,35741708,35741708,35741784,35741784],vcountTimerRanges:[{vcount:96,timer0Min:3193,timer0Max:3194}]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[{vcount:96,timer0Min:3204,timer0Max:3205}]},USA:{nazo:[35741872,35742124,35742124,35742200,35742200],vcountTimerRanges:[{vcount:96,timer0Min:3195,timer0Max:3196}]},GER:{nazo:[35741680,35741932,35741932,35742008,35742008],vcountTimerRanges:[{vcount:95,timer0Min:3191,timer0Max:3192}]},FRA:{nazo:[35741744,35741996,35741996,35742072,35742072],vcountTimerRanges:[{vcount:95,timer0Min:3187,timer0Max:3188}]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[{vcount:96,timer0Min:3206,timer0Max:3207}]},ITA:{nazo:[35741616,35741868,35741868,35741944,35741944],vcountTimerRanges:[{vcount:95,timer0Min:3178,timer0Max:3179}]}},W:{JPN:{nazo:[35741488,35741740,35741740,35741816,35741816],vcountTimerRanges:[{vcount:95,timer0Min:3175,timer0Max:3177}]},KOR:{nazo:[35743664,35743916,35743916,35743992,35743992],vcountTimerRanges:[{vcount:96,timer0Min:3195,timer0Max:3196}]},USA:{nazo:[35741904,35742156,35742156,35742232,35742232],vcountTimerRanges:[{vcount:96,timer0Min:3198,timer0Max:3200}]},GER:{nazo:[35741712,35741964,35741964,35742040,35742040],vcountTimerRanges:[{vcount:96,timer0Min:3194,timer0Max:3195}]},FRA:{nazo:[35741776,35742028,35742028,35742104,35742104],vcountTimerRanges:[{vcount:95,timer0Min:3182,timer0Max:3183}]},SPA:{nazo:[35741808,35742060,35742060,35742136,35742136],vcountTimerRanges:[{vcount:95,timer0Min:3184,timer0Max:3185}]},ITA:{nazo:[35741648,35741900,35741900,35741976,35741976],vcountTimerRanges:[{vcount:96,timer0Min:3195,timer0Max:3196}]}},B2:{JPN:{nazo:[34187484,33790665,35649968,35650052,35650052],vcountTimerRanges:[{vcount:130,timer0Min:4354,timer0Max:4360}]},KOR:{nazo:[34190860,33793237,35653456,35653540,35653540],vcountTimerRanges:[{vcount:130,timer0Min:4335,timer0Max:4340}]},USA:{nazo:[34189032,33791465,35651600,35651684,35651684],vcountTimerRanges:[{vcount:130,timer0Min:4354,timer0Max:4360}]},GER:{nazo:[34188840,33791337,35651408,35651492,35651492],vcountTimerRanges:[{vcount:129,timer0Min:4325,timer0Max:4328},{vcount:130,timer0Min:4329,timer0Max:4332}]},FRA:{nazo:[34189064,33791481,35651632,35651716,35651716],vcountTimerRanges:[{vcount:130,timer0Min:4340,timer0Max:4344}]},SPA:{nazo:[34188968,33791417,35651536,35651620,35651620],vcountTimerRanges:[{vcount:130,timer0Min:4353,timer0Max:4358}]},ITA:{nazo:[34188776,33791337,35651344,35651428,35651428],vcountTimerRanges:[{vcount:130,timer0Min:4359,timer0Max:4361},{vcount:131,timer0Min:4361,timer0Max:4365}]}},W2:{JPN:{nazo:[34187516,33790709,3565e4,35650084,35650084],vcountTimerRanges:[{vcount:130,timer0Min:4341,timer0Max:4347}]},KOR:{nazo:[34190892,33793281,35653488,35653572,35653572],vcountTimerRanges:[{vcount:129,timer0Min:4324,timer0Max:4329}]},USA:{nazo:[34189096,33791509,35651664,35651748,35651748],vcountTimerRanges:[{vcount:130,timer0Min:4338,timer0Max:4342}]},GER:{nazo:[34188872,33791381,35651440,35651524,35651524],vcountTimerRanges:[{vcount:130,timer0Min:4325,timer0Max:4333}]},FRA:{nazo:[34189096,33791525,35651664,35651748,35651748],vcountTimerRanges:[{vcount:130,timer0Min:4332,timer0Max:4336}]},SPA:{nazo:[34189e3,33791461,35651568,35651652,35651652],vcountTimerRanges:[{vcount:130,timer0Min:4335,timer0Max:4340}]},ITA:{nazo:[34188840,33791381,35651408,35651492,35651492],vcountTimerRanges:[{vcount:130,timer0Min:4351,timer0Max:4356}]}}},ye=[["A",0],["B",1],["Select",2],["Start",3],["[→]",4],["[←]",5],["[↑]",6],["[↓]",7],["R",8],["L",9],["X",10],["Y",11]],k=ye.reduce((e,[t,r])=>(e[t]=r,e),{}),we=ye.length,Le=(1<<we)-1,xe=12287,Ue=[1<<k["[↑]"]|1<<k["[↓]"],1<<k["[←]"]|1<<k["[→]"],1<<k.Select|1<<k.Start|1<<k.L|1<<k.R];function ue(e,t){return Number.isFinite(e)?e&Le:0}function Fe(e){const t=ue(e);return xe^t}function We(e){const t=ue(e);for(const r of Ue)if((t&r)===r)return!0;return!1}function Ie(e){return Fe(e)}function ze(e){const t=ue(e),r=[];for(let o=0;o<we;o+=1)(t&1<<o)!==0&&r.push(o);const n=[],a=1<<r.length;for(let o=0;o<a;o+=1){let c=0;for(let s=0;s<r.length;s+=1)(o&1<<s)!==0&&(c|=1<<r[s]);We(c)||n.push((c^xe)>>>0)}return n}const Z=1e3,ce=60,Re=60,Me=24,be=ce*Re,Ge=be*Me,Ee=Ge*Z;function Ne(e){const t=e.timeRange;if(!t)throw new Error("timeRange is required for seed search");const r=ae("hour",t.hour,0,Me-1),n=ae("minute",t.minute,0,Re-1),a=ae("second",t.second,0,ce-1),o=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,0,0,0),c=new Date(e.dateRange.endYear,e.dateRange.endMonth-1,e.dateRange.endDay,0,0,0),s=o.getTime(),l=c.getTime();if(s>l)throw new Error("開始日が終了日より後に設定されています");const g=Math.floor((l-s)/Ee)+1;if(g<=0)throw new Error("探索日数が検出できませんでした");const p=r.count*n.count*a.count;if(p<=0)throw new Error("時刻レンジの組み合わせ数が0です");const _=new Date(e.dateRange.startYear,e.dateRange.startMonth-1,e.dateRange.startDay,r.start,n.start,a.start,0);return{plan:{dayCount:g,combosPerDay:p,hourRangeStart:r.start,hourRangeCount:r.count,minuteRangeStart:n.start,minuteRangeCount:n.count,secondRangeStart:a.start,secondRangeCount:a.count,startDayTimestampMs:s},firstCombinationDate:_}}function de(e,t){const r=Math.max(e.minuteRangeCount,1),n=Math.max(e.secondRangeCount,1),a=Math.max(e.combosPerDay,1),o=Math.max(0,Math.trunc(t)),c=Math.floor(o/a),s=o-c*a,l=r*n,g=Math.floor(s/l),p=s-g*l,_=Math.floor(p/n),w=p-_*n,M=e.hourRangeStart+g,E=e.minuteRangeStart+_,P=e.secondRangeStart+w,T=e.startDayTimestampMs+c*Ee+M*be*Z+E*ce*Z+P*Z;return new Date(T)}function ae(e,t,r,n){if(!t)throw new Error(`${e} range is required for WebGPU search`);const a=Math.trunc(t.start),o=Math.trunc(t.end);if(Number.isNaN(a)||Number.isNaN(o))throw new Error(`${e} range must be numeric`);if(a<r||o>n)throw new Error(`${e} range must be within ${r} to ${n}`);if(a>o)throw new Error(`${e} range start must be <= end`);return{start:a,end:o,count:o-a+1}}const Ye=100663296,He=4294967295;function Ke(e,t=[],r){const{plan:n,firstCombinationDate:a}=Ne(e),o=qe(r),c=Ze(t),s=Xe(e,n,a),l=$e(s,o),g=l.reduce((p,_)=>p+_.messageCount,0);return{segments:l,targetSeeds:c,timePlan:n,summary:{totalMessages:g,totalSegments:l.length,targetSeedCount:c.length,rangeSeconds:s.rangeSeconds},limits:o,conditions:e}}function $e(e,t){const r=[];if(e.rangeSeconds<=0)return r;const n=Math.max(1,t.workgroupSize*t.maxWorkgroupsPerDispatch),a=Math.min(t.maxMessagesPerDispatch,n),o={startDayOfWeek:e.startDayOfWeek,macLower:e.macLower,data7Swapped:e.data7Swapped,hardwareType:e.hardwareType,nazoSwapped:e.nazoSwapped,startYear:e.startYear,startDayOfYear:e.startDayOfYear,hourRangeStart:e.hourRangeStart,hourRangeCount:e.hourRangeCount,minuteRangeStart:e.minuteRangeStart,minuteRangeCount:e.minuteRangeCount,secondRangeStart:e.secondRangeStart,secondRangeCount:e.secondRangeCount};let c=0,s=0;for(const l of e.keyCodes){const g=Q(l>>>0);for(const p of e.timer0Segments)for(let _=p.timer0Min;_<=p.timer0Max;_+=1){const w=p.vcount;let M=e.rangeSeconds,E=0;const P=Q((w&65535)<<16|_&65535);for(;M>0;){const b=Math.min(M,a),T=Je(b,t),i=()=>je({...o,timer0VcountSwapped:P,keyInputSwapped:g});r.push({id:`seg-${s}`,keyCode:l,timer0:_,vcount:w,messageCount:b,baseSecondOffset:E,globalMessageOffset:c,workgroupCount:T,getUniformWords:i}),M-=b,E+=b,c+=b,s+=1}}}return r}function Je(e,t){const r=Math.max(1,Math.ceil(e/t.workgroupSize)),n=Math.max(1,t.maxWorkgroupsPerDispatch);return Math.min(r,n)}function qe(e){if(!e?.limits)throw new Error("Seed search job limits are required for WebGPU execution");return Ve(e.limits)}function Ve(e){const t=G(e.workgroupSize,"workgroupSize"),r=G(e.maxWorkgroupsPerDispatch,"maxWorkgroupsPerDispatch"),n=G(e.candidateCapacityPerDispatch,"candidateCapacityPerDispatch"),a=G(e.maxMessagesPerDispatch,"maxMessagesPerDispatch"),o=G(e.maxDispatchesInFlight,"maxDispatchesInFlight"),c=Math.max(1,Math.floor(He/Math.max(1,t))),s=Math.min(r,c),l=Math.max(1,t*s),g=Math.min(a,l);return{workgroupSize:t,maxWorkgroupsPerDispatch:s,candidateCapacityPerDispatch:n,maxMessagesPerDispatch:g,maxDispatchesInFlight:o}}function Xe(e,t,r){const n=et(e),a=ze(e.keyInput);if(a.length===0)throw new Error("入力されたキー条件から生成できる組み合わせがありません");const o=tt(e,n);if(o.length===0)throw new Error("timer0の範囲が正しく設定されていません");const c=ot(n.nazo),{macLower:s,data7Swapped:l}=rt(e.macAddress,Qe[e.hardware]),g=t.dayCount*t.combosPerDay;if(g<=0)throw new Error("探索対象の秒数が0以下です");return{rangeSeconds:g,timer0Segments:o,keyCodes:a,nazoSwapped:c,macLower:s,data7Swapped:l,hardwareType:at(e.hardware),startYear:r.getFullYear(),startDayOfYear:st(r),startDayOfWeek:r.getDay(),hourRangeStart:t.hourRangeStart,hourRangeCount:t.hourRangeCount,minuteRangeStart:t.minuteRangeStart,minuteRangeCount:t.minuteRangeCount,secondRangeStart:t.secondRangeStart,secondRangeCount:t.secondRangeCount}}function Ze(e){if(!e||e.length===0)return new Uint32Array(0);const t=[];for(const r of e)typeof r!="number"||!Number.isFinite(r)||t.push(r>>>0);return Uint32Array.from(t)}function G(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function je(e){const t=new Uint32Array(20);return t[0]=e.timer0VcountSwapped>>>0,t[1]=e.macLower>>>0,t[2]=e.data7Swapped>>>0,t[3]=e.keyInputSwapped>>>0,t[4]=e.hardwareType>>>0,t[5]=e.startYear>>>0,t[6]=e.startDayOfYear>>>0,t[7]=e.startDayOfWeek>>>0,t[8]=e.hourRangeStart>>>0,t[9]=e.hourRangeCount>>>0,t[10]=e.minuteRangeStart>>>0,t[11]=e.minuteRangeCount>>>0,t[12]=e.secondRangeStart>>>0,t[13]=e.secondRangeCount>>>0,t[14]=(e.nazoSwapped[0]??0)>>>0,t[15]=(e.nazoSwapped[1]??0)>>>0,t[16]=(e.nazoSwapped[2]??0)>>>0,t[17]=(e.nazoSwapped[3]??0)>>>0,t[18]=(e.nazoSwapped[4]??0)>>>0,t[19]=0,t}const Qe={DS:8,DS_LITE:6,"3DS":9};function et(e){const t=Se[e.romVersion];if(!t)throw new Error(`ROMバージョン ${e.romVersion} は未対応です`);const r=t[e.romRegion];if(!r)throw new Error(`ROMリージョン ${e.romRegion} は未対応です`);return{nazo:[...r.nazo],vcountTimerRanges:r.vcountTimerRanges.map(n=>({...n}))}}function tt(e,t){const{timer0VCountConfig:{useAutoConfiguration:r,timer0Range:{min:n,max:a},vcountRange:{min:o,max:c}}}=e;if(!r){const s=[];for(let l=o;l<=c;l+=1)s.push({timer0Min:n,timer0Max:a,vcount:l});return s}return t.vcountTimerRanges.map(s=>({vcount:s.vcount,timer0Min:s.timer0Min,timer0Max:s.timer0Max}))}function rt(e,t){const r=nt(e),n=(r[4]&255)<<8|r[5]&255,o=((r[0]&255|(r[1]&255)<<8|(r[2]&255)<<16|(r[3]&255)<<24)^Ye^t)>>>0;return{macLower:n,data7Swapped:Q(o)}}function nt(e){const t=new Array(6).fill(0);for(let r=0;r<6;r+=1){const n=e[r]??0;t[r]=(Number(n)&255)>>>0}return t}function at(e){switch(e){case"DS":return 0;case"DS_LITE":return 1;case"3DS":return 2;default:return 0}}function Q(e){return((e&255)<<24|(e>>>8&255)<<16|(e>>>16&255)<<8|e>>>24&255)>>>0}function ot(e){const t=new Uint32Array(e.length);for(let r=0;r<e.length;r+=1)t[r]=Q(e[r]>>>0);return t}function st(e){const t=new Date(e.getFullYear(),0,1),r=e.getTime()-t.getTime();return Math.floor(r/(1440*60*1e3))+1}class le{calculateHash(t){if(t.length!==16)throw new Error("Message must be exactly 16 32-bit words (64 bytes)");const r=1732584193,n=4023233417,a=2562383102,o=271733878,c=3285377520,s=new Array(80);for(let i=0;i<16;i++)s[i]=t[i];for(let i=16;i<80;i++)s[i]=this.leftRotate(s[i-3]^s[i-8]^s[i-14]^s[i-16],1);let l=r,g=n,p=a,_=o,w=c;for(let i=0;i<80;i++){let f;i<20?f=this.leftRotate(l,5)+(g&p|~g&_)+w+s[i]+1518500249&4294967295:i<40?f=this.leftRotate(l,5)+(g^p^_)+w+s[i]+1859775393&4294967295:i<60?f=this.leftRotate(l,5)+(g&p|g&_|p&_)+w+s[i]+2400959708&4294967295:f=this.leftRotate(l,5)+(g^p^_)+w+s[i]+3395469782&4294967295,w=_,_=p,p=this.leftRotate(g,30),g=l,l=f}const M=this.add32(r,l),E=this.add32(n,g),P=this.add32(a,p),b=this.add32(o,_),T=this.add32(c,w);return{h0:M,h1:E,h2:P,h3:b,h4:T}}leftRotate(t,r){return(t<<r|t>>>32-r)>>>0}add32(t,r){return(t+r&4294967295)>>>0}static hashToHex(t,r,n,a,o){return t.toString(16).padStart(8,"0")+r.toString(16).padStart(8,"0")+n.toString(16).padStart(8,"0")+a.toString(16).padStart(8,"0")+o.toString(16).padStart(8,"0")}}let L=null,N=null;async function it(){return L||N||(N=(async()=>{try{const e=await import("./wasm_pkg-DkL6Qplp.js");let t;if(typeof process<"u"&&!!process.versions?.node){const n=await import("./__vite-browser-external-9wXp6ZBx.js"),o=(await import("./__vite-browser-external-9wXp6ZBx.js")).join(process.cwd(),"src/wasm/wasm_pkg_bg.wasm");t={module_or_path:n.readFileSync(o)}}else t={module_or_path:new URL("/pokemon-gen5-initseed/assets/wasm_pkg_bg-DkPxcpUG.wasm",import.meta.url)};return await e.default(t),L={BWGenerationConfig:e.BWGenerationConfig,PokemonGenerator:e.PokemonGenerator,SeedEnumerator:e.SeedEnumerator,EncounterType:e.EncounterType,GameVersion:e.GameVersion,GameMode:e.GameMode,EggSeedEnumeratorJs:e.EggSeedEnumeratorJs,ParentsIVsJs:e.ParentsIVsJs,GenerationConditionsJs:e.GenerationConditionsJs,EverstonePlanJs:e.EverstonePlanJs,IndividualFilterJs:e.IndividualFilterJs,TrainerIds:e.TrainerIds,GenderRatio:e.GenderRatio,StatRange:e.StatRange,EggBootTimingSearchResult:e.EggBootTimingSearchResult,EggBootTimingSearchIterator:e.EggBootTimingSearchIterator,DSConfigJs:e.DSConfigJs,SegmentParamsJs:e.SegmentParamsJs,TimeRangeParamsJs:e.TimeRangeParamsJs,SearchRangeParamsJs:e.SearchRangeParamsJs,MtSeedBootTimingSearchIterator:e.MtSeedBootTimingSearchIterator,MtSeedBootTimingSearchResult:e.MtSeedBootTimingSearchResult,MtSeedBootTimingSearchResults:e.MtSeedBootTimingSearchResults,calculate_game_offset:e.calculate_game_offset,sha1_hash_batch:e.sha1_hash_batch,mt_seed_search_segment:e.mt_seed_search_segment},L}catch(e){throw console.error("Failed to load WebAssembly module:",e),L=null,N=null,e}})(),N)}function ut(){if(!L)throw new Error("WebAssembly module not initialized. Call initWasm() first.");return L}function me(){return L!==null}const ct={DS:8,DS_LITE:6,"3DS":9};class dt{sha1;useWasm=!1;constructor(){this.sha1=new le}async initializeWasm(){try{return await it(),this.useWasm=!0,!0}catch(t){return console.warn("WebAssembly initialization failed, falling back to TypeScript:",t),this.useWasm=!1,!1}}isUsingWasm(){return this.useWasm&&me()}getWasmModule(){return ut()}setUseWasm(t){if(t&&!me()){console.warn("Cannot enable WebAssembly: module not initialized");return}this.useWasm=t}getROMParameters(t,r){const n=Se[t];if(!n)return console.error(`ROM version not found: ${t}`),null;const a=n[r];return a?{nazo:[...a.nazo],vcountTimerRanges:a.vcountTimerRanges.map(o=>({vcount:o.vcount,timer0Min:o.timer0Min,timer0Max:o.timer0Max}))}:(console.error(`ROM region not found: ${r} for version ${t}`),null)}toLittleEndian32(t){return((t&255)<<24|(t>>>8&255)<<16|(t>>>16&255)<<8|t>>>24&255)>>>0}toLittleEndian16(t){return(t&255)<<8|t>>>8&255}getDayOfWeek(t,r,n){return new Date(t,r-1,n).getDay()}generateMessage(t,r,n,a,o){const c=this.getROMParameters(t.romVersion,t.romRegion);if(!c)throw new Error(`No parameters found for ${t.romVersion} ${t.romRegion}`);const s=new Array(16).fill(0);for(let R=0;R<5;R++)s[R]=this.toLittleEndian32(c.nazo[R]);s[5]=this.toLittleEndian32(n<<16|r);const l=t.macAddress[4]<<8|t.macAddress[5];s[6]=l;const g=t.macAddress[0]<<0|t.macAddress[1]<<8|t.macAddress[2]<<16|t.macAddress[3]<<24,p=100663296,_=ct[t.hardware],w=g^p^_;s[7]=this.toLittleEndian32(w);const M=a.getFullYear()%100,E=a.getMonth()+1,P=a.getDate(),b=this.getDayOfWeek(a.getFullYear(),E,P),T=Math.floor(M/10)*16+M%10,i=Math.floor(E/10)*16+E%10,f=Math.floor(P/10)*16+P%10,u=Math.floor(b/10)*16+b%10;s[8]=T<<24|i<<16|f<<8|u;const d=a.getHours(),h=a.getMinutes(),m=a.getSeconds(),S=(t.hardware==="DS"||t.hardware==="DS_LITE")&&d>=12?1:0,y=Math.floor(d/10)*16+d%10,x=Math.floor(h/10)*16+h%10,B=Math.floor(m/10)*16+m%10;s[9]=S<<30|y<<24|x<<16|B<<8|0,s[10]=0,s[11]=0;const C=o??Ie(t.keyInput);return s[12]=this.toLittleEndian32(C),s[13]=2147483648,s[14]=0,s[15]=416,s}calculateSeed(t){const r=this.sha1.calculateHash(t),n=BigInt(this.toLittleEndian32(r.h0)),o=BigInt(this.toLittleEndian32(r.h1))<<32n|n,l=o*0x5D588B656C078965n+0x269EC3n;return{seed:Number(l>>32n&0xFFFFFFFFn),hash:le.hashToHex(r.h0,r.h1,r.h2,r.h3,r.h4),lcgSeed:o}}parseTargetSeeds(t){const r=t.split(`
`).map(c=>c.trim()).filter(c=>c.length>0),n=[],a=[],o=new Set;return r.forEach((c,s)=>{try{let l=c.toLowerCase();if(l.startsWith("0x")&&(l=l.substring(2)),!/^[0-9a-f]{1,8}$/.test(l)){a.push({line:s+1,value:c,error:"Invalid hexadecimal format. Expected 1-8 hex digits."});return}const g=parseInt(l,16);if(o.has(g))return;o.add(g),n.push(g)}catch(l){const g=l instanceof Error?l.message:String(l);a.push({line:s+1,value:c,error:g||"Failed to parse as hexadecimal number."})}}),{validSeeds:n,errors:a}}getVCountForTimer0(t,r){for(const n of t.vcountTimerRanges)if(r>=n.timer0Min&&r<=n.timer0Max)return n.vcount;return t.vcountTimerRanges.length>0?t.vcountTimerRanges[0].vcount:96}}const K=2,j=1,fe={requiredFeatures:[],powerPreference:"high-performance"},Pe={workgroupSize:256,candidateCapacityPerDispatch:4096},lt=K*Uint32Array.BYTES_PER_ELEMENT,mt=4294967295,ge={mobile:1,integrated:2,discrete:4,unknown:1},ft=1,pe=8;function Ce(){return typeof navigator<"u"&&typeof navigator.gpu<"u"}const gt=Ce;async function ve(e){if(!Ce())throw new Error("WebGPU is not available in this environment");const r=await navigator.gpu.requestAdapter({powerPreference:fe.powerPreference});if(!r)throw new Error("Failed to acquire WebGPU adapter");const n={requiredFeatures:fe.requiredFeatures,requiredLimits:e?.requiredLimits,label:"seed-search-device"},[a,o]=await Promise.all([r.requestDevice(n),St(r)]);let c=!1;const s=a.lost.then(p=>(c=!0,console.warn("[webgpu] device lost:",p.message),p)),l=pt(r,a),g=a.limits;return{getAdapter:()=>r,getDevice:()=>a,getQueue:()=>a.queue,getLimits:()=>g,getCapabilities:()=>l,getGpuProfile:()=>o,deriveSearchJobLimits:p=>ht(l.limits,o,p),isLost:()=>c,waitForLoss:()=>s,getSupportedWorkgroupSize:p=>De(l.limits,p)}}function pt(e,t){const r=new Set;return e.features.forEach(n=>r.add(n)),{limits:t.limits,features:r}}function ht(e,t,r){const n={...Pe,...r},a=vt(n),o=De(e,a.workgroupSize),c=ee(e.maxComputeWorkgroupsPerDimension),s=a.maxWorkgroupsPerDispatch??c,l=Math.max(1,Math.floor(mt/Math.max(1,o))),g=$(Math.min(s,c,l),"maxWorkgroupsPerDispatch"),p=o*g,_=a.maxMessagesPerDispatch??p,w=$(Math.min(_,p),"maxMessagesPerDispatch"),M=Math.max(1,Math.floor(ee(e.maxStorageBufferBindingSize)/lt)),E=a.candidateCapacityPerDispatch??M,P=$(Math.min(E,M),"candidateCapacityPerDispatch"),b=_t(t,a);return{workgroupSize:o,maxWorkgroupsPerDispatch:g,maxMessagesPerDispatch:w,candidateCapacityPerDispatch:P,maxDispatchesInFlight:b}}function De(e,t){const r=Pe.workgroupSize,n=typeof t=="number"&&Number.isFinite(t)&&t>0?Math.floor(t):r,a=ee(e.maxComputeWorkgroupSizeX),o=ee(e.maxComputeInvocationsPerWorkgroup),c=Math.max(1,Math.min(a,o));return Math.max(1,Math.min(n,c))}function ee(e){return typeof e!="number"||!Number.isFinite(e)||e<=0?Number.MAX_SAFE_INTEGER:Math.floor(e)}function $(e,t){if(!Number.isFinite(e)||e<=0)throw new Error(`${t} must be a positive finite number`);return Math.floor(e)}function _t(e,t){if(typeof t.maxDispatchesInFlight=="number")return $(Math.min(t.maxDispatchesInFlight,pe),"maxDispatchesInFlight");const r=e.isFallbackAdapter?ft:ge[e.kind]??ge.unknown;return $(Math.min(r,pe),"maxDispatchesInFlight")}async function St(e){const t=yt(),n=!!e.isFallbackAdapter,a=bt();if(a){const o={description:a.renderer};return{kind:a.kind,source:"webgl",userAgent:t,adapterInfo:o,isFallbackAdapter:n}}return n?{kind:"integrated",source:"fallback",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}:{kind:"unknown",source:"unknown",userAgent:t,adapterInfo:void 0,isFallbackAdapter:n}}function yt(){return typeof navigator>"u"?"":navigator.userAgent||""}const wt=["mali","adreno","powervr","apple gpu","apple m","snapdragon","exynos"],xt=["nvidia","geforce","rtx","gtx","quadro","amd","radeon rx","radeon pro","arc"],Rt=["intel","iris","uhd","hd graphics","radeon graphics","apple"];function oe(e,t){return t.some(r=>e.includes(r))}function Mt(e){if(!e)return;const t=e.toLowerCase();if(oe(t,wt))return"mobile";if(oe(t,xt))return"discrete";if(oe(t,Rt))return"integrated"}function bt(){const e=Et();if(!e)return;const t=Mt(e);if(t)return{kind:t,renderer:e}}function Et(){const e=Pt();if(e)try{const t=Ct(e);if(!t)return;const r=t.getExtension("WEBGL_debug_renderer_info");if(!r)return;const n=t.getParameter(r.UNMASKED_RENDERER_WEBGL),a=t.getExtension("WEBGL_lose_context");return a&&a.loseContext(),typeof n=="string"?n:void 0}catch(t){console.warn("[webgpu] webgl renderer detection failed:",t);return}}function Pt(){if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(1,1);if(typeof document<"u"&&typeof document.createElement=="function"){const e=document.createElement("canvas");return e.width=1,e.height=1,e}}function Ct(e){const t=e,r=t.getContext;if(typeof r!="function")return null;const n=a=>r.call(t,a)??null;return n("webgl2")??n("webgl")}function vt(e,t){return e}var Dt=`const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;\r
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
}\r
`;const Tt=/WORKGROUP_SIZE_PLACEHOLDER/g,Bt="seed-search-kernel-module",Ot="seed-search-kernel",At="seed-search-kernel-layout",kt="seed-search-kernel-bind-layout";function Lt(e){return Dt.replace(Tt,String(e))}function Ut(e,t){const r=e.createShaderModule({label:Bt,code:Lt(t)});r.getCompilationInfo?.().then(c=>{c.messages.length>0&&console.warn("[seed-search-kernel] compilation diagnostics",c.messages.map(s=>({message:s.message,line:s.lineNum,column:s.linePos,type:s.type})))}).catch(c=>{console.warn("[seed-search-kernel] compilation info failed",c)});const n=e.createBindGroupLayout({label:kt,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),a=e.createPipelineLayout({label:At,bindGroupLayouts:[n]});return{pipeline:e.createComputePipeline({label:Ot,layout:a,compute:{module:r,entryPoint:"sha1_generate"}}),layout:n}}const Ft=4,he=256,se=new Uint32Array([0]);function Te(e,t){const r={context:t??null,pipeline:null,bindGroupLayout:null,targetBuffer:null,targetCapacity:0,workgroupSize:0,candidateCapacity:0,currentLimits:null,dispatchSlots:[],availableSlots:[],slotWaiters:[],desiredDispatchSlots:1},n=async(i,f)=>{r.context||(r.context=await ve());const u=r.context.getDevice(),d=r.context.getSupportedWorkgroupSize(i.workgroupSize),h=Math.max(1,f?.dispatchSlots??r.desiredDispatchSlots??1),m=!r.currentLimits||r.workgroupSize!==d||r.candidateCapacity!==i.candidateCapacityPerDispatch;if(!r.pipeline||m){const{pipeline:y,layout:x}=Ut(u,d);r.pipeline=y,r.bindGroupLayout=x}r.workgroupSize=d,r.candidateCapacity=i.candidateCapacityPerDispatch,r.currentLimits=i,r.desiredDispatchSlots=h,a(u,h,i.candidateCapacityPerDispatch),r.currentLimits=i},a=(i,f,u)=>{for(const d of r.dispatchSlots)l(i,d,u);for(;r.dispatchSlots.length<f;){const d=r.dispatchSlots.length,h=o(i,d,u);r.dispatchSlots.push(h)}for(;r.dispatchSlots.length>f;){const d=r.dispatchSlots.pop();d&&g(d)}r.availableSlots=[...r.dispatchSlots],r.slotWaiters.length=0},o=(i,f,u)=>{const d=new Uint32Array(Ft),h=Y(d.byteLength),m=i.createBuffer({label:`seed-search-dispatch-state-${f}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),{matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}=s(i,u,f);return{id:f,dispatchStateBuffer:m,dispatchStateData:d,uniformBuffer:null,uniformCapacityWords:0,matchOutputBuffer:S,readbackBuffer:y,matchBufferSize:x}},c=(i,f,u)=>{const d=Y(u*Uint32Array.BYTES_PER_ELEMENT);(!f.uniformBuffer||f.uniformCapacityWords<u)&&(f.uniformBuffer?.destroy(),f.uniformBuffer=i.createBuffer({label:`seed-search-uniform-${f.id}`,size:d,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),f.uniformCapacityWords=u)},s=(i,f,u)=>{const d=j+f*K,h=Y(d*Uint32Array.BYTES_PER_ELEMENT),m=i.createBuffer({label:`seed-search-output-${u}`,size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),S=i.createBuffer({label:`seed-search-readback-${u}`,size:h,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return{matchOutputBuffer:m,readbackBuffer:S,matchBufferSize:h}},l=(i,f,u)=>{const d=j+u*K,h=Y(d*Uint32Array.BYTES_PER_ELEMENT);if(f.matchBufferSize===h)return;f.matchOutputBuffer.destroy(),f.readbackBuffer.destroy();const m=s(i,u,f.id);f.matchOutputBuffer=m.matchOutputBuffer,f.readbackBuffer=m.readbackBuffer,f.matchBufferSize=m.matchBufferSize},g=i=>{i.dispatchStateBuffer.destroy(),i.uniformBuffer?.destroy(),i.matchOutputBuffer.destroy(),i.readbackBuffer.destroy()},p=()=>r.availableSlots.length>0?Promise.resolve(r.availableSlots.pop()):new Promise(i=>{r.slotWaiters.push(i)}),_=i=>{const f=r.slotWaiters.shift();if(f){f(i);return}r.availableSlots.push(i)};return{ensureConfigured:n,setTargetSeeds:i=>{if(!r.context)throw new Error("SeedSearchEngine is not configured yet");const f=r.context.getDevice(),u=i.length,d=1+u,h=Y(d*Uint32Array.BYTES_PER_ELEMENT);(!r.targetBuffer||r.targetCapacity<u)&&(r.targetBuffer?.destroy(),r.targetBuffer=f.createBuffer({label:"seed-search-target-seeds",size:h,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.targetCapacity=u);const m=new Uint32Array(d);m[0]=u>>>0;for(let S=0;S<u;S+=1)m[1+S]=i[S]>>>0;f.queue.writeBuffer(r.targetBuffer,0,m.buffer,m.byteOffset,m.byteLength)},executeSegment:async i=>{if(!r.context||!r.pipeline||!r.bindGroupLayout)throw new Error("SeedSearchEngine is not ready");if(!r.targetBuffer)throw new Error("Target seed buffer is not prepared");if(r.dispatchSlots.length===0)throw new Error("Dispatch slots are not configured");const f=r.context.getDevice(),u=f.queue,d=Math.max(1,i.workgroupCount),h=d,m=await p();try{const S=V();u.writeBuffer(m.matchOutputBuffer,0,se.buffer,se.byteOffset,se.byteLength);const y=m.dispatchStateData;y[0]=i.messageCount>>>0,y[1]=i.baseSecondOffset>>>0,y[2]=r.candidateCapacity>>>0,y[3]=0,u.writeBuffer(m.dispatchStateBuffer,0,y.buffer,y.byteOffset,y.byteLength);const x=i.getUniformWords();c(f,m,x.length),u.writeBuffer(m.uniformBuffer,0,x.buffer,x.byteOffset,x.byteLength);const B=V(),C=f.createBindGroup({label:`seed-search-bind-group-${i.id}-slot-${m.id}`,layout:r.bindGroupLayout,entries:[{binding:0,resource:{buffer:m.dispatchStateBuffer}},{binding:1,resource:{buffer:m.uniformBuffer}},{binding:2,resource:{buffer:r.targetBuffer}},{binding:3,resource:{buffer:m.matchOutputBuffer}}]}),R=f.createCommandEncoder({label:`seed-search-encoder-${i.id}`}),O=R.beginComputePass({label:`seed-search-pass-${i.id}`});O.setPipeline(r.pipeline),O.setBindGroup(0,C),O.dispatchWorkgroups(d),O.end(),R.copyBufferToBuffer(m.matchOutputBuffer,0,m.readbackBuffer,0,m.matchBufferSize);const U=R.finish();u.submit([U]),await m.readbackBuffer.mapAsync(GPUMapMode.READ,0,m.matchBufferSize);const A=V(),re=m.readbackBuffer.getMappedRange(0,m.matchBufferSize),W=new Uint32Array(re.slice(0));m.readbackBuffer.unmap();const I=V(),J=W[0]??0,z=Math.min(J,r.candidateCapacity),ne=Math.min(W.length,j+z*K),q={words:W.slice(0,ne),matchCount:z};return e?.onDispatchComplete?.({segmentId:i.id,messageCount:i.messageCount,workgroupCount:h,matchCount:z,candidateCapacity:r.candidateCapacity,timings:{totalMs:I-S,setupMs:B-S,gpuMs:A-B,readbackMs:I-A},timestampMs:I}),q}finally{_(m)}},dispose:()=>{for(const i of r.dispatchSlots)g(i);r.dispatchSlots=[],r.availableSlots=[],r.slotWaiters.length=0,r.targetBuffer?.destroy(),r.context=null,r.pipeline=null,r.bindGroupLayout=null,r.targetBuffer=null,r.targetCapacity=0,r.currentLimits=null},getWorkgroupSize:()=>r.workgroupSize,getCandidateCapacity:()=>r.candidateCapacity,getSupportedLimits:()=>r.context?.getLimits()??null}}function Y(e){return Math.ceil(e/he)*he}function V(){return typeof performance<"u"?performance.now():Date.now()}const Wt=1024,It=500;function zt(e){const t=new dt,r=e??Te(),n={isRunning:!1,isPaused:!1,shouldStop:!1,job:null,progress:null,callbacks:null,timer:{cumulativeRunTime:0,segmentStartTime:0,isPaused:!1},lastProgressUpdate:0},a=async(u,d,h)=>{if(n.isRunning)throw new Error("Seed search is already running");n.isRunning=!0,n.isPaused=!1,n.shouldStop=h?.aborted??!1,n.job=u,n.callbacks=d,n.lastProgressUpdate=0,n.progress={currentStep:0,totalSteps:u.summary.totalMessages,elapsedTime:0,estimatedTimeRemaining:0,matchesFound:0,currentDateTime:u.timePlan?new Date(u.timePlan.startDayTimestampMs).toISOString():void 0},M(),_(!1);let m;if(h){const S=()=>{n.shouldStop=!0};h.addEventListener("abort",S),m=()=>h.removeEventListener("abort",S),n.abortCleanup=m}try{if(u.summary.totalMessages===0){d.onComplete("探索対象の組み合わせが存在しません");return}const S=Math.max(1,Math.min(u.limits.maxDispatchesInFlight??1,u.segments.length||1));await r.ensureConfigured(u.limits,{dispatchSlots:S}),r.setTargetSeeds(u.targetSeeds);const y=new Set,x=new Set,B=C=>{const R=(async()=>{if(n.shouldStop)return;const{words:O,matchCount:U}=await r.executeSegment(C);if(n.shouldStop)return;const A=(async()=>{await c(C,O,U)})();x.add(A),A.finally(()=>x.delete(A))})();y.add(R),R.finally(()=>y.delete(R))};for(const C of u.segments){if(n.shouldStop||(await p(),n.shouldStop))break;B(C),y.size>=S&&await Promise.race(y)}y.size>0&&await Promise.all(y),x.size>0&&await Promise.all(x),o()}catch(S){const y=S instanceof Error?S.message:"WebGPU検索中に不明なエラーが発生しました",x=globalThis.GPUValidationError,B=x&&S instanceof x?"WEBGPU_VALIDATION_ERROR":void 0;throw d.onError(y,B),S}finally{m?.(),n.abortCleanup=void 0,n.isRunning=!1,n.isPaused=!1,b(),n.job=null,n.callbacks=null,n.progress=null,n.shouldStop=!1,n.lastProgressUpdate=0}},o=()=>{const u=n.callbacks,d=n.progress;if(!(!u||!d)){if(w(d),n.shouldStop){u.onStopped("検索を停止しました",d);return}u.onProgress(d),u.onComplete(`検索が完了しました。${d.matchesFound}件ヒットしました。`)}},c=async(u,d,h)=>{const m=n.job,S=n.callbacks,y=n.progress;if(!m||!S||!y)return;const x=j,B=K;for(let C=0;C<h&&!(n.shouldStop||C%Wt===0&&(await p(),n.shouldStop));C+=1){const R=x+C*B,O=d[R],U=d[R+1]>>>0,A=u.globalMessageOffset+O,re=u.baseSecondOffset+O,W=u.timer0,I=u.vcount,J=de(m.timePlan,re),z=t.generateMessage(m.conditions,W,I,J,u.keyCode),{hash:ne,seed:q,lcgSeed:Ae}=t.calculateSeed(z);q!==U&&console.warn("GPU/CPU seed mismatch detected",{gpuSeed:U,cpuSeed:q,messageIndex:A});const ke={seed:U,datetime:J,timer0:W,vcount:I,keyCode:u.keyCode,conditions:m.conditions,message:z,sha1Hash:ne,lcgSeed:Ae,isMatch:!0};S.onResult(ke),y.matchesFound+=1}if(u.messageCount>0){const C=u.messageCount-1,R=u.baseSecondOffset+C;y.currentDateTime=de(m.timePlan,R).toISOString()}y.currentStep+=u.messageCount,_(!0)},s=()=>{!n.isRunning||n.isPaused||(n.isPaused=!0,E(),n.callbacks?.onPaused())},l=()=>{!n.isRunning||!n.isPaused||(n.isPaused=!1,P(),n.callbacks?.onResumed())},g=()=>{n.isRunning&&(n.shouldStop=!0,n.isPaused=!1,P())},p=async()=>{for(;n.isPaused&&!n.shouldStop;)await f(25)},_=u=>{const d=n.callbacks,h=n.progress;if(!d||!h)return;const m=Date.now();u&&h.currentStep<h.totalSteps&&m-n.lastProgressUpdate<It||(w(h),d.onProgress(h),n.lastProgressUpdate=m)},w=u=>{const d=T();u.elapsedTime=d,u.estimatedTimeRemaining=i(u.currentStep,u.totalSteps,d)},M=()=>{n.timer.cumulativeRunTime=0,n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1},E=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},P=()=>{n.timer.isPaused&&(n.timer.segmentStartTime=Date.now(),n.timer.isPaused=!1)},b=()=>{n.timer.isPaused||(n.timer.cumulativeRunTime+=Date.now()-n.timer.segmentStartTime,n.timer.isPaused=!0)},T=()=>n.timer.isPaused?n.timer.cumulativeRunTime:n.timer.cumulativeRunTime+(Date.now()-n.timer.segmentStartTime),i=(u,d,h)=>{if(u===0||u>=d)return 0;const m=h/u;return Math.round(m*(d-u))},f=u=>new Promise(d=>setTimeout(d,u));return{run:a,pause:s,resume:l,stop:g}}const Be=self,Gt=3n,_e=0x100000000n,D={isRunning:!1,isPaused:!1};let F=null,te=null,ie=null,X=null;function Nt(e,t){if(t<=0||e.maxMessagesPerDispatch<=0)return e.candidateCapacityPerDispatch;const r=Gt*BigInt(e.maxMessagesPerDispatch)*BigInt(t),n=Number((r+_e-1n)/_e);return Math.max(1,n)}async function Oe(){return ie||(ie=ve()),ie}async function Yt(){if(F)return F;const e=await Oe(),t=Te(void 0,e);return F=zt(t),F}async function Ht(){return X||(X=(await Oe()).deriveSearchJobLimits(),X)}function v(e){Be.postMessage(e)}function Kt(){v({type:"READY",message:"WebGPU worker initialized"})}function H(){D.isRunning=!1,D.isPaused=!1,te=null}function $t(){return gt()?!0:(v({type:"ERROR",error:"WebGPU is not supported in this environment",errorCode:"WEBGPU_UNSUPPORTED"}),!1)}async function Jt(e){if(D.isRunning){v({type:"ERROR",error:"Search is already running"});return}if(!e.conditions||!e.targetSeeds){v({type:"ERROR",error:"Missing conditions or target seeds"});return}if(!$t())return;D.isRunning=!0,D.isPaused=!1;let t,r;try{const[a,o]=await Promise.all([Ht(),Yt()]),c=Nt(a,e.targetSeeds.length),s={...a,candidateCapacityPerDispatch:Math.min(a.candidateCapacityPerDispatch,c)};t=Ke(e.conditions,e.targetSeeds,{limits:s}),r=o}catch(a){H();const o=a instanceof Error?a.message:"検索条件の解析中にエラーが発生しました";v({type:"ERROR",error:o,errorCode:"WEBGPU_CONTEXT_ERROR"});return}te=new AbortController;const n={onProgress:a=>{v({type:"PROGRESS",progress:a})},onResult:a=>{v({type:"RESULT",result:a})},onComplete:a=>{H(),v({type:"COMPLETE",message:a})},onError:(a,o)=>{H(),v({type:"ERROR",error:a,errorCode:o})},onPaused:()=>{D.isPaused=!0,v({type:"PAUSED"})},onResumed:()=>{D.isPaused=!1,v({type:"RESUMED"})},onStopped:(a,o)=>{H(),v({type:"STOPPED",message:a,progress:o})}};try{await r.run(t,n,te.signal)}catch(a){if(!D.isRunning)return;H();const o=a instanceof Error?a.message:"WebGPU search failed with unknown error";v({type:"ERROR",error:o,errorCode:"WEBGPU_RUNTIME_ERROR"})}}function qt(){!D.isRunning||D.isPaused||F?.pause()}function Vt(){!D.isRunning||!D.isPaused||F?.resume()}function Xt(){D.isRunning&&(F?.stop(),te?.abort())}Kt();Be.onmessage=e=>{const t=e.data;switch(t.type){case"START_SEARCH":Jt(t);break;case"PAUSE_SEARCH":qt();break;case"RESUME_SEARCH":Vt();break;case"STOP_SEARCH":Xt();break;default:v({type:"ERROR",error:`Unknown request type: ${t.type}`})}};
