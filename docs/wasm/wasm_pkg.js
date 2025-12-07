let wasm;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(takeObject(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let heap = new Array(128).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

const ArrayUtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_arrayutils_free(ptr >>> 0, 1));

const BWGenerationConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bwgenerationconfig_free(ptr >>> 0, 1));

const BitUtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bitutils_free(ptr >>> 0, 1));

const DSConfigJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_dsconfigjs_free(ptr >>> 0, 1));

const EggBootTimingSearchIteratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_eggboottimingsearchiterator_free(ptr >>> 0, 1));

const EggBootTimingSearchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_eggboottimingsearchresult_free(ptr >>> 0, 1));

const EggSeedEnumeratorJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_eggseedenumeratorjs_free(ptr >>> 0, 1));

const EncounterCalculatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_encountercalculator_free(ptr >>> 0, 1));

const EndianUtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_endianutils_free(ptr >>> 0, 1));

const EnumeratedPokemonDataFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_enumeratedpokemondata_free(ptr >>> 0, 1));

const EverstonePlanJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_everstoneplanjs_free(ptr >>> 0, 1));

const ExtraResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_extraresult_free(ptr >>> 0, 1));

const GenderRatioFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_genderratio_free(ptr >>> 0, 1));

const GenerationConditionsJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_generationconditionsjs_free(ptr >>> 0, 1));

const IdAdjustmentSearchIteratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_idadjustmentsearchiterator_free(ptr >>> 0, 1));

const IdAdjustmentSearchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_idadjustmentsearchresult_free(ptr >>> 0, 1));

const IdAdjustmentSearchResultsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_idadjustmentsearchresults_free(ptr >>> 0, 1));

const IndividualFilterJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_individualfilterjs_free(ptr >>> 0, 1));

const MtSeedBootTimingSearchIteratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_mtseedboottimingsearchiterator_free(ptr >>> 0, 1));

const MtSeedBootTimingSearchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_mtseedboottimingsearchresult_free(ptr >>> 0, 1));

const MtSeedBootTimingSearchResultsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_mtseedboottimingsearchresults_free(ptr >>> 0, 1));

const NumberUtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_numberutils_free(ptr >>> 0, 1));

const OffsetCalculatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_offsetcalculator_free(ptr >>> 0, 1));

const PIDCalculatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pidcalculator_free(ptr >>> 0, 1));

const ParentsIVsJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_parentsivsjs_free(ptr >>> 0, 1));

const PersonalityRNGFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_personalityrng_free(ptr >>> 0, 1));

const PokemonGeneratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pokemongenerator_free(ptr >>> 0, 1));

const RawPokemonDataFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_rawpokemondata_free(ptr >>> 0, 1));

const SearchRangeParamsJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_searchrangeparamsjs_free(ptr >>> 0, 1));

const SeedEnumeratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_seedenumerator_free(ptr >>> 0, 1));

const SegmentParamsJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_segmentparamsjs_free(ptr >>> 0, 1));

const ShinyCheckerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_shinychecker_free(ptr >>> 0, 1));

const StatRangeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_statrange_free(ptr >>> 0, 1));

const TidSidResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tidsidresult_free(ptr >>> 0, 1));

const TimeRangeParamsJsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timerangeparamsjs_free(ptr >>> 0, 1));

const TrainerIdsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trainerids_free(ptr >>> 0, 1));

const ValidationUtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_validationutils_free(ptr >>> 0, 1));

/**
 * 配列操作ユーティリティ
 */
export class ArrayUtils {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ArrayUtilsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_arrayutils_free(ptr, 0);
    }
    /**
     * 32bit配列の合計値を計算
     *
     * # Arguments
     * * `array` - 対象配列
     *
     * # Returns
     * 合計値
     * @param {Uint32Array} array
     * @returns {bigint}
     */
    static sum_u32_array(array) {
        const ptr0 = passArray32ToWasm0(array, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.arrayutils_sum_u32_array(ptr0, len0);
        return BigInt.asUintN(64, ret);
    }
    /**
     * 32bit配列の平均値を計算
     *
     * # Arguments
     * * `array` - 対象配列
     *
     * # Returns
     * 平均値
     * @param {Uint32Array} array
     * @returns {number}
     */
    static average_u32_array(array) {
        const ptr0 = passArray32ToWasm0(array, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.arrayutils_average_u32_array(ptr0, len0);
        return ret;
    }
    /**
     * 32bit配列の最大値を取得
     *
     * # Arguments
     * * `array` - 対象配列
     *
     * # Returns
     * 最大値（配列が空の場合は0）
     * @param {Uint32Array} array
     * @returns {number}
     */
    static max_u32_array(array) {
        const ptr0 = passArray32ToWasm0(array, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.arrayutils_max_u32_array(ptr0, len0);
        return ret >>> 0;
    }
    /**
     * 32bit配列の最小値を取得
     *
     * # Arguments
     * * `array` - 対象配列
     *
     * # Returns
     * 最小値（配列が空の場合は0）
     * @param {Uint32Array} array
     * @returns {number}
     */
    static min_u32_array(array) {
        const ptr0 = passArray32ToWasm0(array, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.arrayutils_min_u32_array(ptr0, len0);
        return ret >>> 0;
    }
    /**
     * 配列の重複要素を除去
     *
     * # Arguments
     * * `array` - 対象配列
     *
     * # Returns
     * 重複が除去された配列
     * @param {Uint32Array} array
     * @returns {Uint32Array}
     */
    static deduplicate_u32_array(array) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(array, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.arrayutils_deduplicate_u32_array(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) ArrayUtils.prototype[Symbol.dispose] = ArrayUtils.prototype.free;

/**
 * BW/BW2準拠設定構造体
 */
export class BWGenerationConfig {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BWGenerationConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bwgenerationconfig_free(ptr, 0);
    }
    /**
     * 新しいBW準拠設定を作成
     * @param {GameVersion} version
     * @param {EncounterType} encounter_type
     * @param {number} tid
     * @param {number} sid
     * @param {boolean} sync_enabled
     * @param {number} sync_nature_id
     * @param {boolean} is_shiny_locked
     * @param {boolean} has_shiny_charm
     */
    constructor(version, encounter_type, tid, sid, sync_enabled, sync_nature_id, is_shiny_locked, has_shiny_charm) {
        const ret = wasm.bwgenerationconfig_new(version, encounter_type, tid, sid, sync_enabled, sync_nature_id, is_shiny_locked, has_shiny_charm);
        this.__wbg_ptr = ret >>> 0;
        BWGenerationConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * getter methods
     * @returns {GameVersion}
     */
    get get_version() {
        const ret = wasm.bwgenerationconfig_get_version(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {EncounterType}
     */
    get get_encounter_type() {
        const ret = wasm.bwgenerationconfig_get_encounter_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_tid() {
        const ret = wasm.bwgenerationconfig_get_tid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_sid() {
        const ret = wasm.bwgenerationconfig_get_sid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get get_sync_enabled() {
        const ret = wasm.bwgenerationconfig_get_sync_enabled(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get get_sync_nature_id() {
        const ret = wasm.bwgenerationconfig_get_sync_nature_id(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get get_is_shiny_locked() {
        const ret = wasm.bwgenerationconfig_get_is_shiny_locked(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get get_has_shiny_charm() {
        const ret = wasm.bwgenerationconfig_get_has_shiny_charm(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) BWGenerationConfig.prototype[Symbol.dispose] = BWGenerationConfig.prototype.free;

/**
 * ビット操作ユーティリティ
 */
export class BitUtils {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BitUtilsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bitutils_free(ptr, 0);
    }
    /**
     * 32bit値の左ローテート
     *
     * # Arguments
     * * `value` - ローテートする値
     * * `count` - ローテート回数
     *
     * # Returns
     * ローテートされた値
     * @param {number} value
     * @param {number} count
     * @returns {number}
     */
    static rotate_left_32(value, count) {
        const ret = wasm.bitutils_rotate_left_32(value, count);
        return ret >>> 0;
    }
    /**
     * 32bit値の右ローテート
     *
     * # Arguments
     * * `value` - ローテートする値
     * * `count` - ローテート回数
     *
     * # Returns
     * ローテートされた値
     * @param {number} value
     * @param {number} count
     * @returns {number}
     */
    static rotate_right_32(value, count) {
        const ret = wasm.bitutils_rotate_right_32(value, count);
        return ret >>> 0;
    }
    /**
     * 指定したビット位置の値を取得
     *
     * # Arguments
     * * `value` - 対象の値
     * * `bit_position` - ビット位置（0-31）
     *
     * # Returns
     * 指定ビットの値（0または1）
     * @param {number} value
     * @param {number} bit_position
     * @returns {number}
     */
    static get_bit(value, bit_position) {
        const ret = wasm.bitutils_get_bit(value, bit_position);
        return ret >>> 0;
    }
    /**
     * 指定したビット位置を設定
     *
     * # Arguments
     * * `value` - 対象の値
     * * `bit_position` - ビット位置（0-31）
     * * `bit_value` - 設定する値（0または1）
     *
     * # Returns
     * ビットが設定された値
     * @param {number} value
     * @param {number} bit_position
     * @param {number} bit_value
     * @returns {number}
     */
    static set_bit(value, bit_position, bit_value) {
        const ret = wasm.bitutils_set_bit(value, bit_position, bit_value);
        return ret >>> 0;
    }
    /**
     * ビット数をカウント
     *
     * # Arguments
     * * `value` - 対象の値
     *
     * # Returns
     * 設定されているビット数
     * @param {number} value
     * @returns {number}
     */
    static count_bits(value) {
        const ret = wasm.bitutils_count_bits(value);
        return ret >>> 0;
    }
    /**
     * ビットフィールドを抽出
     *
     * # Arguments
     * * `value` - 対象の値
     * * `start_bit` - 開始ビット位置
     * * `bit_count` - 抽出するビット数
     *
     * # Returns
     * 抽出されたビットフィールド
     * @param {number} value
     * @param {number} start_bit
     * @param {number} bit_count
     * @returns {number}
     */
    static extract_bits(value, start_bit, bit_count) {
        const ret = wasm.bitutils_extract_bits(value, start_bit, bit_count);
        return ret >>> 0;
    }
}
if (Symbol.dispose) BitUtils.prototype[Symbol.dispose] = BitUtils.prototype.free;

/**
 * DS設定パラメータ（公開型）
 *
 * DSハードウェア固有の設定を保持する。
 * TypeScriptから受け取り、Rust内部型に変換して使用する。
 */
export class DSConfigJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DSConfigJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_dsconfigjs_free(ptr, 0);
    }
    /**
     * 新規作成
     * @param {Uint8Array} mac
     * @param {Uint32Array} nazo
     * @param {string} hardware
     */
    constructor(mac, nazo, hardware) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(mac, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray32ToWasm0(nazo, wasm.__wbindgen_export3);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passStringToWasm0(hardware, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len2 = WASM_VECTOR_LEN;
            wasm.dsconfigjs_new(retptr, ptr0, len0, ptr1, len1, ptr2, len2);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            DSConfigJsFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Uint8Array}
     */
    get mac() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dsconfigjs_mac(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Uint32Array}
     */
    get nazo() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dsconfigjs_nazo(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {string}
     */
    get hardware() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dsconfigjs_hardware(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) DSConfigJs.prototype[Symbol.dispose] = DSConfigJs.prototype.free;

/**
 * 砂煙出現内容の種類
 * @enum {0 | 1 | 2}
 */
export const DustCloudContent = Object.freeze({
    /**
     * ポケモン出現
     */
    Pokemon: 0, "0": "Pokemon",
    /**
     * ジュエル類出現
     */
    Jewel: 1, "1": "Jewel",
    /**
     * 進化石類出現
     */
    EvolutionStone: 2, "2": "EvolutionStone",
});

/**
 * 孵化乱数起動時間検索イテレータ
 *
 * 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
 * TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
 * 各セグメントに対してこのイテレータを作成する。
 *
 * ## 使用例（TypeScript）
 * ```typescript
 * const keyCodes = wasm.generate_egg_key_codes(keyInputMask);
 * for (const timer0 of range(timer0Min, timer0Max)) {
 *   for (const vcount of range(vcountMin, vcountMax)) {
 *     for (const keyCode of keyCodes) {
 *       const iterator = new EggBootTimingSearchIterator(..., timer0, vcount, keyCode);
 *       while (!iterator.isFinished) {
 *         const results = iterator.next_batch(256, 3600);
 *         // 結果処理
 *       }
 *       iterator.free();
 *     }
 *   }
 * }
 * ```
 */
export class EggBootTimingSearchIterator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EggBootTimingSearchIteratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_eggboottimingsearchiterator_free(ptr, 0);
    }
    /**
     * コンストラクタ
     *
     * # Arguments
     * - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
     * - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
     * - `time_range`: 時刻範囲パラメータ
     * - `search_range`: 検索範囲パラメータ
     * - `conditions`: 孵化条件
     * - `parents`: 親個体値
     * - `filter_js`: 個体フィルター（オプション）
     * - `consider_npc_consumption`: NPC消費を考慮するか
     * - `game_mode`: ゲームモード
     * - `user_offset`: ユーザー指定オフセット
     * - `advance_count`: 消費数
     * @param {DSConfigJs} ds_config
     * @param {SegmentParamsJs} segment
     * @param {TimeRangeParamsJs} time_range
     * @param {SearchRangeParamsJs} search_range
     * @param {GenerationConditionsJs} conditions
     * @param {ParentsIVsJs} parents
     * @param {IndividualFilterJs | null | undefined} filter_js
     * @param {boolean} consider_npc_consumption
     * @param {GameMode} game_mode
     * @param {bigint} user_offset
     * @param {number} advance_count
     */
    constructor(ds_config, segment, time_range, search_range, conditions, parents, filter_js, consider_npc_consumption, game_mode, user_offset, advance_count) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(ds_config, DSConfigJs);
            _assertClass(segment, SegmentParamsJs);
            _assertClass(time_range, TimeRangeParamsJs);
            _assertClass(search_range, SearchRangeParamsJs);
            _assertClass(conditions, GenerationConditionsJs);
            _assertClass(parents, ParentsIVsJs);
            let ptr0 = 0;
            if (!isLikeNone(filter_js)) {
                _assertClass(filter_js, IndividualFilterJs);
                ptr0 = filter_js.__destroy_into_raw();
            }
            wasm.eggboottimingsearchiterator_new(retptr, ds_config.__wbg_ptr, segment.__wbg_ptr, time_range.__wbg_ptr, search_range.__wbg_ptr, conditions.__wbg_ptr, parents.__wbg_ptr, ptr0, consider_npc_consumption, game_mode, user_offset, advance_count);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            EggBootTimingSearchIteratorFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * 検索が完了したかどうか
     * @returns {boolean}
     */
    get isFinished() {
        const ret = wasm.eggboottimingsearchiterator_is_finished(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * 処理済み秒数
     * @returns {number}
     */
    get processedSeconds() {
        const ret = wasm.eggboottimingsearchiterator_processed_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 総秒数
     * @returns {number}
     */
    get totalSeconds() {
        const ret = wasm.eggboottimingsearchiterator_total_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 進捗率 (0.0 - 1.0)
     * @returns {number}
     */
    get progress() {
        const ret = wasm.eggboottimingsearchiterator_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * 次のバッチを取得
     *
     * - result_limit件見つかったら即return
     * - chunk_seconds秒分処理したら結果がなくても一旦return
     * - 検索範囲を全て処理したらfinished=trueになる
     * @param {number} result_limit
     * @param {number} chunk_seconds
     * @returns {Array<any>}
     */
    next_batch(result_limit, chunk_seconds) {
        const ret = wasm.eggboottimingsearchiterator_next_batch(this.__wbg_ptr, result_limit, chunk_seconds);
        return takeObject(ret);
    }
}
if (Symbol.dispose) EggBootTimingSearchIterator.prototype[Symbol.dispose] = EggBootTimingSearchIterator.prototype.free;

/**
 * 検索結果1件（起動条件 + 個体情報）
 */
export class EggBootTimingSearchResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(EggBootTimingSearchResult.prototype);
        obj.__wbg_ptr = ptr;
        EggBootTimingSearchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EggBootTimingSearchResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_eggboottimingsearchresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get year() {
        const ret = wasm.eggboottimingsearchresult_year(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get month() {
        const ret = wasm.eggboottimingsearchresult_month(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get date() {
        const ret = wasm.eggboottimingsearchresult_date(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get hour() {
        const ret = wasm.eggboottimingsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get minute() {
        const ret = wasm.eggboottimingsearchresult_minute(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get second() {
        const ret = wasm.eggboottimingsearchresult_second(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get timer0() {
        const ret = wasm.eggboottimingsearchresult_timer0(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get vcount() {
        const ret = wasm.eggboottimingsearchresult_vcount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get keyCode() {
        const ret = wasm.eggboottimingsearchresult_key_code(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get lcgSeedHex() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eggboottimingsearchresult_lcg_seed_hex(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {bigint}
     */
    get advance() {
        const ret = wasm.eggboottimingsearchresult_advance(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {boolean}
     */
    get isStable() {
        const ret = wasm.eggboottimingsearchresult_is_stable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {Uint8Array}
     */
    get ivs() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eggboottimingsearchresult_ivs(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    get nature() {
        const ret = wasm.eggboottimingsearchresult_nature(this.__wbg_ptr);
        return ret;
    }
    /**
     * Gender: 0=Male, 1=Female, 2=Genderless
     * @returns {number}
     */
    get gender() {
        const ret = wasm.eggboottimingsearchresult_gender(this.__wbg_ptr);
        return ret;
    }
    /**
     * Ability slot: 0=Ability1, 1=Ability2, 2=Hidden
     * @returns {number}
     */
    get ability() {
        const ret = wasm.eggboottimingsearchresult_ability(this.__wbg_ptr);
        return ret;
    }
    /**
     * Shiny type: 0=Normal (not shiny), 1=Square shiny, 2=Star shiny
     * @returns {number}
     */
    get shiny() {
        const ret = wasm.eggboottimingsearchresult_shiny(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get pid() {
        const ret = wasm.eggboottimingsearchresult_pid(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get hpType() {
        const ret = wasm.eggboottimingsearchresult_hp_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get hpPower() {
        const ret = wasm.eggboottimingsearchresult_hp_power(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get hpKnown() {
        const ret = wasm.eggboottimingsearchresult_hp_known(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get mtSeed() {
        const ret = wasm.eggboottimingsearchresult_mt_seed(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get mtSeedHex() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eggboottimingsearchresult_mt_seed_hex(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) EggBootTimingSearchResult.prototype[Symbol.dispose] = EggBootTimingSearchResult.prototype.free;

/**
 * WASM wrapper for EggSeedEnumerator
 */
export class EggSeedEnumeratorJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EggSeedEnumeratorJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_eggseedenumeratorjs_free(ptr, 0);
    }
    /**
     * @param {bigint} base_seed
     * @param {bigint} user_offset
     * @param {number} count
     * @param {GenerationConditionsJs} conditions
     * @param {ParentsIVsJs} parents
     * @param {IndividualFilterJs} filter
     * @param {boolean} consider_npc_consumption
     * @param {GameMode} game_mode
     */
    constructor(base_seed, user_offset, count, conditions, parents, filter, consider_npc_consumption, game_mode) {
        _assertClass(conditions, GenerationConditionsJs);
        _assertClass(parents, ParentsIVsJs);
        _assertClass(filter, IndividualFilterJs);
        const ret = wasm.eggseedenumeratorjs_new(base_seed, user_offset, count, conditions.__wbg_ptr, parents.__wbg_ptr, filter.__wbg_ptr, consider_npc_consumption, game_mode);
        this.__wbg_ptr = ret >>> 0;
        EggSeedEnumeratorJsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Returns the next egg as a JsValue or undefined if exhausted
     * @returns {any}
     */
    next_egg() {
        const ret = wasm.eggseedenumeratorjs_next_egg(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {number}
     */
    get remaining() {
        const ret = wasm.eggseedenumeratorjs_remaining(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) EggSeedEnumeratorJs.prototype[Symbol.dispose] = EggSeedEnumeratorJs.prototype.free;

/**
 * エンカウント計算エンジン
 */
export class EncounterCalculator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EncounterCalculatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_encountercalculator_free(ptr, 0);
    }
    /**
     * 新しいEncounterCalculatorインスタンスを作成
     */
    constructor() {
        const ret = wasm.encountercalculator_new();
        this.__wbg_ptr = ret >>> 0;
        EncounterCalculatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * エンカウントスロットを計算
     *
     * # Arguments
     * * `version` - ゲームバージョン
     * * `encounter_type` - エンカウントタイプ
     * * `random_value` - 乱数値（32bit）
     *
     * # Returns
     * エンカウントスロット番号（0-11）
     * @param {GameVersion} version
     * @param {EncounterType} encounter_type
     * @param {number} random_value
     * @returns {number}
     */
    static calculate_encounter_slot(version, encounter_type, random_value) {
        const ret = wasm.encountercalculator_calculate_encounter_slot(version, encounter_type, random_value);
        return ret;
    }
    /**
     * スロット番号をテーブルインデックスに変換
     *
     * # Arguments
     * * `encounter_type` - エンカウントタイプ
     * * `slot` - スロット番号
     *
     * # Returns
     * テーブルインデックス
     * @param {EncounterType} encounter_type
     * @param {number} slot
     * @returns {number}
     */
    static slot_to_table_index(encounter_type, slot) {
        const ret = wasm.encountercalculator_slot_to_table_index(encounter_type, slot);
        return ret >>> 0;
    }
    /**
     * 砂煙の出現内容を判定
     *
     * # Arguments
     * * `slot` - 砂煙スロット値（0-2）
     *
     * # Returns
     * 出現内容の種類
     * @param {number} slot
     * @returns {DustCloudContent}
     */
    static get_dust_cloud_content(slot) {
        const ret = wasm.encountercalculator_get_dust_cloud_content(slot);
        return ret;
    }
}
if (Symbol.dispose) EncounterCalculator.prototype[Symbol.dispose] = EncounterCalculator.prototype.free;

/**
 * エンカウントタイプ列挙型
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 11 | 12 | 13 | 20}
 */
export const EncounterType = Object.freeze({
    /**
     * 通常エンカウント（草むら・洞窟・ダンジョン共通）
     */
    Normal: 0, "0": "Normal",
    /**
     * なみのり
     */
    Surfing: 1, "1": "Surfing",
    /**
     * つりざお
     */
    Fishing: 2, "2": "Fishing",
    /**
     * 揺れる草むら（特殊エンカウント）
     */
    ShakingGrass: 3, "3": "ShakingGrass",
    /**
     * 砂煙（特殊エンカウント）
     */
    DustCloud: 4, "4": "DustCloud",
    /**
     * ポケモンの影（特殊エンカウント）
     */
    PokemonShadow: 5, "5": "PokemonShadow",
    /**
     * 水泡（なみのり版特殊エンカウント）
     */
    SurfingBubble: 6, "6": "SurfingBubble",
    /**
     * 水泡釣り（釣り版特殊エンカウント）
     */
    FishingBubble: 7, "7": "FishingBubble",
    /**
     * 固定シンボル（レジェンダリー等）- シンクロ有効
     */
    StaticSymbol: 10, "10": "StaticSymbol",
    /**
     * 御三家受け取り - シンクロ無効
     */
    StaticStarter: 11, "11": "StaticStarter",
    /**
     * 化石復元 - シンクロ無効
     */
    StaticFossil: 12, "12": "StaticFossil",
    /**
     * イベント配布 - シンクロ無効
     */
    StaticEvent: 13, "13": "StaticEvent",
    /**
     * 徘徊ポケモン（ドキュメント仕様準拠）
     */
    Roamer: 20, "20": "Roamer",
});

/**
 * エンディアン変換ユーティリティ
 */
export class EndianUtils {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EndianUtilsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_endianutils_free(ptr, 0);
    }
    /**
     * 32bit値のバイトスワップ
     *
     * # Arguments
     * * `value` - 変換する32bit値
     *
     * # Returns
     * バイトスワップされた値
     * @param {number} value
     * @returns {number}
     */
    static swap_bytes_32(value) {
        const ret = wasm.endianutils_le32_to_be(value);
        return ret >>> 0;
    }
    /**
     * 16bit値のバイトスワップ
     *
     * # Arguments
     * * `value` - 変換する16bit値
     *
     * # Returns
     * バイトスワップされた値
     * @param {number} value
     * @returns {number}
     */
    static swap_bytes_16(value) {
        const ret = wasm.endianutils_swap_bytes_16(value);
        return ret;
    }
    /**
     * 64bit値のバイトスワップ
     *
     * # Arguments
     * * `value` - 変換する64bit値
     *
     * # Returns
     * バイトスワップされた値
     * @param {bigint} value
     * @returns {bigint}
     */
    static swap_bytes_64(value) {
        const ret = wasm.endianutils_swap_bytes_64(value);
        return BigInt.asUintN(64, ret);
    }
    /**
     * ビッグエンディアン32bit値をリトルエンディアンに変換
     * @param {number} value
     * @returns {number}
     */
    static be32_to_le(value) {
        const ret = wasm.endianutils_be32_to_le(value);
        return ret >>> 0;
    }
    /**
     * リトルエンディアン32bit値をビッグエンディアンに変換
     * @param {number} value
     * @returns {number}
     */
    static le32_to_be(value) {
        const ret = wasm.endianutils_le32_to_be(value);
        return ret >>> 0;
    }
}
if (Symbol.dispose) EndianUtils.prototype[Symbol.dispose] = EndianUtils.prototype.free;

export class EnumeratedPokemonData {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(EnumeratedPokemonData.prototype);
        obj.__wbg_ptr = ptr;
        EnumeratedPokemonDataFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EnumeratedPokemonDataFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_enumeratedpokemondata_free(ptr, 0);
    }
    /**
     * @returns {bigint}
     */
    get get_advance() {
        const ret = wasm.eggboottimingsearchresult_advance(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {bigint}
     */
    get get_seed() {
        const ret = wasm.enumeratedpokemondata_get_seed(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get get_pid() {
        const ret = wasm.eggboottimingsearchresult_date(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get get_nature() {
        const ret = wasm.enumeratedpokemondata_get_nature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get get_sync_applied() {
        const ret = wasm.enumeratedpokemondata_get_sync_applied(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get get_ability_slot() {
        const ret = wasm.enumeratedpokemondata_get_ability_slot(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_gender_value() {
        const ret = wasm.enumeratedpokemondata_get_gender_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_encounter_slot_value() {
        const ret = wasm.enumeratedpokemondata_get_encounter_slot_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_encounter_type() {
        const ret = wasm.enumeratedpokemondata_get_encounter_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get get_level_rand_value() {
        const ret = wasm.enumeratedpokemondata_get_level_rand_value(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get get_shiny_type() {
        const ret = wasm.enumeratedpokemondata_get_shiny_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * 任意: 元の RawPokemonData を複製して取得
     * @returns {RawPokemonData}
     */
    into_raw() {
        const ret = wasm.enumeratedpokemondata_into_raw(this.__wbg_ptr);
        return RawPokemonData.__wrap(ret);
    }
}
if (Symbol.dispose) EnumeratedPokemonData.prototype[Symbol.dispose] = EnumeratedPokemonData.prototype.free;

/**
 * WASM wrapper for EverstonePlan
 */
export class EverstonePlanJs {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(EverstonePlanJs.prototype);
        obj.__wbg_ptr = ptr;
        EverstonePlanJsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EverstonePlanJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_everstoneplanjs_free(ptr, 0);
    }
    /**
     * @returns {EverstonePlanJs}
     */
    static get None() {
        const ret = wasm.everstoneplanjs_none();
        return EverstonePlanJs.__wrap(ret);
    }
    /**
     * @param {number} nature_index
     * @returns {EverstonePlanJs}
     */
    static fixed(nature_index) {
        const ret = wasm.everstoneplanjs_fixed(nature_index);
        return EverstonePlanJs.__wrap(ret);
    }
}
if (Symbol.dispose) EverstonePlanJs.prototype[Symbol.dispose] = EverstonePlanJs.prototype.free;

/**
 * Extra処理結果（BW2専用）
 */
export class ExtraResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ExtraResult.prototype);
        obj.__wbg_ptr = ptr;
        ExtraResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ExtraResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_extraresult_free(ptr, 0);
    }
    /**
     * 消費した乱数回数
     * @returns {number}
     */
    get advances() {
        const ret = wasm.__wbg_get_extraresult_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 消費した乱数回数
     * @param {number} arg0
     */
    set advances(arg0) {
        wasm.__wbg_set_extraresult_advances(this.__wbg_ptr, arg0);
    }
    /**
     * 成功フラグ（重複回避完了）
     * @returns {boolean}
     */
    get success() {
        const ret = wasm.__wbg_get_extraresult_success(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * 成功フラグ（重複回避完了）
     * @param {boolean} arg0
     */
    set success(arg0) {
        wasm.__wbg_set_extraresult_success(this.__wbg_ptr, arg0);
    }
    /**
     * 最終的な3つの値
     * @returns {number}
     */
    get value1() {
        const ret = wasm.__wbg_get_extraresult_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 最終的な3つの値
     * @param {number} arg0
     */
    set value1(arg0) {
        wasm.__wbg_set_extraresult_value1(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get value2() {
        const ret = wasm.__wbg_get_extraresult_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set value2(arg0) {
        wasm.__wbg_set_extraresult_value2(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get value3() {
        const ret = wasm.__wbg_get_extraresult_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set value3(arg0) {
        wasm.__wbg_set_extraresult_value3(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get get_advances() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    get get_success() {
        const ret = wasm.extraresult_get_success(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get get_value1() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get get_value2() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get get_value3() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) ExtraResult.prototype[Symbol.dispose] = ExtraResult.prototype.free;

/**
 * ゲームモード列挙型（仕様書準拠）
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6 | 7}
 */
export const GameMode = Object.freeze({
    /**
     * BW 始めから（セーブ有り）
     */
    BwNewGameWithSave: 0, "0": "BwNewGameWithSave",
    /**
     * BW 始めから（セーブ無し）
     */
    BwNewGameNoSave: 1, "1": "BwNewGameNoSave",
    /**
     * BW 続きから
     */
    BwContinue: 2, "2": "BwContinue",
    /**
     * BW2 始めから（思い出リンク済みセーブ有り）
     */
    Bw2NewGameWithMemoryLinkSave: 3, "3": "Bw2NewGameWithMemoryLinkSave",
    /**
     * BW2 始めから（思い出リンク無しセーブ有り）
     */
    Bw2NewGameNoMemoryLinkSave: 4, "4": "Bw2NewGameNoMemoryLinkSave",
    /**
     * BW2 始めから（セーブ無し）
     */
    Bw2NewGameNoSave: 5, "5": "Bw2NewGameNoSave",
    /**
     * BW2 続きから（思い出リンク済み）
     */
    Bw2ContinueWithMemoryLink: 6, "6": "Bw2ContinueWithMemoryLink",
    /**
     * BW2 続きから（思い出リンク無し）
     */
    Bw2ContinueNoMemoryLink: 7, "7": "Bw2ContinueNoMemoryLink",
});

/**
 * ゲームバージョン列挙型
 * @enum {0 | 1 | 2 | 3}
 */
export const GameVersion = Object.freeze({
    B: 0, "0": "B",
    W: 1, "1": "W",
    B2: 2, "2": "B2",
    W2: 3, "3": "W2",
});

export class GenderRatio {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GenderRatioFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_genderratio_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get threshold() {
        const ret = wasm.__wbg_get_genderratio_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set threshold(arg0) {
        wasm.__wbg_set_genderratio_threshold(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get genderless() {
        const ret = wasm.__wbg_get_genderratio_genderless(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set genderless(arg0) {
        wasm.__wbg_set_genderratio_genderless(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} threshold
     * @param {boolean} genderless
     */
    constructor(threshold, genderless) {
        const ret = wasm.genderratio_new(threshold, genderless);
        this.__wbg_ptr = ret >>> 0;
        GenderRatioFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} gender_value
     * @returns {number}
     */
    resolve(gender_value) {
        const ret = wasm.genderratio_resolve(this.__wbg_ptr, gender_value);
        return ret;
    }
}
if (Symbol.dispose) GenderRatio.prototype[Symbol.dispose] = GenderRatio.prototype.free;

/**
 * WASM wrapper for GenerationConditions
 */
export class GenerationConditionsJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GenerationConditionsJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_generationconditionsjs_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get has_nidoran_flag() {
        const ret = wasm.__wbg_get_generationconditionsjs_has_nidoran_flag(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set has_nidoran_flag(arg0) {
        wasm.__wbg_set_generationconditionsjs_has_nidoran_flag(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get uses_ditto() {
        const ret = wasm.__wbg_get_generationconditionsjs_uses_ditto(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set uses_ditto(arg0) {
        wasm.__wbg_set_generationconditionsjs_uses_ditto(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get allow_hidden_ability() {
        const ret = wasm.__wbg_get_generationconditionsjs_allow_hidden_ability(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set allow_hidden_ability(arg0) {
        wasm.__wbg_set_generationconditionsjs_allow_hidden_ability(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get female_parent_has_hidden() {
        const ret = wasm.__wbg_get_generationconditionsjs_female_parent_has_hidden(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set female_parent_has_hidden(arg0) {
        wasm.__wbg_set_generationconditionsjs_female_parent_has_hidden(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get reroll_count() {
        const ret = wasm.__wbg_get_generationconditionsjs_reroll_count(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set reroll_count(arg0) {
        wasm.__wbg_set_generationconditionsjs_reroll_count(this.__wbg_ptr, arg0);
    }
    constructor() {
        const ret = wasm.generationconditionsjs_new();
        this.__wbg_ptr = ret >>> 0;
        GenerationConditionsJsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {EverstonePlanJs} plan
     */
    set_everstone(plan) {
        _assertClass(plan, EverstonePlanJs);
        wasm.generationconditionsjs_set_everstone(this.__wbg_ptr, plan.__wbg_ptr);
    }
    /**
     * @param {TrainerIds} ids
     */
    set_trainer_ids(ids) {
        _assertClass(ids, TrainerIds);
        wasm.generationconditionsjs_set_trainer_ids(this.__wbg_ptr, ids.__wbg_ptr);
    }
    /**
     * @param {GenderRatio} ratio
     */
    set_gender_ratio(ratio) {
        _assertClass(ratio, GenderRatio);
        wasm.generationconditionsjs_set_gender_ratio(this.__wbg_ptr, ratio.__wbg_ptr);
    }
}
if (Symbol.dispose) GenerationConditionsJs.prototype[Symbol.dispose] = GenerationConditionsJs.prototype.free;

/**
 * ID調整検索イテレータ
 *
 * 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
 * TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
 * 各セグメントに対してこのイテレータを作成する。
 */
export class IdAdjustmentSearchIterator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IdAdjustmentSearchIteratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_idadjustmentsearchiterator_free(ptr, 0);
    }
    /**
     * コンストラクタ
     *
     * # Arguments
     * - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
     * - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
     * - `time_range`: 時刻範囲パラメータ
     * - `search_range`: 検索範囲パラメータ
     * - `target_tid`: 検索対象の表ID（-1で指定なし）
     * - `target_sid`: 検索対象の裏ID（-1で指定なし）
     * - `shiny_pid`: 色違いにしたい個体のPID（-1で指定なし）
     * - `game_mode`: ゲームモード (0-7)
     * @param {DSConfigJs} ds_config
     * @param {SegmentParamsJs} segment
     * @param {TimeRangeParamsJs} time_range
     * @param {SearchRangeParamsJs} search_range
     * @param {number} target_tid
     * @param {number} target_sid
     * @param {number} shiny_pid
     * @param {number} game_mode
     */
    constructor(ds_config, segment, time_range, search_range, target_tid, target_sid, shiny_pid, game_mode) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(ds_config, DSConfigJs);
            _assertClass(segment, SegmentParamsJs);
            _assertClass(time_range, TimeRangeParamsJs);
            _assertClass(search_range, SearchRangeParamsJs);
            wasm.idadjustmentsearchiterator_new(retptr, ds_config.__wbg_ptr, segment.__wbg_ptr, time_range.__wbg_ptr, search_range.__wbg_ptr, target_tid, target_sid, shiny_pid, game_mode);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            IdAdjustmentSearchIteratorFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * 検索が完了したかどうか
     * @returns {boolean}
     */
    get isFinished() {
        const ret = wasm.idadjustmentsearchiterator_is_finished(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * 処理済み秒数
     * @returns {number}
     */
    get processedSeconds() {
        const ret = wasm.idadjustmentsearchiterator_processed_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 総秒数
     * @returns {number}
     */
    get totalSeconds() {
        const ret = wasm.idadjustmentsearchiterator_total_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 進捗率 (0.0 - 1.0)
     * @returns {number}
     */
    get progress() {
        const ret = wasm.idadjustmentsearchiterator_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * 次のバッチを取得
     *
     * - max_results件見つかったら即return
     * - chunk_seconds秒分処理したら結果がなくても一旦return
     * - 検索範囲を全て処理したらfinished=trueになる
     * @param {number} max_results
     * @param {number} chunk_seconds
     * @returns {IdAdjustmentSearchResults}
     */
    next_batch(max_results, chunk_seconds) {
        const ret = wasm.idadjustmentsearchiterator_next_batch(this.__wbg_ptr, max_results, chunk_seconds);
        return IdAdjustmentSearchResults.__wrap(ret);
    }
}
if (Symbol.dispose) IdAdjustmentSearchIterator.prototype[Symbol.dispose] = IdAdjustmentSearchIterator.prototype.free;

/**
 * ID調整検索結果1件
 */
export class IdAdjustmentSearchResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(IdAdjustmentSearchResult.prototype);
        obj.__wbg_ptr = ptr;
        IdAdjustmentSearchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IdAdjustmentSearchResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_idadjustmentsearchresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get lcgSeedHigh() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get lcgSeedLow() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get lcgSeedHex() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.idadjustmentsearchresult_lcg_seed_hex(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get tid() {
        const ret = wasm.idadjustmentsearchresult_tid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get sid() {
        const ret = wasm.idadjustmentsearchresult_sid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get shinyType() {
        const ret = wasm.idadjustmentsearchresult_shiny_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get year() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get month() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get day() {
        const ret = wasm.eggboottimingsearchresult_month(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get hour() {
        const ret = wasm.idadjustmentsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get minute() {
        const ret = wasm.eggboottimingsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get second() {
        const ret = wasm.idadjustmentsearchresult_second(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get timer0() {
        const ret = wasm.eggboottimingsearchresult_second(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get vcount() {
        const ret = wasm.idadjustmentsearchresult_vcount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get keyCode() {
        const ret = wasm.eggboottimingsearchresult_vcount(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) IdAdjustmentSearchResult.prototype[Symbol.dispose] = IdAdjustmentSearchResult.prototype.free;

/**
 * バッチ検索結果
 */
export class IdAdjustmentSearchResults {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(IdAdjustmentSearchResults.prototype);
        obj.__wbg_ptr = ptr;
        IdAdjustmentSearchResultsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IdAdjustmentSearchResultsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_idadjustmentsearchresults_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get length() {
        const ret = wasm.idadjustmentsearchresults_length(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get processedInChunk() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 結果をJavaScript配列として取得
     * @returns {Array<any>}
     */
    to_array() {
        const ret = wasm.idadjustmentsearchresults_to_array(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * 指定インデックスの結果を取得
     * @param {number} index
     * @returns {IdAdjustmentSearchResult | undefined}
     */
    get(index) {
        const ret = wasm.idadjustmentsearchresults_get(this.__wbg_ptr, index);
        return ret === 0 ? undefined : IdAdjustmentSearchResult.__wrap(ret);
    }
}
if (Symbol.dispose) IdAdjustmentSearchResults.prototype[Symbol.dispose] = IdAdjustmentSearchResults.prototype.free;

/**
 * WASM wrapper for IndividualFilter
 */
export class IndividualFilterJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IndividualFilterJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_individualfilterjs_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.individualfilterjs_new();
        this.__wbg_ptr = ret >>> 0;
        IndividualFilterJsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} stat_index
     * @param {number} min
     * @param {number} max
     */
    set_iv_range(stat_index, min, max) {
        wasm.individualfilterjs_set_iv_range(this.__wbg_ptr, stat_index, min, max);
    }
    /**
     * @param {number} nature_index
     */
    set_nature(nature_index) {
        wasm.individualfilterjs_set_nature(this.__wbg_ptr, nature_index);
    }
    /**
     * @param {number} gender
     */
    set_gender(gender) {
        wasm.individualfilterjs_set_gender(this.__wbg_ptr, gender);
    }
    /**
     * @param {number} ability
     */
    set_ability(ability) {
        wasm.individualfilterjs_set_ability(this.__wbg_ptr, ability);
    }
    /**
     * Set shiny filter mode
     * 0 = All (no filter), 1 = Shiny (star OR square), 2 = Star, 3 = Square, 4 = NonShiny
     * @param {number} mode
     */
    set_shiny_filter_mode(mode) {
        wasm.individualfilterjs_set_shiny_filter_mode(this.__wbg_ptr, mode);
    }
    /**
     * @param {number} hp_type
     */
    set_hidden_power_type(hp_type) {
        wasm.individualfilterjs_set_hidden_power_type(this.__wbg_ptr, hp_type);
    }
    /**
     * @param {number} power
     */
    set_hidden_power_power(power) {
        wasm.individualfilterjs_set_hidden_power_power(this.__wbg_ptr, power);
    }
}
if (Symbol.dispose) IndividualFilterJs.prototype[Symbol.dispose] = IndividualFilterJs.prototype.free;

/**
 * MT Seed 起動時間検索イテレータ
 *
 * 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
 * TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
 * 各セグメントに対してこのイテレータを作成する。
 */
export class MtSeedBootTimingSearchIterator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MtSeedBootTimingSearchIteratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_mtseedboottimingsearchiterator_free(ptr, 0);
    }
    /**
     * コンストラクタ
     *
     * # Arguments
     * - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
     * - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
     * - `time_range`: 時刻範囲パラメータ
     * - `search_range`: 検索範囲パラメータ
     * - `target_seeds`: 検索対象のMT Seed値（複数可）
     * @param {DSConfigJs} ds_config
     * @param {SegmentParamsJs} segment
     * @param {TimeRangeParamsJs} time_range
     * @param {SearchRangeParamsJs} search_range
     * @param {Uint32Array} target_seeds
     */
    constructor(ds_config, segment, time_range, search_range, target_seeds) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(ds_config, DSConfigJs);
            _assertClass(segment, SegmentParamsJs);
            _assertClass(time_range, TimeRangeParamsJs);
            _assertClass(search_range, SearchRangeParamsJs);
            const ptr0 = passArray32ToWasm0(target_seeds, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.mtseedboottimingsearchiterator_new(retptr, ds_config.__wbg_ptr, segment.__wbg_ptr, time_range.__wbg_ptr, search_range.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            MtSeedBootTimingSearchIteratorFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * 検索が完了したかどうか
     * @returns {boolean}
     */
    get isFinished() {
        const ret = wasm.mtseedboottimingsearchiterator_is_finished(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * 処理済み秒数
     * @returns {number}
     */
    get processedSeconds() {
        const ret = wasm.idadjustmentsearchiterator_total_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 総秒数
     * @returns {number}
     */
    get totalSeconds() {
        const ret = wasm.mtseedboottimingsearchiterator_total_seconds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 進捗率 (0.0 - 1.0)
     * @returns {number}
     */
    get progress() {
        const ret = wasm.mtseedboottimingsearchiterator_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * 次のバッチを取得
     *
     * - max_results件見つかったら即return
     * - chunk_seconds秒分処理したら結果がなくても一旦return
     * - 検索範囲を全て処理したらfinished=trueになる
     * @param {number} max_results
     * @param {number} chunk_seconds
     * @returns {MtSeedBootTimingSearchResults}
     */
    next_batch(max_results, chunk_seconds) {
        const ret = wasm.mtseedboottimingsearchiterator_next_batch(this.__wbg_ptr, max_results, chunk_seconds);
        return MtSeedBootTimingSearchResults.__wrap(ret);
    }
}
if (Symbol.dispose) MtSeedBootTimingSearchIterator.prototype[Symbol.dispose] = MtSeedBootTimingSearchIterator.prototype.free;

/**
 * MT Seed 起動時間検索結果1件
 */
export class MtSeedBootTimingSearchResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MtSeedBootTimingSearchResult.prototype);
        obj.__wbg_ptr = ptr;
        MtSeedBootTimingSearchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MtSeedBootTimingSearchResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_mtseedboottimingsearchresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get mtSeed() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get mtSeedHex() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.mtseedboottimingsearchresult_mt_seed_hex(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get lcgSeedHigh() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get lcgSeedLow() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get lcgSeedHex() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.mtseedboottimingsearchresult_lcg_seed_hex(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get year() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get month() {
        const ret = wasm.eggboottimingsearchresult_month(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get day() {
        const ret = wasm.idadjustmentsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get hour() {
        const ret = wasm.eggboottimingsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get minute() {
        const ret = wasm.idadjustmentsearchresult_second(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get second() {
        const ret = wasm.eggboottimingsearchresult_second(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get timer0() {
        const ret = wasm.idadjustmentsearchresult_vcount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get vcount() {
        const ret = wasm.eggboottimingsearchresult_vcount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get keyCode() {
        const ret = wasm.mtseedboottimingsearchresult_key_code(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) MtSeedBootTimingSearchResult.prototype[Symbol.dispose] = MtSeedBootTimingSearchResult.prototype.free;

/**
 * バッチ検索結果
 */
export class MtSeedBootTimingSearchResults {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MtSeedBootTimingSearchResults.prototype);
        obj.__wbg_ptr = ptr;
        MtSeedBootTimingSearchResultsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MtSeedBootTimingSearchResultsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_mtseedboottimingsearchresults_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get length() {
        const ret = wasm.mtseedboottimingsearchresults_length(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get processedInChunk() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 結果をJavaScript配列として取得
     * @returns {Array<any>}
     */
    to_array() {
        const ret = wasm.mtseedboottimingsearchresults_to_array(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * 指定インデックスの結果を取得
     * @param {number} index
     * @returns {MtSeedBootTimingSearchResult | undefined}
     */
    get(index) {
        const ret = wasm.mtseedboottimingsearchresults_get(this.__wbg_ptr, index);
        return ret === 0 ? undefined : MtSeedBootTimingSearchResult.__wrap(ret);
    }
}
if (Symbol.dispose) MtSeedBootTimingSearchResults.prototype[Symbol.dispose] = MtSeedBootTimingSearchResults.prototype.free;

/**
 * 数値変換ユーティリティ
 */
export class NumberUtils {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NumberUtilsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_numberutils_free(ptr, 0);
    }
    /**
     * 16進数文字列を32bit整数に変換
     *
     * # Arguments
     * * `hex_str` - 16進数文字列（0xプレフィックス可）
     *
     * # Returns
     * 変換された整数値（エラー時は0）
     * @param {string} hex_str
     * @returns {number}
     */
    static hex_string_to_u32(hex_str) {
        const ptr0 = passStringToWasm0(hex_str, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.numberutils_hex_string_to_u32(ptr0, len0);
        return ret >>> 0;
    }
    /**
     * 32bit整数を16進数文字列に変換
     *
     * # Arguments
     * * `value` - 変換する整数値
     * * `uppercase` - 大文字で出力するか
     *
     * # Returns
     * 16進数文字列
     * @param {number} value
     * @param {boolean} uppercase
     * @returns {string}
     */
    static u32_to_hex_string(value, uppercase) {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.numberutils_u32_to_hex_string(retptr, value, uppercase);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * BCD（Binary Coded Decimal）エンコード
     *
     * # Arguments
     * * `value` - エンコードする値（0-99）
     *
     * # Returns
     * BCDエンコードされた値
     * @param {number} value
     * @returns {number}
     */
    static encode_bcd(value) {
        const ret = wasm.numberutils_encode_bcd(value);
        return ret;
    }
    /**
     * BCD（Binary Coded Decimal）デコード
     *
     * # Arguments
     * * `bcd_value` - デコードするBCD値
     *
     * # Returns
     * デコードされた値
     * @param {number} bcd_value
     * @returns {number}
     */
    static decode_bcd(bcd_value) {
        const ret = wasm.numberutils_decode_bcd(bcd_value);
        return ret;
    }
    /**
     * パーセンテージを乱数閾値に変換
     *
     * # Arguments
     * * `percentage` - パーセンテージ（0.0-100.0）
     *
     * # Returns
     * 32bit乱数閾値
     * @param {number} percentage
     * @returns {number}
     */
    static percentage_to_threshold(percentage) {
        const ret = wasm.numberutils_percentage_to_threshold(percentage);
        return ret >>> 0;
    }
    /**
     * 32bit乱数閾値をパーセンテージに変換
     *
     * # Arguments
     * * `threshold` - 32bit乱数閾値
     *
     * # Returns
     * パーセンテージ
     * @param {number} threshold
     * @returns {number}
     */
    static threshold_to_percentage(threshold) {
        const ret = wasm.numberutils_threshold_to_percentage(threshold);
        return ret;
    }
}
if (Symbol.dispose) NumberUtils.prototype[Symbol.dispose] = NumberUtils.prototype.free;

/**
 * オフセット計算エンジン
 */
export class OffsetCalculator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OffsetCalculatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_offsetcalculator_free(ptr, 0);
    }
    /**
     * 新しいOffsetCalculatorインスタンスを作成
     *
     * # Arguments
     * * `seed` - 初期Seed値
     * @param {bigint} seed
     */
    constructor(seed) {
        const ret = wasm.offsetcalculator_new(seed);
        this.__wbg_ptr = ret >>> 0;
        OffsetCalculatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * 次の32bit乱数値を取得（上位32bit）
     *
     * # Returns
     * 32bit乱数値
     * @returns {number}
     */
    next_rand() {
        const ret = wasm.offsetcalculator_next_rand(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 指定回数だけ乱数を消費（Rand×n）
     *
     * # Arguments
     * * `count` - 消費する回数
     * @param {number} count
     */
    consume_random(count) {
        wasm.offsetcalculator_consume_random(this.__wbg_ptr, count);
    }
    /**
     * 現在の進行回数を取得
     *
     * # Returns
     * 進行回数
     * @returns {number}
     */
    get get_advances() {
        const ret = wasm.eggboottimingsearchresult_year(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 現在のSeed値を取得
     *
     * # Returns
     * 現在のSeed値
     * @returns {bigint}
     */
    get get_current_seed() {
        const ret = wasm.eggboottimingsearchresult_advance(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * 計算器をリセット
     *
     * # Arguments
     * * `new_seed` - 新しいSeed値
     * @param {bigint} new_seed
     */
    reset(new_seed) {
        wasm.offsetcalculator_reset(this.__wbg_ptr, new_seed);
    }
    /**
     * TID/SID決定処理（リファレンス実装準拠）
     *
     * # Returns
     * TidSidResult
     * @returns {TidSidResult}
     */
    calculate_tid_sid() {
        const ret = wasm.offsetcalculator_calculate_tid_sid(this.__wbg_ptr);
        return TidSidResult.__wrap(ret);
    }
    /**
     * 表住人決定処理（BW：固定10回乱数消費）
     */
    determine_front_residents() {
        wasm.offsetcalculator_determine_front_residents(this.__wbg_ptr);
    }
    /**
     * 裏住人決定処理（BW：固定3回乱数消費）
     */
    determine_back_residents() {
        wasm.offsetcalculator_determine_back_residents(this.__wbg_ptr);
    }
    /**
     * 住人決定一括処理（BW専用）
     */
    determine_all_residents() {
        wasm.offsetcalculator_determine_all_residents(this.__wbg_ptr);
    }
    /**
     * Probability Table処理（仕様書準拠の6段階テーブル処理）
     */
    probability_table_process() {
        wasm.offsetcalculator_probability_table_process(this.__wbg_ptr);
    }
    /**
     * PT操作×n回
     * @param {number} count
     */
    probability_table_process_multiple(count) {
        wasm.offsetcalculator_probability_table_process_multiple(this.__wbg_ptr, count);
    }
    /**
     * Extra処理（BW2専用：重複値回避ループ）
     * 3つの値（0-14範囲）がすべて異なるまでループ
     * @returns {ExtraResult}
     */
    extra_process() {
        const ret = wasm.offsetcalculator_extra_process(this.__wbg_ptr);
        return ExtraResult.__wrap(ret);
    }
    /**
     * ゲーム初期化処理の総合実行（仕様書準拠）
     *
     * # Arguments
     * * `mode` - ゲームモード
     *
     * # Returns
     * 初期化完了時の進行回数
     * @param {GameMode} mode
     * @returns {number}
     */
    execute_game_initialization(mode) {
        const ret = wasm.offsetcalculator_execute_game_initialization(this.__wbg_ptr, mode);
        return ret >>> 0;
    }
}
if (Symbol.dispose) OffsetCalculator.prototype[Symbol.dispose] = OffsetCalculator.prototype.free;

/**
 * PID計算エンジン
 */
export class PIDCalculator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PIDCalculatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pidcalculator_free(ptr, 0);
    }
    /**
     * 新しいPIDCalculatorインスタンスを作成
     */
    constructor() {
        const ret = wasm.encountercalculator_new();
        this.__wbg_ptr = ret >>> 0;
        PIDCalculatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * BW/BW2準拠 統一PID生成
     * 32bit乱数 ^ 0x10000 の計算（固定・野生共通）
     *
     * # Arguments
     * * `r1` - 乱数値1
     *
     * # Returns
     * 基本PID（ID補正前）
     * @param {number} r1
     * @returns {number}
     */
    static generate_base_pid(r1) {
        const ret = wasm.pidcalculator_generate_base_pid(r1);
        return ret >>> 0;
    }
    /**
     * ID補正処理
     * 性格値下位 ^ トレーナーID ^ 裏ID の奇偶性で最上位bitを調整
     *
     * # Arguments
     * * `pid` - 基本PID
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     *
     * # Returns
     * ID補正後PID
     * @param {number} pid
     * @param {number} tid
     * @param {number} sid
     * @returns {number}
     */
    static apply_id_correction(pid, tid, sid) {
        const ret = wasm.pidcalculator_apply_id_correction(pid, tid, sid);
        return ret >>> 0;
    }
    /**
     * BW/BW2準拠 野生ポケモンのPID生成
     * 32bit乱数 ^ 0x10000 + ID補正処理
     *
     * # Arguments
     * * `r1` - 乱数値1
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     *
     * # Returns
     * 生成されたPID（ID補正適用後）
     * @param {number} r1
     * @param {number} tid
     * @param {number} sid
     * @returns {number}
     */
    static generate_wild_pid(r1, tid, sid) {
        const ret = wasm.pidcalculator_generate_roamer_pid(r1, tid, sid);
        return ret >>> 0;
    }
    /**
     * BW/BW2準拠 固定シンボルポケモンのPID生成
     * 32bit乱数 ^ 0x10000 + ID補正処理
     *
     * # Arguments
     * * `r1` - 乱数値1
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     *
     * # Returns
     * 生成されたPID（ID補正適用後）
     * @param {number} r1
     * @param {number} tid
     * @param {number} sid
     * @returns {number}
     */
    static generate_static_pid(r1, tid, sid) {
        const ret = wasm.pidcalculator_generate_roamer_pid(r1, tid, sid);
        return ret >>> 0;
    }
    /**
     * BW/BW2準拠 徘徊ポケモンのPID生成
     * 32bit乱数 ^ 0x10000 + ID補正処理
     *
     * # Arguments
     * * `r1` - 乱数値1
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     *
     * # Returns
     * 生成されたPID（ID補正適用後）
     * @param {number} r1
     * @param {number} tid
     * @param {number} sid
     * @returns {number}
     */
    static generate_roamer_pid(r1, tid, sid) {
        const ret = wasm.pidcalculator_generate_roamer_pid(r1, tid, sid);
        return ret >>> 0;
    }
    /**
     * BW/BW2準拠 イベントポケモンのPID生成
     * 32bit乱数 ^ 0x10000（ID補正なし - 先頭特性無効）
     *
     * # Arguments
     * * `r1` - 乱数値1
     *
     * # Returns
     * 生成されたPID（ID補正なし）
     * @param {number} r1
     * @returns {number}
     */
    static generate_event_pid(r1) {
        const ret = wasm.pidcalculator_generate_base_pid(r1);
        return ret >>> 0;
    }
    /**
     * ギフトポケモンのPID生成
     * 特殊な計算式を使用
     *
     * # Arguments
     * * `r1` - 乱数値1
     * * `r2` - 乱数値2
     *
     * # Returns
     * 生成されたPID
     * @param {number} r1
     * @param {number} r2
     * @returns {number}
     */
    static generate_gift_pid(r1, r2) {
        const ret = wasm.pidcalculator_generate_gift_pid(r1, r2);
        return ret >>> 0;
    }
    /**
     * タマゴのPID生成
     * 特殊な計算式を使用
     *
     * # Arguments
     * * `r1` - 乱数値1
     * * `r2` - 乱数値2
     *
     * # Returns
     * 生成されたPID
     * @param {number} r1
     * @param {number} r2
     * @returns {number}
     */
    static generate_egg_pid(r1, r2) {
        const ret = wasm.pidcalculator_generate_egg_pid(r1, r2);
        return ret >>> 0;
    }
}
if (Symbol.dispose) PIDCalculator.prototype[Symbol.dispose] = PIDCalculator.prototype.free;

/**
 * WASM wrapper for ParentsIVs
 */
export class ParentsIVsJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ParentsIVsJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_parentsivsjs_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.parentsivsjs_new();
        this.__wbg_ptr = ret >>> 0;
        ParentsIVsJsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Uint8Array} ivs
     */
    set male(ivs) {
        const ptr0 = passArray8ToWasm0(ivs, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.parentsivsjs_set_male(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Uint8Array} ivs
     */
    set female(ivs) {
        const ptr0 = passArray8ToWasm0(ivs, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.parentsivsjs_set_female(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ParentsIVsJs.prototype[Symbol.dispose] = ParentsIVsJs.prototype.free;

/**
 * PersonalityRNG構造体
 * BW仕様64bit線形合同法: S[n+1] = S[n] * 0x5D588B656C078965 + 0x269EC3
 */
export class PersonalityRNG {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PersonalityRNGFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_personalityrng_free(ptr, 0);
    }
    /**
     * 新しいPersonalityRNGインスタンスを作成
     *
     * # Arguments
     * * `seed` - 初期Seed値（64bit）
     * @param {bigint} seed
     */
    constructor(seed) {
        const ret = wasm.personalityrng_new(seed);
        this.__wbg_ptr = ret >>> 0;
        PersonalityRNGFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * 次の32bit乱数値を取得（上位32bit）
     *
     * # Returns
     * 上位32bitの乱数値
     * @returns {number}
     */
    next() {
        const ret = wasm.personalityrng_next(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 次の64bit乱数値を取得
     *
     * # Returns
     * 64bit乱数値（内部状態そのもの）
     * @returns {bigint}
     */
    next_u64() {
        const ret = wasm.personalityrng_next_u64(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * 現在のSeed値を取得
     *
     * # Returns
     * 現在の内部Seed値
     * @returns {bigint}
     */
    get current_seed() {
        const ret = wasm.eggboottimingsearchresult_advance(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Seed値を設定
     *
     * # Arguments
     * * `new_seed` - 新しいSeed値
     * @param {bigint} new_seed
     */
    set seed(new_seed) {
        wasm.personalityrng_reset(this.__wbg_ptr, new_seed);
    }
    /**
     * 指定回数だけ乱数を進める
     *
     * # Arguments
     * * `advances` - 進める回数
     * @param {number} advances
     */
    advance(advances) {
        wasm.personalityrng_advance(this.__wbg_ptr, advances);
    }
    /**
     * Seedをリセット
     *
     * # Arguments
     * * `initial_seed` - リセット後のSeed値
     * @param {bigint} initial_seed
     */
    reset(initial_seed) {
        wasm.personalityrng_reset(this.__wbg_ptr, initial_seed);
    }
    /**
     * 0x0からの進行度を計算
     *
     * # Arguments
     * * `seed` - 計算対象のSeed値
     *
     * # Returns
     * 0x0からの進行度
     * @param {bigint} seed
     * @returns {bigint}
     */
    static get_index(seed) {
        const ret = wasm.personalityrng_get_index(seed);
        return BigInt.asUintN(64, ret);
    }
    /**
     * 2つのSeed間の距離を計算
     *
     * # Arguments
     * * `from_seed` - 開始Seed
     * * `to_seed` - 終了Seed
     *
     * # Returns
     * from_seedからto_seedまでの距離
     * @param {bigint} from_seed
     * @param {bigint} to_seed
     * @returns {bigint}
     */
    static distance_between(from_seed, to_seed) {
        const ret = wasm.personalityrng_distance_between(from_seed, to_seed);
        return BigInt.asUintN(64, ret);
    }
    /**
     * 指定Seedから現在のSeedまでの距離
     *
     * # Arguments
     * * `source_seed` - 開始Seed
     *
     * # Returns
     * source_seedから現在のSeedまでの距離
     * @param {bigint} source_seed
     * @returns {bigint}
     */
    distance_from(source_seed) {
        const ret = wasm.personalityrng_distance_from(this.__wbg_ptr, source_seed);
        return BigInt.asUintN(64, ret);
    }
}
if (Symbol.dispose) PersonalityRNG.prototype[Symbol.dispose] = PersonalityRNG.prototype.free;

/**
 * ポケモン生成エンジン
 */
export class PokemonGenerator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PokemonGeneratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pokemongenerator_free(ptr, 0);
    }
    /**
     * 新しいPokemonGeneratorインスタンスを作成
     */
    constructor() {
        const ret = wasm.encountercalculator_new();
        this.__wbg_ptr = ret >>> 0;
        PokemonGeneratorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * BW/BW2準拠 単体ポケモン生成（統括関数）
     *
     * # Arguments
     * * `seed` - 初期Seed値
     * * `config` - BW準拠設定
     *
     * # Returns
     * 生成されたポケモンデータ
     * @param {bigint} seed
     * @param {BWGenerationConfig} config
     * @returns {RawPokemonData}
     */
    static generate_single_pokemon_bw(seed, config) {
        _assertClass(config, BWGenerationConfig);
        const ret = wasm.pokemongenerator_generate_single_pokemon_bw(seed, config.__wbg_ptr);
        return RawPokemonData.__wrap(ret);
    }
    /**
     * オフセット適用後の生成開始Seedを計算
     * @param {bigint} initial_seed
     * @param {bigint} offset
     * @returns {bigint}
     */
    static calculate_generation_seed(initial_seed, offset) {
        const ret = wasm.pokemongenerator_calculate_generation_seed(initial_seed, offset);
        return BigInt.asUintN(64, ret);
    }
    /**
     * BW/BW2準拠 バッチ生成（offsetのみ）
     *
     * # Arguments
     * * `base_seed` - 列挙の初期Seed
     * * `offset` - 最初の生成までの前進数（ゲーム内不定消費を含めた開始位置）
     * * `count` - 生成数（0なら空）
     * * `config` - BW準拠設定
     *
     * # Returns
     * 生成されたポケモンデータの配列
     * @param {bigint} base_seed
     * @param {bigint} offset
     * @param {number} count
     * @param {BWGenerationConfig} config
     * @returns {RawPokemonData[]}
     */
    static generate_pokemon_batch_bw(base_seed, offset, count, config) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(config, BWGenerationConfig);
            wasm.pokemongenerator_generate_pokemon_batch_bw(retptr, base_seed, offset, count, config.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) PokemonGenerator.prototype[Symbol.dispose] = PokemonGenerator.prototype.free;

/**
 * 生ポケモンデータ構造体
 */
export class RawPokemonData {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(RawPokemonData.prototype);
        obj.__wbg_ptr = ptr;
        RawPokemonDataFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RawPokemonDataFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_rawpokemondata_free(ptr, 0);
    }
    /**
     * getter methods for JavaScript access
     * @returns {bigint}
     */
    get get_seed() {
        const ret = wasm.eggboottimingsearchresult_advance(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get get_pid() {
        const ret = wasm.eggboottimingsearchresult_year(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get get_nature() {
        const ret = wasm.rawpokemondata_get_nature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_ability_slot() {
        const ret = wasm.rawpokemondata_get_ability_slot(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_gender_value() {
        const ret = wasm.rawpokemondata_get_gender_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_encounter_slot_value() {
        const ret = wasm.rawpokemondata_get_encounter_slot_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get get_level_rand_value() {
        const ret = wasm.rawpokemondata_get_level_rand_value(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {number}
     */
    get get_shiny_type() {
        const ret = wasm.rawpokemondata_get_shiny_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get get_sync_applied() {
        const ret = wasm.rawpokemondata_get_sync_applied(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get get_encounter_type() {
        const ret = wasm.rawpokemondata_get_encounter_type(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) RawPokemonData.prototype[Symbol.dispose] = RawPokemonData.prototype.free;

/**
 * 検索範囲パラメータ（公開型）
 */
export class SearchRangeParamsJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SearchRangeParamsJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_searchrangeparamsjs_free(ptr, 0);
    }
    /**
     * @param {number} start_year
     * @param {number} start_month
     * @param {number} start_day
     * @param {number} start_second_offset
     * @param {number} range_seconds
     */
    constructor(start_year, start_month, start_day, start_second_offset, range_seconds) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.searchrangeparamsjs_new(retptr, start_year, start_month, start_day, start_second_offset, range_seconds);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            SearchRangeParamsJsFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    get start_year() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_month() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_day() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get start_second_offset() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get range_seconds() {
        const ret = wasm.eggboottimingsearchresult_month(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) SearchRangeParamsJs.prototype[Symbol.dispose] = SearchRangeParamsJs.prototype.free;

/**
 * 連続列挙用のSeed列挙器（offsetのみ）
 */
export class SeedEnumerator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SeedEnumeratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_seedenumerator_free(ptr, 0);
    }
    /**
     * 列挙器を作成
     * @param {bigint} base_seed
     * @param {bigint} user_offset
     * @param {number} count
     * @param {BWGenerationConfig} config
     * @param {GameMode} game_mode
     */
    constructor(base_seed, user_offset, count, config, game_mode) {
        _assertClass(config, BWGenerationConfig);
        const ret = wasm.seedenumerator_new(base_seed, user_offset, count, config.__wbg_ptr, game_mode);
        this.__wbg_ptr = ret >>> 0;
        SeedEnumeratorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * 次のポケモンを生成（残数0なら undefined を返す）
     * @returns {EnumeratedPokemonData | undefined}
     */
    next_pokemon() {
        const ret = wasm.seedenumerator_next_pokemon(this.__wbg_ptr);
        return ret === 0 ? undefined : EnumeratedPokemonData.__wrap(ret);
    }
    /**
     * 残数を取得
     * @returns {number}
     */
    get remaining() {
        const ret = wasm.eggboottimingsearchresult_date(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) SeedEnumerator.prototype[Symbol.dispose] = SeedEnumerator.prototype.free;

/**
 * セグメントパラメータ（公開型）
 */
export class SegmentParamsJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SegmentParamsJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_segmentparamsjs_free(ptr, 0);
    }
    /**
     * @param {number} timer0
     * @param {number} vcount
     * @param {number} key_code
     */
    constructor(timer0, vcount, key_code) {
        const ret = wasm.segmentparamsjs_new(timer0, vcount, key_code);
        this.__wbg_ptr = ret >>> 0;
        SegmentParamsJsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get timer0() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get vcount() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get key_code() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) SegmentParamsJs.prototype[Symbol.dispose] = SegmentParamsJs.prototype.free;

/**
 * 色違い判定エンジン
 */
export class ShinyChecker {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ShinyCheckerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_shinychecker_free(ptr, 0);
    }
    /**
     * 新しいShinyCheckerインスタンスを作成
     */
    constructor() {
        const ret = wasm.encountercalculator_new();
        this.__wbg_ptr = ret >>> 0;
        ShinyCheckerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * 色違い判定
     *
     * # Arguments
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     * * `pid` - ポケモンのPID
     *
     * # Returns
     * 色違いかどうか
     * @param {number} tid
     * @param {number} sid
     * @param {number} pid
     * @returns {boolean}
     */
    static is_shiny(tid, sid, pid) {
        const ret = wasm.shinychecker_is_shiny(tid, sid, pid);
        return ret !== 0;
    }
    /**
     * 色違い値の計算
     * TID ^ SID ^ PID上位16bit ^ PID下位16bit
     *
     * # Arguments
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     * * `pid` - ポケモンのPID
     *
     * # Returns
     * 色違い値
     * @param {number} tid
     * @param {number} sid
     * @param {number} pid
     * @returns {number}
     */
    static get_shiny_value(tid, sid, pid) {
        const ret = wasm.shinychecker_get_shiny_value(tid, sid, pid);
        return ret;
    }
    /**
     * 色違いタイプの判定
     *
     * # Arguments
     * * `shiny_value` - 色違い値
     *
     * # Returns
     * 色違いタイプ
     * @param {number} shiny_value
     * @returns {ShinyType}
     */
    static get_shiny_type(shiny_value) {
        const ret = wasm.shinychecker_get_shiny_type(shiny_value);
        return ret;
    }
    /**
     * 色違い判定とタイプを同時に取得
     *
     * # Arguments
     * * `tid` - トレーナーID
     * * `sid` - シークレットID
     * * `pid` - ポケモンのPID
     *
     * # Returns
     * 色違いタイプ
     * @param {number} tid
     * @param {number} sid
     * @param {number} pid
     * @returns {ShinyType}
     */
    static check_shiny_type(tid, sid, pid) {
        const ret = wasm.shinychecker_check_shiny_type(tid, sid, pid);
        return ret;
    }
    /**
     * 色違い確率の計算
     * 通常の色違い確率を計算（参考用）
     *
     * # Returns
     * 色違い確率（分母）
     * @returns {number}
     */
    static shiny_probability() {
        const ret = wasm.shinychecker_shiny_probability();
        return ret >>> 0;
    }
    /**
     * 光るお守り効果の確率計算
     *
     * # Arguments
     * * `has_shiny_charm` - 光るお守りを持っているか
     *
     * # Returns
     * 色違い確率（分母）
     * @param {boolean} has_shiny_charm
     * @returns {number}
     */
    static shiny_probability_with_charm(has_shiny_charm) {
        const ret = wasm.shinychecker_shiny_probability_with_charm(has_shiny_charm);
        return ret >>> 0;
    }
}
if (Symbol.dispose) ShinyChecker.prototype[Symbol.dispose] = ShinyChecker.prototype.free;

/**
 * 色違いタイプ列挙型
 * @enum {0 | 1 | 2}
 */
export const ShinyType = Object.freeze({
    /**
     * 通常（色違いでない）
     */
    Normal: 0, "0": "Normal",
    /**
     * 四角い色違い（一般的な色違い）
     */
    Square: 1, "1": "Square",
    /**
     * 星形色違い（特殊な色違い）
     */
    Star: 2, "2": "Star",
});

export class StatRange {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StatRangeFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_statrange_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get min() {
        const ret = wasm.__wbg_get_statrange_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set min(arg0) {
        wasm.__wbg_set_statrange_min(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get max() {
        const ret = wasm.__wbg_get_genderratio_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set max(arg0) {
        wasm.__wbg_set_genderratio_threshold(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    constructor(min, max) {
        const ret = wasm.statrange_new(min, max);
        this.__wbg_ptr = ret >>> 0;
        StatRangeFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} value
     * @returns {boolean}
     */
    contains(value) {
        const ret = wasm.statrange_contains(this.__wbg_ptr, value);
        return ret !== 0;
    }
}
if (Symbol.dispose) StatRange.prototype[Symbol.dispose] = StatRange.prototype.free;

/**
 * TID/SID決定結果
 */
export class TidSidResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(TidSidResult.prototype);
        obj.__wbg_ptr = ptr;
        TidSidResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TidSidResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tidsidresult_free(ptr, 0);
    }
    /**
     * TID（トレーナーID下位16bit）
     * @returns {number}
     */
    get tid() {
        const ret = wasm.__wbg_get_tidsidresult_tid(this.__wbg_ptr);
        return ret;
    }
    /**
     * TID（トレーナーID下位16bit）
     * @param {number} arg0
     */
    set tid(arg0) {
        wasm.__wbg_set_tidsidresult_tid(this.__wbg_ptr, arg0);
    }
    /**
     * SID（シークレットID上位16bit）
     * @returns {number}
     */
    get sid() {
        const ret = wasm.__wbg_get_tidsidresult_sid(this.__wbg_ptr);
        return ret;
    }
    /**
     * SID（シークレットID上位16bit）
     * @param {number} arg0
     */
    set sid(arg0) {
        wasm.__wbg_set_tidsidresult_sid(this.__wbg_ptr, arg0);
    }
    /**
     * 消費した乱数回数
     * @returns {number}
     */
    get advances_used() {
        const ret = wasm.__wbg_get_extraresult_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * 消費した乱数回数
     * @param {number} arg0
     */
    set advances_used(arg0) {
        wasm.__wbg_set_extraresult_advances(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get get_tid() {
        const ret = wasm.tidsidresult_get_tid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_sid() {
        const ret = wasm.tidsidresult_get_sid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get get_advances_used() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) TidSidResult.prototype[Symbol.dispose] = TidSidResult.prototype.free;

/**
 * 時刻範囲パラメータ（公開型）
 */
export class TimeRangeParamsJs {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TimeRangeParamsJsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_timerangeparamsjs_free(ptr, 0);
    }
    /**
     * @param {number} hour_start
     * @param {number} hour_end
     * @param {number} minute_start
     * @param {number} minute_end
     * @param {number} second_start
     * @param {number} second_end
     */
    constructor(hour_start, hour_end, minute_start, minute_end, second_start, second_end) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.timerangeparamsjs_new(retptr, hour_start, hour_end, minute_start, minute_end, second_start, second_end);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            TimeRangeParamsJsFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    get hour_start() {
        const ret = wasm.extraresult_get_advances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get hour_end() {
        const ret = wasm.extraresult_get_value1(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get minute_start() {
        const ret = wasm.extraresult_get_value2(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get minute_end() {
        const ret = wasm.extraresult_get_value3(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get second_start() {
        const ret = wasm.eggboottimingsearchresult_month(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get second_end() {
        const ret = wasm.idadjustmentsearchresult_hour(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) TimeRangeParamsJs.prototype[Symbol.dispose] = TimeRangeParamsJs.prototype.free;

export class TrainerIds {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TrainerIdsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trainerids_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get tid() {
        const ret = wasm.__wbg_get_trainerids_tid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set tid(arg0) {
        wasm.__wbg_set_trainerids_tid(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get sid() {
        const ret = wasm.__wbg_get_trainerids_sid(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set sid(arg0) {
        wasm.__wbg_set_trainerids_sid(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get tsv() {
        const ret = wasm.__wbg_get_trainerids_tsv(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set tsv(arg0) {
        wasm.__wbg_set_trainerids_tsv(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} tid
     * @param {number} sid
     */
    constructor(tid, sid) {
        const ret = wasm.trainerids_new(tid, sid);
        this.__wbg_ptr = ret >>> 0;
        TrainerIdsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) TrainerIds.prototype[Symbol.dispose] = TrainerIds.prototype.free;

/**
 * バリデーションユーティリティ
 */
export class ValidationUtils {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ValidationUtilsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_validationutils_free(ptr, 0);
    }
    /**
     * TIDの妥当性チェック
     *
     * # Arguments
     * * `tid` - トレーナーID
     *
     * # Returns
     * 妥当性
     * @param {number} _tid
     * @returns {boolean}
     */
    static is_valid_tid(_tid) {
        const ret = wasm.validationutils_is_valid_sid(_tid);
        return ret !== 0;
    }
    /**
     * SIDの妥当性チェック
     *
     * # Arguments
     * * `sid` - シークレットID
     *
     * # Returns
     * 妥当性
     * @param {number} _sid
     * @returns {boolean}
     */
    static is_valid_sid(_sid) {
        const ret = wasm.validationutils_is_valid_sid(_sid);
        return ret !== 0;
    }
    /**
     * 性格値の妥当性チェック
     *
     * # Arguments
     * * `nature` - 性格値
     *
     * # Returns
     * 妥当性
     * @param {number} nature
     * @returns {boolean}
     */
    static is_valid_nature(nature) {
        const ret = wasm.validationutils_is_valid_nature(nature);
        return ret !== 0;
    }
    /**
     * 特性スロットの妥当性チェック
     *
     * # Arguments
     * * `ability_slot` - 特性スロット
     *
     * # Returns
     * 妥当性
     * @param {number} ability_slot
     * @returns {boolean}
     */
    static is_valid_ability_slot(ability_slot) {
        const ret = wasm.validationutils_is_valid_ability_slot(ability_slot);
        return ret !== 0;
    }
    /**
     * 16進数文字列の妥当性チェック
     *
     * # Arguments
     * * `hex_str` - 16進数文字列
     *
     * # Returns
     * 妥当性
     * @param {string} hex_str
     * @returns {boolean}
     */
    static is_valid_hex_string(hex_str) {
        const ptr0 = passStringToWasm0(hex_str, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.validationutils_is_valid_hex_string(ptr0, len0);
        return ret !== 0;
    }
    /**
     * Seed値の妥当性チェック
     *
     * # Arguments
     * * `seed` - Seed値
     *
     * # Returns
     * 妥当性
     * @param {bigint} seed
     * @returns {boolean}
     */
    static is_valid_seed(seed) {
        const ret = wasm.validationutils_is_valid_seed(seed);
        return ret !== 0;
    }
}
if (Symbol.dispose) ValidationUtils.prototype[Symbol.dispose] = ValidationUtils.prototype.free;

/**
 * オフセット計算統合API（仕様書準拠）
 * @param {bigint} initial_seed
 * @param {GameMode} mode
 * @returns {number}
 */
export function calculate_game_offset(initial_seed, mode) {
    const ret = wasm.calculate_game_offset(initial_seed, mode);
    return ret >>> 0;
}

/**
 * TID/SID決定処理統合API（仕様書準拠）
 * @param {bigint} initial_seed
 * @param {GameMode} mode
 * @returns {TidSidResult}
 */
export function calculate_tid_sid_from_seed(initial_seed, mode) {
    const ret = wasm.calculate_tid_sid_from_seed(initial_seed, mode);
    return TidSidResult.__wrap(ret);
}

/**
 * WASM公開関数: IVコードデコード
 *
 * IVコードをIVセットにデコードする
 *
 * # Arguments
 * * `code` - IVコード (30bit)
 *
 * # Returns
 * IVセット [HP, Atk, Def, SpA, SpD, Spe]
 * @param {number} code
 * @returns {Uint8Array}
 */
export function decode_iv_code_wasm(code) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.decode_iv_code_wasm(retptr, code);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * WASM公開関数: IVセット導出
 *
 * MT SeedとMT消費数からIVセットを導出する
 *
 * # Arguments
 * * `mt_seed` - MT Seed
 * * `advances` - MT消費数
 *
 * # Returns
 * IVセット [HP, Atk, Def, SpA, SpD, Spe]
 * @param {number} mt_seed
 * @param {number} advances
 * @returns {Uint8Array}
 */
export function derive_iv_set_wasm(mt_seed, advances) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.derive_iv_set_wasm(retptr, mt_seed, advances);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * WASM公開関数: IVコードエンコード
 *
 * IVセットをIVコードにエンコードする
 *
 * # Arguments
 * * `ivs` - IVセット [HP, Atk, Def, SpA, SpD, Spe]
 *
 * # Returns
 * IVコード (30bit)
 * @param {Uint8Array} ivs
 * @returns {number}
 */
export function encode_iv_code_wasm(ivs) {
    const ptr0 = passArray8ToWasm0(ivs, wasm.__wbindgen_export3);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.encode_iv_code_wasm(ptr0, len0);
    return ret >>> 0;
}

/**
 * WASM公開関数: MT Seed検索セグメント実行
 *
 * SIMD最適化版を使用して検索を実行
 *
 * # Arguments
 * * `start` - 検索開始Seed (inclusive)
 * * `end` - 検索終了Seed (inclusive)
 * * `advances` - MT消費数
 * * `target_codes` - 検索対象IVコードのスライス
 *
 * # Returns
 * フラット配列 [seed0, code0, seed1, code1, ...]
 * @param {number} start
 * @param {number} end
 * @param {number} advances
 * @param {Uint32Array} target_codes
 * @returns {Uint32Array}
 */
export function mt_seed_search_segment(start, end, advances, target_codes) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray32ToWasm0(target_codes, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.mt_seed_search_segment(retptr, start, end, advances, ptr0, len0);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v2 = getArrayU32FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 4, 4);
        return v2;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * WebAssembly向けバッチSHA-1計算エントリポイント
 * `messages` は 16 ワード単位（512bit）で並ぶフラットな配列である必要がある
 * @param {Uint32Array} messages
 * @returns {Uint32Array}
 */
export function sha1_hash_batch(messages) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray32ToWasm0(messages, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.sha1_hash_batch(retptr, ptr0, len0);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v2 = getArrayU32FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 4, 4);
        return v2;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_eggboottimingsearchresult_new = function(arg0) {
        const ret = EggBootTimingSearchResult.__wrap(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_idadjustmentsearchresult_new = function(arg0) {
        const ret = IdAdjustmentSearchResult.__wrap(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_mtseedboottimingsearchresult_new = function(arg0) {
        const ret = MtSeedBootTimingSearchResult.__wrap(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_1ba21ce319a06297 = function() {
        const ret = new Object();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_25f239778d6112b9 = function() {
        const ret = new Array();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_with_length_12c6de4fac33117a = function(arg0) {
        const ret = new Array(arg0 >>> 0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_push_7d9be8f38fc13975 = function(arg0, arg1) {
        const ret = getObject(arg0).push(getObject(arg1));
        return ret;
    };
    imports.wbg.__wbg_rawpokemondata_new = function(arg0) {
        const ret = RawPokemonData.__wrap(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_set_781438a03c0c3c81 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = Reflect.set(getObject(arg0), getObject(arg1), getObject(arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_set_7df433eea03a5c14 = function(arg0, arg1, arg2) {
        getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('wasm_pkg_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
