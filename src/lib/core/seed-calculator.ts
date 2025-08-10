import { SHA1 } from './sha1';
import { initWasm, getWasm, isWasmReady } from './wasm-interface';
import type { SearchConditions } from '../../types/search';
import type { ROMParameters, Hardware } from '../../types/rom';
import romParameters from '../../data/rom-parameters';

const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
  DS: 8,
  DS_LITE: 6,
  '3DS': 9
};

/**
 * Utility functions for Pokemon BW/BW2 initial seed calculation
 * Supports both TypeScript and WebAssembly implementations
 */

export class SeedCalculator {
  private sha1: SHA1;
  private useWasm: boolean = false;

  constructor() {
    this.sha1 = new SHA1();
  }

  /**
   * Initialize WebAssembly support (optional)
   * If successful, calculations will use WebAssembly for better performance
   */
  public async initializeWasm(): Promise<boolean> {
    try {
      await initWasm();
      this.useWasm = true;
      return true;
    } catch (error) {
      console.warn('WebAssembly initialization failed, falling back to TypeScript:', error);
      this.useWasm = false;
      return false;
    }
  }

  /**
   * Check if WebAssembly is being used
   */
  public isUsingWasm(): boolean {
    return this.useWasm && isWasmReady();
  }

  /**
   * Get WebAssembly module for integrated search
   */
  public getWasmModule() {
    // 明示型を返すことで any を回避（呼び出し側はインポート経由の型推論に依存）
    return getWasm();
  }

  /**
   * Force enable/disable WebAssembly usage
   */
  public setUseWasm(use: boolean): void {
    if (use && !isWasmReady()) {
      console.warn('Cannot enable WebAssembly: module not initialized');
      return;
    }
    this.useWasm = use;
  }

  /**
   * Get ROM parameters for the specified version and region
   */
  public getROMParameters(version: string, region: string): ROMParameters | null {
    const versionData = romParameters[version as keyof typeof romParameters];
    if (!versionData) {
      console.error(`ROM version not found: ${version}`);
      return null;
    }
    
    const regionData = versionData[region as keyof typeof versionData];
    if (!regionData) {
      console.error(`ROM region not found: ${region} for version ${version}`);
      return null;
    }
    
    return {
      nazo: [...regionData.nazo] as [number, number, number, number, number],
      vcountTimerRanges: regionData.vcountTimerRanges.map(range => 
        [...range] as [number, number, number]
      )
    };
  }

  /**
   * Convert little-endian 32-bit integer
   */
  private toLittleEndian32(value: number): number {
    return (((value & 0xFF) << 24) | 
           (((value >>> 8) & 0xFF) << 16) | 
           (((value >>> 16) & 0xFF) << 8) | 
           ((value >>> 24) & 0xFF)) >>> 0;
  }

  /**
   * Convert little-endian 16-bit integer
   */
  private toLittleEndian16(value: number): number {
    return ((value & 0xFF) << 8) | ((value >>> 8) & 0xFF);
  }

  /**
   * Calculate day of week (0=Sunday, 1=Monday, etc.)
   */
  private getDayOfWeek(year: number, month: number, day: number): number {
    const date = new Date(year, month - 1, day);
    return date.getDay();
  }

  /**
   * Generate message array for SHA-1 calculation
   */
  public generateMessage(conditions: SearchConditions, timer0: number, vcount: number, datetime: Date): number[] {
    const params = this.getROMParameters(conditions.romVersion, conditions.romRegion);
    if (!params) {
      throw new Error(`No parameters found for ${conditions.romVersion} ${conditions.romRegion}`);
    }

    const message = new Array(16).fill(0);
    
    // data[0-4]: Nazo values (little-endian conversion needed)
    for (let i = 0; i < 5; i++) {
      message[i] = this.toLittleEndian32(params.nazo[i]);
    }
    
    // data[5]: (VCount << 16) | (little-endian conversion needed)
    message[5] = this.toLittleEndian32((vcount << 16) | timer0);
    
    // data[6]: MAC address lower 16 bits (no endian conversion)
    const macLower = (conditions.macAddress[4] << 8) | conditions.macAddress[5];
    message[6] = macLower;
    
    // data[7]: MAC address upper 32 bits XOR GxStat XOR Frame (little-endian conversion needed)
    const macUpper = (conditions.macAddress[0] << 0) | (conditions.macAddress[1] << 8) | (conditions.macAddress[2] << 16) | (conditions.macAddress[3] << 24);
    const gxStat = 0x06000000;
    const frame = HARDWARE_FRAME_VALUES[conditions.hardware];
    const data7 = macUpper ^ gxStat ^ frame;
    message[7] = this.toLittleEndian32(data7);
    
    // data[8]: Date and day of week (YYMMDDWW format, 10進数→16進数変換, no endian conversion)
    const year = datetime.getFullYear() % 100;
    const month = datetime.getMonth() + 1;
    const day = datetime.getDate();
    const dayOfWeek = this.getDayOfWeek(datetime.getFullYear(), month, day);
    
    // Build BCD-like decimal representation then treat as hex
    // Example: 2023/12/31 Sunday → 23123100 (decimal) → 0x160D05C4 (hex)
    const yyBCD = Math.floor(year / 10) * 16 + (year % 10);
    const mmBCD = Math.floor(month / 10) * 16 + (month % 10);
    const ddBCD = Math.floor(day / 10) * 16 + (day % 10);
    const wwBCD = Math.floor(dayOfWeek / 10) * 16 + (dayOfWeek % 10);
    message[8] = (yyBCD << 24) | (mmBCD << 16) | (ddBCD << 8) | wwBCD;
    
    // data[9]: Time (HHMMSS00 format, DS/DS Lite adds 0x40 for PM, 10進数→16進数変換, no endian conversion)
    const hour = datetime.getHours();
    const minute = datetime.getMinutes();
    const second = datetime.getSeconds();
    

    // DS/DS Lite hardware PM flag adjustment
    const pmFlag = (conditions.hardware === 'DS' || conditions.hardware === 'DS_LITE') && hour >= 12 ? 0x1 : 0x0;
    
    // Build BCD-like decimal representation then treat as hex
    // Example: 23:59:59 → 63595900 (DS/DS Lite PM) → 0x3C98BC04 (hex)
    const hhBCD = Math.floor(hour / 10) * 16 + (hour % 10);
    const minBCD = Math.floor(minute / 10) * 16 + (minute % 10);
    const secBCD = Math.floor(second / 10) * 16 + (second % 10);
    message[9] = (pmFlag << 30) | (hhBCD << 24) | (minBCD << 16) | (secBCD << 8) | 0x00;
    
    // data[10-11]: Fixed values 0x00000000
    message[10] = 0x00000000;
    message[11] = 0x00000000;
    
    // data[12]: Key input (little-endian conversion needed)
    message[12] = this.toLittleEndian32(conditions.keyInput);
    
    // data[13-15]: SHA-1 padding
    message[13] = 0x80000000;
    message[14] = 0x00000000;
    message[15] = 0x000001A0;
    
    return message;
  }

  /**
   * Calculate initial seed from message
   * Uses TypeScript SHA-1 implementation
   */
  public calculateSeed(message: number[]): { seed: number; hash: string } {
    // TypeScript implementation
    const result = this.sha1.calculateHash(message);

    // Convert hash result to seed
    const h0 = BigInt(this.toLittleEndian32(result.h0));
    const h1 = BigInt(this.toLittleEndian32(result.h1));
    
    // 64bit値を構築
    const lcgSeed = (h1 << 32n) | h0;

    // 64bit演算
    const multiplier = 0x5D588B656C078965n;
    const addValue = 0x269EC3n;
    
    const seed = lcgSeed * multiplier + addValue;
    
    // 上位32bitを取得
    return {seed: Number((seed >> 32n) & 0xFFFFFFFFn), hash: SHA1.hashToHex(result.h0, result.h1, result.h2, result.h3, result.h4)};
  }

  /**
   * Parse and validate target seed input
   */
  public parseTargetSeeds(input: string): { validSeeds: number[]; errors: { line: number; value: string; error: string }[] } {
    const lines = input.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    const validSeeds: number[] = [];
    const errors: { line: number; value: string; error: string }[] = [];
    const seenSeeds = new Set<number>();

    lines.forEach((line, index) => {
      try {
        // Remove 0x prefix if present
        let cleanLine = line.toLowerCase();
        if (cleanLine.startsWith('0x')) {
          cleanLine = cleanLine.substring(2);
        }

        // Validate hex format
        if (!/^[0-9a-f]{1,8}$/.test(cleanLine)) {
          errors.push({
            line: index + 1,
            value: line,
            error: 'Invalid hexadecimal format. Expected 1-8 hex digits.'
          });
          return;
        }

        const seedValue = parseInt(cleanLine, 16);
        
        // Check for duplicates
        if (seenSeeds.has(seedValue)) {
          return; // Skip duplicates silently
        }
        
        seenSeeds.add(seedValue);
        validSeeds.push(seedValue);
      } catch (err) {
        // エラー内容を記録
        const msg = err instanceof Error ? err.message : String(err);
        errors.push({
          line: index + 1,
          value: line,
          error: msg || 'Failed to parse as hexadecimal number.'
        });
      }
    });

    return { validSeeds, errors };
  }

  /**
   * Get VCount value with offset handling for BW2
   */
  public getVCountForTimer0(params: ROMParameters, timer0: number): number {
    // 新しいvcountTimerRanges構造を使用
    for (const [vcount, timer0Min, timer0Max] of params.vcountTimerRanges) {
      if (timer0 >= timer0Min && timer0 <= timer0Max) {
        return vcount;
      }
    }

    // フォールバック: 最初のVCOUNT値を返す
    return params.vcountTimerRanges.length > 0 ? params.vcountTimerRanges[0][0] : 0x60;
  }
}