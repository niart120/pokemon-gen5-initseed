/* eslint-disable no-console */
/**
 * Development-only verification logic
 * Separate from production code to maintain clean architecture
 */

import { InitializationResult } from '@/lib/initialization/app-initializer';

export async function runDevelopmentVerification(_initResult: InitializationResult): Promise<void> {
  if (import.meta.env.MODE === 'production') {
    return;
  }

  const enableVerboseVerification = import.meta.env.VITE_ENABLE_VERBOSE_VERIFICATION !== 'false';

  if (!enableVerboseVerification) {
    console.log('Development verification disabled (minimal mode).');
    return;
  }

  console.log('Development verification utilities are not available.');
}
