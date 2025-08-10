/**
 * Gender determination utilities (Gen 5 semantics)
 *
 * Unified rule:
 * - gender_value < female_threshold -> Female
 * - otherwise -> Male
 * - Genderless / Fixed genders handled by spec
 */

export type GenderSpec =
  | { type: 'genderless' }
  | { type: 'fixed'; fixed: 'male' | 'female' }
  | { type: 'ratio'; femaleThreshold: number };

export type UnifiedGender = 'Male' | 'Female' | 'Genderless';

// Accept any compatible generated spec shape as well
type CompatibleSpec = { type: 'genderless' | 'fixed' | 'ratio'; fixed?: 'male' | 'female'; femaleThreshold?: number };

export function determineGenderFromSpec(genderValue: number, spec: CompatibleSpec): UnifiedGender {
  if (spec.type === 'genderless') return 'Genderless';
  if (spec.type === 'fixed') return spec.fixed === 'male' ? 'Male' : 'Female';

  // ratio
  const threshold = Math.max(0, Math.min(255, Math.floor(spec.femaleThreshold ?? 0)));
  return genderValue < threshold ? 'Female' : 'Male';
}
