const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908B0DF;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7FFFFFFF;

/// Minimal MT19937 implementation for deterministic IV RNG derivation.
pub struct Mt19937 {
    index: usize,
    state: [u32; N],
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { index: N, state }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }

        let mut y = self.state[self.index];
        self.index += 1;

        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;

        y
    }

    fn twist(&mut self) {
        for i in 0..N {
            let x = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut x_a = x >> 1;
            if (x & 1) != 0 {
                x_a ^= MATRIX_A;
            }
            self.state[i] = self.state[(i + M) % N] ^ x_a;
        }
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Mt19937;

    #[test]
    fn mt19937_matches_reference_sequence() {
        let mut mt = Mt19937::new(5489);
        let expected = [
            3499211612, 581869302, 3890346734, 3586334585, 545404204, 4161255391, 3922919429,
            949333985, 2715962298, 1323567403,
        ];

        for &value in expected.iter() {
            assert_eq!(mt.next_u32(), value);
        }
    }
}
