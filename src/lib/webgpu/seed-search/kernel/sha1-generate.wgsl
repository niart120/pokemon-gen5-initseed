const WORKGROUP_SIZE : u32 = WORKGROUP_SIZE_PLACEHOLDERu;

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

@group(0) @binding(0) var<storage, read> state : DispatchState;
@group(0) @binding(1) var<uniform> constants : SearchConstants;
@group(0) @binding(2) var<storage, read> target_seeds : TargetSeedBuffer;
@group(0) @binding(3) var<storage, read_write> output_buffer : MatchOutputBuffer;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
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
    let date_word = (to_bcd(year_mod) << 24u) | (to_bcd(month) << 16u) | (to_bcd(day) << 8u) | to_bcd(day_of_week);
    let is_pm = (constants.hardware_type <= 1u) && (hour >= 12u);
    let pm_flag = select(0u, 1u, is_pm);
    let time_word = (pm_flag << 30u) | (to_bcd(hour) << 24u) | (to_bcd(minute) << 16u) | (to_bcd(second) << 8u);

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
}