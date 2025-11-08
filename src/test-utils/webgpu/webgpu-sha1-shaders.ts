const buildSha1ShaderSource = (workgroupSize: number): string => {
  return /* wgsl */ `
struct Config {
  message_count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(2) var<uniform> config : Config;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
}

@compute @workgroup_size(${workgroupSize})
fn sha1_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index >= config.message_count) {
    return;
  }

  let input_base = index * 16u;
  var w : array<u32, 16>;

  for (var i = 0u; i < 16u; i = i + 1u) {
    w[i] = input_words[input_base + i];
  }

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

  let output_base = index * 5u;
  output_words[output_base] = h0;
  output_words[output_base + 1u] = h1;
  output_words[output_base + 2u] = h2;
  output_words[output_base + 3u] = h3;
  output_words[output_base + 4u] = h4;
}
`;
};

const buildGeneratedSha1ShaderSource = (workgroupSize: number): string => {
  return /* wgsl */ `
struct GeneratedConfig {
  message_count : u32,
  base_offset : u32,
  range_seconds : u32,
  timer0_min : u32,
  timer0_count : u32,
  vcount_min : u32,
  vcount_count : u32,
  start_second_of_day : u32,
  start_day_of_week : u32,
  mac_lower : u32,
  data7_swapped : u32,
  key_input_swapped : u32,
  hardware_type : u32,
  nazo0 : u32,
  nazo1 : u32,
  nazo2 : u32,
  nazo3 : u32,
  nazo4 : u32,
  start_year : u32,
  start_day_of_year : u32,
};

const MONTH_LENGTHS_COMMON : array<u32, 12> = array<u32, 12>(
  31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);
const MONTH_LENGTHS_LEAP : array<u32, 12> = array<u32, 12>(
  31u, 29u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);

@group(0) @binding(0) var<storage, read> config : GeneratedConfig;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
}

fn swap32(value : u32) -> u32 {
  return ((value & 0x000000FFu) << 24u) |
    ((value & 0x0000FF00u) << 8u) |
    ((value & 0x00FF0000u) >> 8u) |
    ((value & 0xFF000000u) >> 24u);
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

@compute @workgroup_size(${workgroupSize})
fn sha1_generate(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let local_index = global_id.x;
  if (local_index >= config.message_count) {
    return;
  }

  let safe_range_seconds = max(config.range_seconds, 1u);
  let safe_vcount_count = max(config.vcount_count, 1u);
  let messages_per_vcount = safe_range_seconds;
  let messages_per_timer0 = messages_per_vcount * safe_vcount_count;

  let message_index = config.base_offset + local_index;
  let timer0_index = message_index / messages_per_timer0;
  let remainder_after_timer0 = message_index - timer0_index * messages_per_timer0;
  let vcount_index = remainder_after_timer0 / messages_per_vcount;
  let second_offset = remainder_after_timer0 - vcount_index * messages_per_vcount;

  let timer0 = config.timer0_min + timer0_index;
  let vcount = config.vcount_min + vcount_index;

  let total_seconds = config.start_second_of_day + second_offset;
  let day_offset = total_seconds / 86400u;
  let seconds_of_day = total_seconds - day_offset * 86400u;

  let hour = seconds_of_day / 3600u;
  let minute = (seconds_of_day % 3600u) / 60u;
  let second = seconds_of_day % 60u;

  var year = config.start_year;
  var day_of_year = config.start_day_of_year + day_offset;
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

  let day_of_week = (config.start_day_of_week + day_offset) % 7u;
  let year_mod = year % 100u;
  let date_word = (to_bcd(year_mod) << 24u) | (to_bcd(month) << 16u) | (to_bcd(day) << 8u) | to_bcd(day_of_week);
  let is_pm = (config.hardware_type <= 1u) && (hour >= 12u);
  let pm_flag = select(0u, 1u, is_pm);
  let time_word = (pm_flag << 30u) | (to_bcd(hour) << 24u) | (to_bcd(minute) << 16u) | (to_bcd(second) << 8u);

  var w : array<u32, 16>;
  w[0] = config.nazo0;
  w[1] = config.nazo1;
  w[2] = config.nazo2;
  w[3] = config.nazo3;
  w[4] = config.nazo4;
  w[5] = swap32((vcount << 16u) | timer0);
  w[6] = config.mac_lower;
  w[7] = config.data7_swapped;
  w[8] = date_word;
  w[9] = time_word;
  w[10] = 0u;
  w[11] = 0u;
  w[12] = config.key_input_swapped;
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

  let output_base = local_index * 5u;
  output_words[output_base] = h0;
  output_words[output_base + 1u] = h1;
  output_words[output_base + 2u] = h2;
  output_words[output_base + 3u] = h3;
  output_words[output_base + 4u] = h4;
}
`;
};

export { buildSha1ShaderSource, buildGeneratedSha1ShaderSource };
