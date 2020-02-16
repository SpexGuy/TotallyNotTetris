extern fn stbi_load(filename: [*]const u8, x: *i32, y: *i32, channels_in_file: *u32, desired_channels: u32) ?[*]u8;
extern fn stbi_failure_reason() ?[*]const u8;
extern fn stbi_image_free(retval_from_stbi_load: [*]u8) void;

pub const load = stbi_load;
pub const failure_reason = stbi_failure_reason;
pub const image_free = stbi_image_free;
