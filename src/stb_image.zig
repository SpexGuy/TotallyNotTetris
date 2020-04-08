extern fn stbi_load(filename: [*:0]const u8, x: *i32, y: *i32, channels_in_file: *i32, desired_channels: i32) ?[*]u8;
extern fn stbi_failure_reason() ?[*:0]const u8;
extern fn stbi_image_free(retval_from_stbi_load: [*]u8) void;

pub const Image = struct {
    width: i32,
    height: i32,
    channels: i32,
    data: [*]u8,
};
pub fn load(filename: [*:0]const u8) !Image {
    return loadWithChannels(filename, 0);
}

pub fn loadWithChannels(filename: [*:0]const u8, desired_channels: i32) !Image {
    var width: i32 = undefined;
    var height: i32 = undefined;
    var channels: i32 = undefined;
    var data = stbi_load(filename, &width, &height, &channels, desired_channels);
    if (data) |dataPtr| {
        return Image{
            .width = width,
            .height = height,
            .channels = channels,
            .data = dataPtr,
        };
    } else return error.StbiFailure;
}

pub const failure_reason = stbi_failure_reason;
pub const image_free = stbi_image_free;
