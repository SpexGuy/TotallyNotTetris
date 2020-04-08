const std = @import("std");

pub const Color3f = extern struct {
    r: f32,
    g: f32,
    b: f32,

    pub fn init(r: f32, g: f32, b: f32) Color3f {
        return Color3f{
            .r = r,
            .g = g,
            .b = b,
        };
    }
};
