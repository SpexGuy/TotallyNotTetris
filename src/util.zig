const std = @import("std");
const assert = std.debug.assert;

fn ArrayPtrType(comptime ptrType: type) type {
    const info = @typeInfo(ptrType);
    switch (info) {
        .Pointer => |ptrInfo| {
            assert(!ptrInfo.is_volatile);
            assert(!ptrInfo.is_allowzero);
            return if (ptrInfo.is_const) [*]const ptrInfo.child else [*]ptrInfo.child;
        },
        else => @compileError("arrayPtr can only operate on a pointer type!"),
    }
}

fn SingleSliceType(comptime ptrType: type) type {
    const info = @typeInfo(ptrType);
    switch (info) {
        .Pointer => |ptrInfo| {
            assert(!ptrInfo.is_volatile);
            assert(!ptrInfo.is_allowzero);
            return if (ptrInfo.is_const) []const ptrInfo.child else []ptrInfo.child;
        },
        else => @compileError("singleSlice can only operate on a pointer type!"),
    }
}

pub fn arrayPtr(ptr: var) ArrayPtrType(@typeOf(ptr)) {
    return @ptrCast(ArrayPtrType(@typeOf(ptr)), ptr);
}

pub fn singleSlice(ptr: var) SingleSliceType(@typeOf(ptr)) {
    return arrayPtr(ptr)[0..1];
}

pub fn emptySlice(comptime T: type) []T {
    return ([_]T{})[0..0];
}
