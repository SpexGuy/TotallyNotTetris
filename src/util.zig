const std = @import("std");
const assert = std.debug.assert;

/// Takes a pointer type like *T, *const T, *align(4)T, etc,
/// returns the pointer type *[1]T, *const [1]T, *align(4) [1]T, etc.
fn ArrayPtrType(comptime ptrType: type) type {
    comptime {
        // Check that the input is of type *T
        var info = @typeInfo(ptrType);
        assert(info == .Pointer);
        assert(info.Pointer.size == .One);
        assert(info.Pointer.sentinel == null);

        // Create the new value type, [1]T
        const arrayInfo = std.builtin.TypeInfo{
            .Array = .{
                .len = 1,
                .child = info.Pointer.child,
                .sentinel = @as(?info.Pointer.child, null),
            },
        };

        // Patch the type to be *[1]T, preserving other modifiers
        const singleArrayType = @Type(arrayInfo);
        info.Pointer.child = singleArrayType;
        // also need to change the type of the sentinel
        // we checked that this is null above so no work needs to be done here.
        info.Pointer.sentinel = @as(?singleArrayType, null);
        return @Type(info);
    }
}

pub fn arrayPtr(ptr: var) ArrayPtrType(@TypeOf(ptr)) {
    return @as(ArrayPtrType(@TypeOf(ptr)), ptr);
}

pub fn emptySlice(comptime T: type) []T {
    return ([_]T{})[0..0];
}
