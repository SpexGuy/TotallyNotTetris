pub fn arrayPtr(ptr: var) [*]@typeOf(ptr).Child {
    return @ptrCast([*]@typeOf(ptr).Child, ptr);
}
